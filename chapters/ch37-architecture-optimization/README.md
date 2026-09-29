# 第 37 章：架构相关的高级优化（选读）

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch36-distributed-communication/README.md)

本章保留一个在普通 CUDA GPU 上可运行的三点 stencil，再给支持条件满足的设备增加 `cp.async` 全局内存到共享内存搬运。[async_stencil.cu](examples/async_stencil.cu) 对小/大两种尺寸做 CPU 参考与两版互比，打印寄存器/共享内存资源与重复 kernel 耗时。本章已在 zyk 用 nvcc 12.8 的 sm_120 目标独立编译并运行；通用与 cp.async 路径均通过 CPU 对照，资源和原始计时见[验证记录](results/validation.md)。这些时间不构成稳定加速结论。TMA、线程块集群和 Warp Specialization 在本章解释适用条件，**没有伪装成已实现的可运行路径**。

## 1. 从通用实现与编译报告出发

输入长 `N`，两端为零填充。CPU 和普通 GPU 都计算 `out[i]=0.25 in[i-1]+0.5 in[i]+0.25 in[i+1]`。内存实际布局是 `padded[0]=0`、`padded[i+1]=in[i]`、`padded[N+1]=0`，尾部另外保留两个安全填充元素；输出不再更新，因此这是一次滤波/stencil，不是第 33 章的多时间步热方程。`N` 取 1024 和 `2^18`，都是 256 的整数倍，用以观察工作量变化时固定同步/搬运开销的占比。示例的输入函数、边界和输出在两版完全相同。

在已有通用 kernel 正确后，先看编译器资源报告：CMake 给 CUDA 编译加 `-Xptxas=-v`，运行时 `cudaFuncGetAttributes` 打印 `numRegs`、静态共享内存字节与二进制目标版本。可选用 `cuobjdump`/`nvdisasm` 检查实际 SASS，再结合 Nsight Compute 记录寄存器占用、访存交易、缓存命中和 stall 原因。PTX 是中间表示，不保证设备最终执行相同指令序列；真正的性能假设应以**该 GPU、该 Toolkit、该编译选项**下的 SASS 和测量为准。仅观察“共享内存更多”或“指令更先进”不能推断提速。

## 2. `cp.async` 条件、对齐与同步

PTX 的普通 `cp.async.ca.shared.global` 需要 `sm_80+`。示例每个 256 线程块处理 256 个输出，先把 260 个 `float`（包含两端 Halo 与 16 字节搬运所需尾部填充）放入 16 字节对齐的共享内存。前 65 个线程各发一次 16 字节 `cp.async`，目标与源都保持 16 字节对齐：`cudaMalloc` 基址对齐、块起点相隔 256 个 float、每个参与线程前进 4 个 float。额外读取的最后两个 float 仅在已分配的零填充区域，绝不能将“输出 N 个”误当作可安全读取 N+4 个未经分配的元素。

每个线程用 `cp.async.commit_group` 提交自己的拷贝组，再以 `cp.async.wait_group 0` 等本线程的异步拷贝完成；随后**整个线程块**执行 `__syncthreads()`，让所有线程可以读取其他线程搬进共享内存的值。`wait_group` 只针对调用线程自己的异步组，单靠它不能代替块内的跨线程同步。少于 65 个发起拷贝的线程也会走相同的提交/等待/块同步路径，不能在 barrier 前提前返回。代码通过编译期 `__CUDA_ARCH__ >= 800` 编译 PTX 路径，并在运行时同时检查设备 compute capability 与函数 `binaryVersion`；不满足时只执行通用路径，明确打印 `SKIP cp.async`。

此例是一轮“发起搬运→等待→计算”。它演示异步拷贝的**正确同步**，并没有把下一 tile 的搬运与当前 tile 计算重叠，因此不是多级流水线，可能比简单的全局加载更慢。真正的 pipeline 需要至少两组缓冲、阶段生产/消费顺序、正确的 barrier 到达与等待、有限的在途组数量，以及寄存器和共享内存占用控制。Warp Specialization 进一步分工为搬运 warp 与计算 warp，只有任务规模和算力/内存重叠足够时才可能获益；若生产消费信号顺序错误，容易死锁或读到未完成数据。

TMA（Tensor Memory Accelerator）是计算能力 9.0 引入的另一类大块/多维异步搬运，使用的 bulk 异步组和 barrier 语义与本例 `cp.async` 不同，不能简单把指令名替换。线程块集群也从 9.0 引入，并需要按实际设备、MIG/资源约束确认可用集群大小。即使 zyk 的计算能力高于阈值，是否需要 TMA/集群仍取决于张量形状、对齐、布局、共享内存与线程块协作；本章只列选择条件，没有在未验证环境下提供假想性能结论。相关官方依据见 [CUDA 异步拷贝说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)、[PTX `cp.async` 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)、[CUDA 集群说明](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。

## 3. 构建、检查与测量

在远程仓库根目录：

```bash
cmake -S chapters/ch37-architecture-optimization/examples -B build/ch37 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch37 -j --verbose
ctest --test-dir build/ch37 --output-on-failure
./build/ch37/async_stencil
# 如本机安装 cuobjdump，再检查 SASS：
cuobjdump --dump-sass build/ch37/async_stencil
```

`CMAKE_CUDA_ARCHITECTURES=120` 是 zyk 此前使用的目标架构；若实际 GPU/Toolkit 不支持，先从 `nvidia-smi`、`nvcc --version` 和 GPU 属性确认支持的 `sm_XX`，再调整构建目标，而不是猜测或升级驱动。运行时两版先比较 CPU 最大绝对误差（源码暂设 `2e-6`）；`cp.async` 路径可用时再检查两版差异。每种 kernel 预热 8 次、重复 100 次，用 CUDA event 报告单次平均值和**观察到的** `generic_ms/async_ms`。不含上传下载、资源创建和多次独立测量，不能当作稳定的端到端加速比。若要比较收益，分别保存多轮分布/中位数、两种尺寸、编译报告和 SASS，检查运行频率与其他负载，并在不同 tile/边界/数据大小下复测。任何 `cp.async` 版不占优的结果都应如实记录。

## 4. 常见错误

- 在 `sm_80` 以下设备或不含对应目标的二进制中执行 `cp.async` 路径，或仅凭设备型号猜测编译目标。
- 16 字节拷贝的源/目的地址未对齐，尾部 Halo 后的填充未分配，导致错位或越界。
- 发起 `cp.async` 后立即读共享内存，缺少 `wait_group`；仅等待自己的一组却没有块级 `__syncthreads()`。
- 条件分支中部分线程绕过 `__syncthreads()`，造成未定义行为或挂起。
- 将普通 `cp.async` 的完成机制与 TMA 的 bulk 异步/事务 barrier 混用。
- 只对一个大数组测速而不核对 CPU 结果、边界和小输入；或只看一次 event 平均值就宣称架构优化普遍有效。
- 误把本章一轮 stencil 的异步搬运称为多阶段 pipeline 或 Warp Specialization。

## 5. 练习与参考答案

1. 256 个输出、左右各需一个邻居，为何共享数组分配 260 而不是 258 个 float？答：逻辑输入需要 258 个；每次拷 4 个 float，总拷贝元素要向上补到 4 的倍数，即 260，还必须在全局内存分配对应的安全填充。
2. 65 个线程发拷贝，其他线程没有发拷贝，为什么仍都要到 `__syncthreads()`？答：所有后续读共享内存的线程都要等待整个块的拷贝完成；条件提前返回会破坏 barrier 的参与集合。
3. `cp.async.wait_group 0` 能让线程 A 看到线程 B 的拷贝结果吗？答：它只等待 A 的异步组；还需块级同步使 B 完成并让 A 安全读 B 写的共享位置。
4. 若小输入 `cp.async` 比通用版慢，是否说明实现一定错？答：不一定；固定同步与搬运成本可能大于节省的全局加载，先检查两版数值，再看 SASS、资源占用和重复测量。
5. 怎样把本例改成真正的流水线？答：需要至少双缓冲和多个 tile，让下一 tile 的异步搬运与当前 tile 的计算重叠，按阶段提交/等待并控制每组生命周期；还要验证边界、同步与资源占用，不能简单去掉 `wait_group`。

[下一章](../ch38-integrated-projects/README.md) 把各专题整合成完整项目，按接口、正确性、故障与性能的顺序交付。
