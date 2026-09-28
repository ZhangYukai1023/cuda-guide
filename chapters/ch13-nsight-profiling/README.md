# 第 13 章：用 Nsight 找到瓶颈

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch12-benchmarking/README.md)

第 12 章告诉我们某个范围用了多久，本章用时间线和 kernel 指标解释这些时间去哪了。示例 [profile_cases.cu](examples/profile_cases.cu) 提供相同输入长度的 `tiny`、`batch` 与 `stride` 三种运行模式，均与 CPU 结果比较。有 NVTX3 头文件时，源码加入 NVTX 范围，便于把主机提交阶段与 GPU 工作关联起来。三种模式已在 zyk 上编译并通过 GPU/CPU 对照；当前服务器缺少 NVTX3、Nsight Systems 和 Nsight Compute，因此尚无 NVTX 时间线或 profiler 指标。

## 1. 先用 Nsight Systems 看整条时间线

`tiny` 把 16384 项分成 64 个 256 项分块，每块各做一次 H2D 拷贝、一次 kernel 和一次 D2H 拷贝；主机在每次提交后**故意**休眠 200 微秒，让 CPU 提交空隙在时间线上容易辨认。`batch` 对相同 16384 项各做一次整块 H2D、kernel、D2H，没有故意休眠。两种模式的数值输出相同，任务组织方式不同。

若 NVTX3 可用，`profile_work_including_wait` 覆盖提交到流同步，内部的 `tiny_copies_with_cpu_gaps` / `batched_copy_and_kernel` 标记主机提交代码；每个 `one_tiny_chunk` 范围帮助数清小操作。NVTX 主机范围的结束不自动代表 GPU 操作完成，本例的外层范围包括 `cudaStreamSynchronize`，可在时间线中分清主机与设备的相对位置。注意 200 微秒休眠是教学构造，不能由 `tiny` 与 `batch` 的总时间差推断“仅合并拷贝”的纯收益。

在安装了 NVTX3 头文件且 `nsys` 可用的环境中，重新构建后分别采集：

```bash
nsys profile --trace=cuda,nvtx -o build/nsys-ch13-tiny \
  ./build/outline/book_ch13/profile_cases tiny
nsys profile --trace=cuda,nvtx -o build/nsys-ch13-batch \
  ./build/outline/book_ch13/profile_cases batch
nsys stats build/nsys-ch13-tiny.nsys-rep
nsys stats build/nsys-ch13-batch.nsys-rep
```

打开 `.nsys-rep`，查看 CPU 线程上的 NVTX、CUDA API 调用、GPU 上的 memcpy 与 kernel 条带。按时间顺序回答：GPU 是否在等下一次提交？CPU 是否卡在同步？小拷贝和短 kernel 的数量是否与源码一致？单看总耗时不能回答这些问题。CLI 选项和报告内容以 [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/) 为准。

## 2. 再用 Nsight Compute 检查一个 kernel

`stride` 只启动一次 `gather_stride`，每个输出项从 `(i*17)%16384` 读取输入；`batch` 的 `transform_contiguous` 从相同下标读取。两者都写连续输出，并使用相同输入数据与输出公式。CPU 对照分别计算正确结果。跨步读可能改变内存事务和缓存行为，但不能只凭源码宣布它是“访存受限”；需结合设备上的 profiler 指标与第 12 章的时间。

```bash
ncu --set full -o build/ncu-ch13-stride \
  ./build/outline/book_ch13/profile_cases stride
ncu --set full -o build/ncu-ch13-contiguous \
  ./build/outline/book_ch13/profile_cases batch
```

先看实际被采集的 kernel 名称、Grid/Block、执行时长，再看内存工作负载、有效吞吐、占用和相关警告。Nsight Compute 可能通过重放 kernel 收集指标，分析时会影响运行；采集结果用于诊断，不能直接替代第 12 章不插桩的性能基准。若环境缺少性能计数器权限，记录工具提示与可获取的指标，不把没有采集到的字段填为 0。具体 CLI 选项以 [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 为准。

## 3. 用可证伪的假设推进优化

一次合理的记录可以写成：

> 假设：`tiny` 的 GPU 时间线上有多段空闲，因为主机每块提交之间有 200 微秒间隔。证据：Nsight Systems 中 NVTX `one_tiny_chunk` 与 GPU 操作的相对时间。改动：去掉人为休眠并把 64 次拷贝/启动合成一个批量请求。验证：正确性相同，按第 12 章的同一计时边界与多次样本比较。

这里把“故意休眠”和“很多小操作”分成两个可能原因；要单独验证拷贝批量化的作用，可增加第三种模式：不休眠但仍保留 64 次小调用。没有这个对照时，不应把全部差异归因于拷贝数量。

针对 `stride`，假设可以是“读地址分散使相关内存指标变差”；用 Nsight Compute 的两份报告、相同规模与编译选项检验，再考虑改数据布局。若指标没有支持，就回头看算法规模、缓存命中和其它瓶颈。时间线决定看哪里，kernel 指标帮助解释为什么；两者都要回到业务任务的端到端收益。

## 4. 构建与正确性

本章正确性示例可在缺少 NVTX3 头文件时构建，并输出 `SKIP NVTX ranges`；此时范围标记为空实现。要做时间线实验，需在具有 NVTX3 与 Nsight 的环境中重新构建。CMake 使用静态 CUDA 运行库与平台动态加载库。在仓库根目录：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target profile_cases -j2
ctest --test-dir build/outline -R '^ch13_profile_' --output-on-failure
```

也可独立配置 `chapters/ch13-nsight-profiling/examples`。三种模式的 GPU 结果在程序内逐项与 CPU 参考精确比较，独立与根工程 CTest 均已通过。`nsys`、`ncu` 和 NVTX3 当前不可用，profiler 验证未完成；普通正确性验证与 profiler 验证分别记录。[验证记录](results/validation.md)会保存实际工具输出。

## 5. 常见误区

- 一上来采集所有 kernel 的所有指标，却不知道目标阶段在哪里；先用 Systems 找时间线，再选 Compute 的目标 kernel。
- NVTX 主机范围结束便认定 GPU 已完成；异步提交的完成要看设备时间线或显式同步。
- 把 profiler 插桩下的耗时与未插桩基准直接比较。
- 把可见的 GPU 空闲全部归因于访存慢，而不检查 CPU 提交空隙与同步。
- 一次同时改变休眠、拷贝次数和算法，然后断言其中一个因素造成全部变化。
- 收集了很多指标，却没有写可被下一次实验否定的具体假设。

## 6. 练习与参考答案

1. `tiny` 模式应出现多少次 H2D、kernel、D2H？答：各 64 次；实际采集还要检查驱动是否合并或改变展示方式。
2. `batch` 模式各有几次？答：各 1 次。
3. 若 GPU 时间线有明显空隙，下一步看哪里？答：对齐 CPU 提交、CUDA API、NVTX 和同步事件，确认 GPU 是否在等工作。
4. 若 `stride` 的内存指标变差，但端到端时间几乎不变，能否说业务得到显著优化空间？答：不能；还要看该 kernel 在完整任务中的占比和测量波动。
5. 怎样分离人为休眠与小调用数量的影响？答：增加“64 次小调用但不休眠”的第三个对照，其余输入和编译条件保持一致。

[下一章：访存优化](../ch14-memory-optimization/README.md)用转置实验研究合并访存、对齐与数据重用。
