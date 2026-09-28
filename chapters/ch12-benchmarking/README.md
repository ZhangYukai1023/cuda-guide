# 第 12 章：建立可信的性能基准

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch11-cuda-libraries/README.md)

第 11 章给出库实现的正确性基线。本章开始比较时间，但先统一计时口径。示例 [benchmark_vector.cu](examples/benchmark_vector.cu) 对长度 7、4096、1048576 的 float 向量加法分别记录 kernel、带传输的 GPU 流水段和主机等待的耗时，并与 CPU 循环做同输入对照。示例已在 zyk 编译运行，三组 CPU/GPU 正确性对照通过；测量时 GPU 被其他任务占满，以下计时只作为受干扰的观测，尚不足以得出性能收益结论。实际值见[验证记录](results/validation.md)。

## 1. 一次计时为什么不够

CUDA kernel 启动通常对主机异步返回。若只用 CPU 时钟围住 `add<<<...>>>`，记录的多半是提交时间，不是 GPU 做完工作的时间。示例先完成 CUDA 上下文准备，再对每个尺寸做 5 次预热，随后每种口径采集 20 个样本，打印中位数和第 10、90 百分位附近的样本值。小任务的数值可能接近计时分辨率与提交开销，不能根据几微秒的单次差异宣布优化成功。

所有 Event 都记录在同一条显式创建的 CUDA Stream 上，`stop` Event 完成后再读取 elapsed time。主机墙钟口径则在流同步之后停止。对照验证在计时循环结束后进行，发现错误立即退出。NVIDIA 的 [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)与[异步执行文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)说明了异步启动与 Event 计时的关系。

## 2. 五列数字各代表什么

| 输出字段 | 起止范围 | 包含什么 |
| --- | --- | --- |
| `setup_ms` | 构造一次实验上下文 | 页锁定 host 缓冲区、device 缓冲区、Stream、Event 的创建与输入填充；不含初次 CUDA 上下文创建 |
| `kernel_ms` | 同一 Stream 的两枚 Event | 输入已在设备上的一次 kernel；不含分配和传输 |
| `pipeline_ms` | 两枚 Event 之间 | 两次 H2D、kernel、一次 D2H 的设备流时间；用页锁定 host 缓冲区与异步复制 |
| `wall_ms` | 主机 steady clock | 相同传输与 kernel 的提交以及等流完成的主机等待；复用已建缓冲区 |
| `cpu_ms` | 主机 steady clock | 已准备输入上的 CPU 逐元素加法；输出保留在 host |

`pipeline_ms` 与 `wall_ms` 测的是复用缓冲区的**稳态任务**，并非含文件读取、建上下文与销毁资源的业务全流程。`setup_ms` 单独显示初始化成本，避免把它静默排除。要评估一次性任务，需把与实际应用相同的初始化和清理一起纳入墙钟；不能简单把不同样本的两个中位数相加当作真实一次请求。

输入是以 1/8 和 1/16 为单位的浮点数，CPU 与 GPU 结果逐项比较。`n=7` 检查短任务，`4096` 是中等规模，`2^20` 用来观察较大向量；这三个点本身不足以拟合完整性能曲线。为了使 H2D/D2H 真正可异步提交，本例使用 `cudaHostAlloc` 的页锁定内存；页锁定分配有资源成本，因此记在 `setup_ms`。

## 3. 带宽、吞吐量和算术强度

一次向量加法逻辑上读 A、读 B、写 C，共 `3*n*sizeof(float)=12n` 字节，做 `n` 次加法。若用 kernel Event 的时间 `t_ms` 计算有效数据吞吐：

```text
kernel_effective_GBps = 12*n / (t_ms * 10^6)
算术强度 ≈ n FLOP / (12*n B) = 1/12 FLOP/B
```

这是按**逻辑数据量**得出的有效带宽指标，不是直接测出的 DRAM 物理流量。缓存命中、写策略、计时噪声都可能改变它与硬件标称带宽的关系。Roofline 用算术强度与带宽/计算上限判断潜在瓶颈，但需要正确的工作负载数据流与设备指标；本章不拿单个小实验画出硬件 Roofline，也不推测瓶颈已经确定。

“GPU 加速比”同样须写清分子分母。`cpu_ms/kernel_ms` 只比较 CPU 计算与 GPU kernel，忽略传输；对于一次输入在 CPU、输出也要回 CPU 的任务，应先比较 `cpu_ms` 与包含 H2D/D2H 的 `wall_ms`，并说明设备资源是否可复用。第 13 章再用 profiler 检查哪些阶段是真正的瓶颈。

## 4. 运行与记录

在仓库根目录运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target benchmark_vector -j2
./build/outline/book_ch12/benchmark_vector
ctest --test-dir build/outline -R '^ch12_benchmark_vector$' --output-on-failure
```

可用 `cmake -S chapters/ch12-benchmarking/examples -B build/ch12-benchmark-standalone` 独立构建，仍需指定 CUDA 编译器与架构。保存日期、GPU/驱动、时钟或功耗设置、nvcc 版本、编译选项、GPU 上的其他负载、输入尺寸、预热与重复次数，并保留全部原始样本或至少所用统计量。[验证记录](results/validation.md)保存当前设备上的三次程序运行摘要、负载条件和限制；设备空闲后仍需复测才能形成可信性能基线。

## 5. 常见误区

- 主机只测 kernel 提交而不等待完成，把很小的数字当作计算时间。
- 把普通 pageable host 内存的同步复制与本例 pinned 异步复制混在一张比较表中，却不标注条件。
- 第一轮包含 CUDA 初始化、后续轮次没有；仍把第一轮当稳态中位数。
- 用仅含 kernel 的时间对比包含数据准备的 CPU 全流程，得到没有共同任务边界的加速比。
- 以一次计时或一个尺寸作为性能结论，忽略波动、缓存与其他 GPU 负载。
- 结果没有经过 CPU 对照就开始“优化”，最后加速了一个错误计算。

## 6. 练习与参考答案

1. 若 kernel 0.02 ms、三次传输加 kernel 的任务墙钟 0.5 ms，能否说整体耗时是 0.02 ms？答：不能；对该任务应使用包含所需传输与等待的口径。
2. 长度 `n=1024` 的 float 加法逻辑读写量是多少？答：`12*1024=12288` 字节。
3. 为什么 Event 要在同一 Stream 的 kernel 前后记录？答：这样时间戳按该 Stream 的工作顺序包住目标操作；还须等 stop Event 完成才可读取耗时。
4. 如果小任务 CPU 中位数 0.001 ms、GPU 墙钟 0.04 ms，应先得出什么结论？答：在当前输入、设备和任务边界下，GPU 路径更慢；还需检查计时分辨率与波动，不能外推到大规模任务。
5. 想比较“每次都新建缓冲区”的服务接口，怎么改实验？答：把分配、填充、传输、计算、结果回传和释放全放进每个请求的墙钟区间，再重复采样；保留正确性检查和环境记录。

[下一章：用 Nsight 找到瓶颈](../ch13-nsight-profiling/README.md)使用 Nsight Systems 与 Nsight Compute 在时间线和 kernel 指标中定位瓶颈。
