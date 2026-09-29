# 第 36 章：多机通信与分布式计算

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch35-multi-gpu/README.md)

本章提供三个独立层次：无需 MPI 的 [行分区自检](examples/partition_selftest.cpp)、用 MPI 交换 GPU 热传导 Halo 的 [mpi_halo_heat.cu](examples/mpi_halo_heat.cu)，以及 MPI 引导 NCCL 的 [nccl_collectives.cu](examples/nccl_collectives.cu)。MPI 默认使用显式主机中转；CUDA-aware MPI 模式需环境确实支持才可启用。CMake 在缺少 MPI/NCCL 时只构建可用目标，**不会安装依赖**。纯 C++ 行分区自检已在 zyk 独立构建并通过；当前环境没有 MPI C++ 和 NCCL 开发依赖，MPI/NCCL CUDA 源码未编译或运行。只有一张可见 GPU，多节点路径也未实测，见[验证记录](results/validation.md)。

## 1. Rank、设备与集合通信

MPI rank 是参与通信的进程编号。示例通过 `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` 取得同一节点内的 `local_rank`，再把它映射到本节点可见的 GPU 编号；每个 rank 管理一张 GPU。若某节点可见 GPU 数小于其本地 rank 数，程序返回 77 表示跳过。实际作业调度器可能给每个进程分别设置 `CUDA_VISIBLE_DEVICES`，使每个进程只见到自己的设备 0；那种配置需要把映射规则改为“每进程选设备 0”，不能直接套本例的 `local_rank`。必须记录 rank、节点名、可见设备及物理 GPU UUID，避免两个 rank 无意压在同一 GPU 上。

NCCL 示例由 rank 0 创建 `ncclUniqueId`，经 MPI `Bcast` 发送给所有 rank，然后每个 rank 调用 `ncclCommInitRank`。三个集合操作依次入同一 CUDA stream，再同步并对照解析公式：

| 操作 | 每 rank 输入 | 每 rank 输出 | 验证要点 |
| --- | --- | --- | --- |
| AllReduce | 长 8、值由 rank 和索引确定 | 长 8、各 rank 同一求和结果 | 每个元素等于所有 rank 的对应元素之和 |
| AllGather | 长 4、本 rank 的片段 | 长 `4*ranks`、按 rank 顺序拼接 | 片段次序和偏移正确 |
| ReduceScatter | 长 `4*ranks` | 长 4、归约后第 `rank` 块 | NCCL 的 count 是**每 rank 接收元素数** |

CUDA `stream` 中的 NCCL 调用排队后仍需同步才能读结果。MPI 与 NCCL 在同一进程混用时，所有 rank 的集合调用顺序、count、dtype、op、communicator 必须匹配；一个 rank 抛异常或提前退出，其他 rank 可能挂在通信中，示例在致命异常时 `MPI_Abort`。长时间作业还应增加超时、NCCL 异步错误查询、作业调度器清理和网络故障恢复。NCCL AllReduce 适合梯度等设备向量归约，AllGather 适合收集分片，ReduceScatter 适合归约后继续分片计算；它们不自动替代任意邻域 Halo 的消息匹配。

## 2. 行分区热传导和 Halo

全局网格 32×24，固定零温 Dirichlet 边界，中间热矩形初始为 1，二维五点显式热扩散运行 40 步，`r=0.2<=1/4`。每个 rank 持有连续的若干行，另外在设备缓冲两端各留一行 ghost/Halo。第 `rank` 个分区拥有 `height/ranks` 行，再把余数前 `height%ranks` 行各加一行；`partition_selftest` 在 2、3、5、7 个 rank 情形检查 24 行恰好覆盖一次。

每步先完成前一步 GPU kernel，再交换当前分区首尾的**旧时间层**：上方 rank 收到本 rank 第一行作为其下 Halo，本 rank 从上方收到其末行作为上 Halo；下方类似，MPI 标签 10/20 使方向匹配。`MPI_Sendrecv` 使用明确邻居与两个调用顺序；边界邻居使用 `MPI_PROC_NULL`，外层 Halo 保持 0。然后本 rank 的 GPU kernel 只写自己拥有的行，并交换两块设备缓冲。这样跨 rank 同样保持双缓冲时间层一致；在 Halo 未到达前不能计算依赖边界的格点。

默认模式将设备边界行拷到 Host，MPI 在 Host 缓冲上交换，再上传 ghost 行，适用于常规 MPI。仅当 MPI 构建和传输栈明确支持设备指针时，才传 `--cuda-aware`，此时 `MPI_Sendrecv` 直接拿设备缓冲指针。是否真的经 GPUDirect RDMA、是否内部仍经 Host 中转，需要 MPI 能力查询和实际性能/拓扑证据，不能由“接受设备指针”直接推出。示例为清楚起见每步同步和阻塞通信，尚未实现计算/通信重叠；优化时可以先算不依赖 Halo 的内部行，同时非阻塞交换边界，然后等待并算分区边缘，但必须维护流事件、缓冲生命周期和消息顺序。

最终每个 rank 下载自己的行，用 `MPI_Allreduce` 求全场温度和，用 `MPI_Gatherv` 在 rank 0 重建全场。rank 0 用 CPU 双精度执行**同一**初始/边界/时间步问题，逐格比较最大误差和全局和；所有 rank 接收验证结论。`--require-two-nodes` 会检查不同节点名至少两个；仅同一节点双卡通过可验证跨进程通信，不能算多节点验收。

## 3. 构建与运行

在远程仓库根目录：

```bash
cmake -S chapters/ch36-distributed-communication/examples -B build/ch36 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch36 -j
ctest --test-dir build/ch36 --output-on-failure
./build/ch36/partition_selftest
```

只有 CMake 报告 `MPI_CXX_FOUND` 并生成相应目标后，才运行 MPI 路径；NCCL 目标还需 `nccl.h` 和 `libnccl`。两 GPU 在同一节点上的功能测试可用 `mpirun -np 2 ./build/ch36/mpi_halo_heat` 与 `mpirun -np 2 ./build/ch36/nccl_collectives`，前提是两个 rank 分别有可访问的 GPU。多节点时根据集群调度器提供的 hostfile/分配启动两个 rank，再运行 `mpi_halo_heat --require-two-nodes`；若 MPI 已验证 CUDA-aware，可额外运行 `mpi_halo_heat --cuda-aware --require-two-nodes`。不要在不支持设备指针的 MPI 上盲目传 `--cuda-aware`，那可能崩溃或卡住。MPI/NCCL 缺失或只有一张 GPU 时如实记录 `未构建`/`Skipped`，并保留默认分区测试结果。

单次输出中的 `slowest_rank_seconds` 是 40 步阶段各 rank 的最大耗时，包含每步同步与 Halo 交换，不含初始上传和最后全场 Gather。性能报告至少要分离计算、通信、同步、初始化与最终收集，预热并重复。强扩展固定全局 32×24 问题、增加 rank，效率 `T_1/(P*T_P)`；弱扩展保持每 rank 的局部规模不变、随 rank 增大全局网格。该教学网格非常小，通信主导，不能由其速度推断大型 PDE 的扩展性。NCCL 测试也应增加真实数据量与多次重复，不把三个 4/8 元素集合操作的单次延迟当带宽。

接口和 count 约定可核对 NVIDIA [NCCL 示例](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)和 [NCCL 与 MPI 说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html)。MPI 是否 CUDA-aware 需看实际 MPI 发行版的文档及构建配置。

## 4. 常见错误

- 以全局 rank 当本节点 GPU 编号，导致第二节点 rank 2、3 访问不存在的 GPU 2、3。
- 两个相邻 rank 使用不同消息标签或调用顺序，造成死锁；一个 rank 先退出也可能让其他 rank 永久等待。
- 把新时间层边界行发给邻居，另一边却读取旧时间层；或忘记 `MPI_PROC_NULL` 边界 ghost 为 0。
- 用 `MPI_Gather` 假设各 rank 行数相等；24 行不能被 5、7 个 rank 整除，应使用 `MPI_Gatherv` 的 counts/displacements。
- 常规 MPI 直接接收 CUDA 指针而未确认 CUDA-aware 支持；或认为“支持设备指针”必然表示 RDMA 零拷贝。
- NCCL AllGather/ReduceScatter 的 count 按总元素数填写，导致越界或结果错位。
- 单节点双卡、只有 CPU 分区自检或依赖缺失时的跳过，被记为已通过多节点扩展实验。

## 5. 练习与参考答案

1. 24 行由 5 个 rank 分区，每个 rank 分别有几行？答：前 4 个各 5 行，最后 1 个 4 行；起始行为 0、5、10、15、20。
2. 为什么 Halo 要从 `old` 交换、不能从 `next` 交换？答：同一时间步所有更新都应使用旧层邻居；混合新旧层会改变离散方程并引入顺序依赖。
3. 4 个 rank 的 ReduceScatter，每 rank 要收 4 个 float，NCCL 的 count 填多少，每 rank 输入多长？答：count=4；每 rank 输入 `4*4=16` 个 float。
4. `MPI_Allreduce` 得到全局温度和与 CPU 相同，是否足以证明场正确？答：不足；不同格点可能互相抵消，仍须按正确顺序 Gather 并逐格比较。
5. 同一节点双卡通过 `--require-two-nodes` 吗？答：不会，应返回跳过；它只能证明双进程/双设备功能，不能证明网络通信与跨节点性能。

[下一章](../ch37-architecture-optimization/README.md) 将讨论架构特定优化及版本/硬件门控，在通用路径正确后再尝试异步拷贝与流水线。
