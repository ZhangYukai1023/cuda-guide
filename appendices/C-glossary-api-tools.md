# 附录 C：术语、API 与工具命令速查

命令中的二进制名与目标路径要换成当前章节实际值；先用 `command -v` 确认工具可用，并记录 `--version`/`-v`。此表是检索入口，具体参数、支持 GPU 和输出格式以本机 Toolkit 与官方文档为准。

## C.1 执行与内存术语

| 术语 | 本书中的含义 | 易混点 |
| --- | --- | --- |
| Host / Device | CPU 侧代码与 GPU 侧执行/内存 | `float*` 的 C++ 类型不表示它在哪一侧分配 |
| Thread / Block / Grid | 单线程、协作线程块、一次 kernel 的块网格 | `__syncthreads()` 仅同步同一 block，不同步全 grid |
| Warp / SM | 通常 32 线程的执行分组 / 流多处理器 | 分支分歧、寄存器与占用率依 GPU 架构和 kernel 而变 |
| Global / Shared / Register | GPU 设备内存、块共享内存、线程私有寄存器 | Shared 有块作用域；寄存器溢出可产生 local memory 访问 |
| Coalescing | 相邻线程的访问能合并为高效内存交易 | 并不要求逻辑矩阵必须列优先；看实际地址与线程映射 |
| H2D / D2H / P2P | 主机到设备、设备到主机、设备间传输 | API “Async” 不保证 pageable Host 缓冲一定与计算重叠 |
| Stream / Event | 提交队列 / 设备时间点与依赖工具 | 不同 Stream 的相互依赖需要显式建立 |
| Occupancy | 实际或理论活跃 warp/SM 的比例 | 高占用率不是性能目标本身；还要看访存/指令瓶颈 |
| Warmup | 正式计时前运行以准备上下文、JIT/缓存等 | 预热次数和是否含输入传输都要记录 |
| Tolerance | 数值比较允许的误差 | 需说明绝对/相对、零附近、NaN/Inf 和边界子集 |

## C.2 常用 CUDA Runtime 与库 API

| 操作 | 常用 API | 核对重点 |
| --- | --- | --- |
| 枚举/选择设备 | `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaSetDevice` | 当前设备、逻辑到物理映射、架构 |
| 分配/释放 | `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost` | 字节数、配对释放、异步生命周期 |
| 复制/清零 | `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaMemcpyPeer` | 方向、长度、Stream、完成依赖 |
| 启动/错误 | `kernel<<<grid,block,shared,stream>>>`, `cudaGetLastError`, `cudaStreamSynchronize` | 即时启动错误与异步执行错误分开查 |
| 计时 | `cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime` | 相同 Stream/阶段、预热与重复 |
| 稠密线代 | cuBLAS `cublasSgemm`, `cublasSgemv` 等 | 行列主序、leading dimension、计算/累加精度 |
| 稀疏 | cuSPARSE `cusparseSpMV` 等 | CSR 描述符、索引类型、工作区与 Stream |
| 求解 | cuSOLVER LU/QR 等 | 矩阵是否被覆盖、`devInfo`、残差 |
| FFT | cuFFT 计划/执行 API | 变换方向、R2C 布局、工作区、逆变换归一化 |
| 多 GPU/多节点 | `cudaDeviceCanAccessPeer`, MPI, NCCL | 实际拓扑、消息顺序、通信进度与设备映射 |

各库接口可能随着 Toolkit 版本演进。正式交付应在代码和验证记录中写明已使用的具体函数、库版本、数据布局与精度，而不是只记“用了 cuBLAS”。

## C.3 构建、正确性与性能工具

```bash
nvcc --version
nvidia-smi -L
nvidia-smi topo -m
cmake -S chapters/ch33-stencil-heat/examples -B build/ch33 \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/ch33 -j
ctest --test-dir build/ch33 --output-on-failure
compute-sanitizer --tool memcheck ./build/ch33/stencil_heat
nsys profile --trace=cuda,nvtx -o build/ch33/timeline ./build/ch33/stencil_heat
ncu --set full -o build/ch33/kernel ./build/ch33/stencil_heat
cuobjdump --dump-sass build/ch33/stencil_heat
```

`compute-sanitizer --tool memcheck` 重点找越界、未对齐和部分 API/泄漏问题；racecheck/initcheck/synccheck 可按故障类型另跑，不能用性能 profiler 代替正确性检查。[Compute Sanitizer 官方手册](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)描述其支持项和限制。`nsys` 展示 CPU API、Stream、传输和 kernel 时间线，适合找串行化及端到端瓶颈；`ncu` 采集单 kernel 指标，可能让程序变慢或重放 kernel，不能直接把 profiler 下的墙钟时间当正常运行时间。[Nsight Systems 命令说明](https://docs.nvidia.com/nsight-systems/UserGuide/)与 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)应与本机版本对照。`cuobjdump` 检查当前二进制包含的 SASS，不能从它独自判断实际吞吐。

## C.4 最小记录

每次测试至少保存：Git SHA/未提交改动、GPU 名称及计算能力、驱动/Toolkit/nvcc、CMake 目标架构与编译选项、输入形状/种子/文件哈希、正确性容差及最大误差、测试命令/退出码、原始 stdout/stderr。性能实验另需预热、重复次数、各阶段时间和统计量。`PASS` 必须对应真实执行，`Skipped`、缺工具、编译失败和未运行是不同状态。
