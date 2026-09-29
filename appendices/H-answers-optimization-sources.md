# 附录 H：练习答案索引、优化记录与官方资料

全书每章末尾都把练习和参考答案放在一起，便于读者做完立即核对。本附录提供跨章节的答案线索与优化记录格式，避免只记“更快”却忘了算法和验证条件。答案属于正文知识点，GPU 上的正确性/速度仍以真实运行结果为准。

## H.1 练习答案导航

| 章节 | 核对练习时的关键结论 |
| --- | --- |
| 1—4：环境、线程、数组、资源 | Host/Device 的位置与生命周期；`grid*block` 覆盖范围和尾块 mask；传输长度按字节计算 |
| 5—8：Warp、内存、共享与归约 | 分歧/合并访存是不同问题；块内 barrier 不能同步全网格；归约要处理非整块与原子顺序 |
| 9—11：数值、调试与库 | CPU 参考同语义；小误差按 dtype/范围判定；库调用必须写清布局与工作区；工具出错先查最早错误 |
| 12—16：基准、Nsight、访存、执行与 GEMM | 区分 kernel 和端到端；单次计时不排名；寄存器/共享内存/occupancy 有取舍；矩阵布局和精度必须一致 |
| 17—20：Stream、Graph、工程与回归 | 异步缓冲在完成前不能复用；Graph/池适合重复任务但有准备成本；构建/测试/部署需记录版本与回归基线 |
| 21—26：图像 | 通道/stride/ROI、边界与半像素坐标决定输出；17×13 半尺寸是 9×7；预览与归一化张量分开验收 |
| 27—30：AI 算子 | 当前 Stream、shape/stride/dtype 是接口；Softmax 减最大值；LayerNorm 用指定方差；Tensor Core/GEMM 比较同精度；Attention mask 与数值稳定性要明确 |
| 31—34：科学计算 | CSR 行偏移末项等于 nnz；CG 检查真残差；FFT 线性卷积需 `N+K-1` 填充和 IFFT 归一化；二维显式热扩散 `r<=1/4`；Monte Carlo 标准误差约 `N^{-1/2}` |
| 35—38：分布式与交付 | 单卡跳过不等于双卡通过；ReduceScatter 的 count 为每 rank 输出数；`cp.async` 的线程组等待后仍要块同步；A/B/C 通过不代表 D 完成 |

例如：8 点序列与 3 点核的完整卷积为 10 项；17×13 图像按半尺寸缩放为 9×7；24 行分给 5 个 MPI rank 为 5、5、5、5、4 行；193 粒子以 128 为 tile 时第二 tile 有 65 个有效粒子。这些都是可以手算的边界样例。遇到具体题目，以对应章节“练习与参考答案”的完整过程和当前源码约定为准；若发现答案与源码语义不一致，应修正文稿并重新跑章节测试。

## H.2 优化记录卡

| 字段 | 每次修改要写的内容 |
| --- | --- |
| 问题与输入 | 尺寸、batch、dtype、stride、数据分布、边界及真实代表性 |
| 原实现 | Git SHA、函数/文件、CPU/库参考、误差与性能基线 |
| 假设 | 是访存、启动、算术、同步、库计划、传输还是负载不均的成本 |
| 单一改动 | tile、布局、融合、Stream/Graph、精度等，只写本次真正改的内容 |
| 正确性 | 同语义逐元素/边界检查、最大误差、超阈值、Sanitizer、失败样本 |
| 资源 | 寄存器、Shared、显式设备字节、可用时的峰值与占用率 |
| 时间 | 预热/重复、kernel 与端到端中位数/p95、CPU 基线与测量边界 |
| 结果 | 哪些尺寸有收益/回退，是否有统计波动，支持的 GPU/Toolkit 条件 |
| 下一步/回退 | 下一次只测试一个假设；不支持或无收益时回到通用路径 |

一个优化版本即使更快，只要边界/精度/失败处理改变，就应列为**不同语义或精度模式**，不能混入旧结果表。多 GPU 还要加设备映射与合并成本，多节点要加网络/拓扑与每 rank 时间。记录图像时按附录 G 保存局部图、差异图和质量指标配置。

## H.3 官方资料索引

| 主题 | 官方入口 | 用途 |
| --- | --- | --- |
| CUDA 编程模型与架构能力 | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | kernel、内存、同步、异步拷贝、集群与功能条件 |
| CUDA C++ 实践 | [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 测量、访存、数据传输和优化方法 |
| 编译 | [nvcc 手册](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) | 架构目标、编译/链接与资源报告 |
| PTX | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | 第 37 章 `cp.async` 指令及同步语义 |
| 稠密/稀疏/求解 | [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)、[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)、[cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) | 数据布局、描述符、工作区、精度和状态码 |
| FFT | [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) | 计划、批处理、R2C/C2R、逆变换归一化 |
| 多 GPU 集合通信 | [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) | rank/communicator、AllReduce/AllGather/ReduceScatter 与故障 |
| 正确性工具 | [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | 越界、未初始化、竞争、同步检查 |
| 时间线/内核分析 | [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) | Host/Stream/传输时间线、单 kernel 指标 |
| PyTorch 扩展 | [PyTorch 自定义 C++/CUDA 算子教程](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) | 第 27 章注册、测试与自动求导约定 |

资料会随版本变化；研究具体 API 时选与当前 Toolkit/库版本相符的官方文档，代码验收仍以实际服务器构建与运行输出为准。官方指南的“支持某功能”不是对本书示例已测试、已获性能收益的证明。
