# 原始大纲覆盖核对

依据 [38 章原始大纲](outline.md) 核对当前压缩初版。下表的“已有素材”只表示存在相关正文或实测示例，不表示原大纲对应章节已经写完；未列出的要求仍须逐条完成。每个章节最终需有独立的中文正文、完整源码、练习答案和验证记录。

| 原大纲章节 | 当前已有素材 | 待完成的关键内容 |
| --- | --- | --- |
| 1 认识 CUDA，准备环境 | [现有第 1 章](chapters/ch01-getting-started/README.md) | 按原大纲核对官方示例与远程开发说明 |
| 2 从一个 GPU 线程开始 | [现有第 2 章单线程示例](chapters/ch02-thread-indexing/README.md) | 独立成章，完整讲解手动资源生命周期 |
| 3 从线程编号到数组计算 | [现有第 2 章下标示例](chapters/ch02-thread-indexing/README.md) | 整理成独立章节并补边界练习 |
| 4 管理数据、内存和资源 | [现有第 4 章](chapters/ch04-engineering-and-timing/README.md) | 传输合并、资源复用的递进实验 |
| 5 执行模型与 Warp | 无 | 全章 |
| 6 内存层次与数据布局 | [现有第 3 章](chapters/ch03-memory-and-synchronization/README.md) | AoS/SoA、跨步访问、Unified Memory 对照 |
| 7 共享内存与同步 | [现有第 3 章转置示例](chapters/ch03-memory-and-synchronization/README.md) | 块内求和与同步故障定位 |
| 8 归约、原子操作和基础模式 | [现有第 8 章部分归约](chapters/ch08-ai-operators/README.md) | 独立归约、原子、直方图、Scan |
| 9 正确性与数值验证 | [公共验证代码](common/cuda_support.cuh) | 误差顺序、NaN/Inf、随机与极端输入 |
| 10 调试工具与故障定位 | [现有第 4 章工具说明](chapters/ch04-engineering-and-timing/README.md) | 最小故障示例；工具缺失时如实注明未测 |
| 11 先认识库 | 无 | Thrust、CUB、cuBLAS、NPP 的实际案例 |
| 12 建立可信的性能基准 | [现有第 4 章计时](chapters/ch04-engineering-and-timing/README.md) | 多种输入规模、统计波动与 CPU 基线 |
| 13 用 Nsight 找到瓶颈 | 无 | Nsight 实验；当前工具缺失须注明限制 |
| 14 访存优化 | 无 | 连续/跨步、布局与实测对照 |
| 15 执行效率与资源取舍 | 无 | Block、寄存器、占用率实验 |
| 16 矩阵乘优化贯穿案例 | [现有第 8 章直接 GEMM](chapters/ch08-ai-operators/README.md) | 逐步优化与可信前后对比 |
| 17 Stream、Event 与流水线 | [现有第 7 章](chapters/ch07-image-pipeline/README.md) | 明确重叠条件与时间线证据 |
| 18 CUDA Graphs 与内存池 | 无 | 全章 |
| 19 组织 CUDA C++ 工程 | [现有 CMake 工程](CMakeLists.txt) | 模块化、接口、独立构建与分发 |
| 20 测试、部署与性能回归 | [现有 CTest](CMakeLists.txt) | 回归基线、部署矩阵与验收流程 |
| 21 像素、通道与布局 | [图像公共代码](common/image_support.hpp) | 多通道、布局、逐像素操作 |
| 22 均值与高斯滤波 | [现有第 5 章均值滤波](chapters/ch05-image-filtering/README.md) | 高斯滤波及对照 |
| 23 去噪与质量评价 | [现有第 5 章中值去噪](chapters/ch05-image-filtering/README.md) | 更多噪声模型与质量指标 |
| 24 插值与几何重采样 | [现有第 6 章](chapters/ch06-image-resampling/README.md) | 一般几何变换与边界策略 |
| 25 边缘、形态学、对比度 | 无 | 全章 |
| 26 图像处理流水线 | [现有第 7 章](chapters/ch07-image-pipeline/README.md) | 多算子图像质量和真实吞吐量对照 |
| 27 PyTorch 自定义算子 | 无 | 全章 |
| 28 Softmax 与归一化 | [现有第 8 章 Softmax](chapters/ch08-ai-operators/README.md) | 归一化算子与更多维度 |
| 29 Tensor Core、GEMM、CUTLASS | [现有第 8 章直接 GEMM](chapters/ch08-ai-operators/README.md) | Tensor Core 与 CUTLASS；硬件能力需核验 |
| 30 Attention 与推理 | 无 | 全章 |
| 31 线性代数与稀疏求解 | 无 | 全章 |
| 32 FFT 与频域计算 | 无 | 全章 |
| 33 Stencil、PDE、热传导 | [现有第 9 章](chapters/ch09-scientific-computing/README.md) | 更完整的稳定性、边界与误差分析 |
| 34 随机计算与粒子模拟 | 无 | 全章 |
| 35 单机多 GPU | [现有第 10 章](chapters/ch10-multi-gpu/README.md) | 真正双卡路径待具备设备时验证 |
| 36 多机通信与分布式 | 无 | 全章；需要多机环境验证 |
| 37 架构相关高级优化 | 无 | 全章；仅报告本机可测部分 |
| 38 综合项目与交付验收 | 无 | 全章 |

当前可见 GPU 只有一张，Compute Sanitizer 与 Nsight 工具尚未找到。后续章节涉及这些设备或工具时，应将未验证的路径明确列出，不写成“通过”。
