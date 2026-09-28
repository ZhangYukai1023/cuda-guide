# 原始大纲覆盖核对

依据 [38 章原始大纲](outline.md) 核对已完成的正式章节和压缩初版。下表的“已有素材”只表示存在相关正文或实测示例，不表示原大纲对应章节已经写完；未列出的要求仍须逐条完成。每个章节最终需有独立的中文正文、完整源码、练习答案和验证记录。

| 原大纲章节 | 当前已有素材 | 待完成的关键内容 |
| --- | --- | --- |
| 1 认识 CUDA，准备环境 | [正式第 1 章](chapters/ch01-getting-started/README.md) | 官方示例来源仍可补充；远程流程已核对 |
| 2 从一个 GPU 线程开始 | [正式第 2 章](chapters/ch02-first-thread/README.md) | 基础章节已完成；后续可加边界与故障练习 |
| 3 从线程编号到数组计算 | [正式第 3 章](chapters/ch03-array-indexing/README.md) | 一维与二维基础已完成；可扩展更大索引范围 |
| 4 管理数据、内存和资源 | [正式第 4 章](chapters/ch04-memory-resources/README.md) | 平方、复用与打包已完成；更多传输规模对照可扩展 |
| 5 执行模型与 Warp | [正式第 5 章](chapters/ch05-warp-execution/README.md) | 正确性与谓词分布已验证；性能分析后置 |
| 6 内存层次与数据布局 | [正式第 6 章](chapters/ch06-memory-layout/README.md) | 连续/跨步、行跨度、AoS/SoA 与 Managed 已验证；性能分析后置 |
| 7 共享内存与同步 | [正式第 7 章](chapters/ch07-shared-memory/README.md) | 块内交换、Warp 同步、分块转置、块内求和均已通过 CPU 对照；故障工具检查未测 |
| 8 归约、原子操作和基础模式 | [正式第 8 章](chapters/ch08-reduction-atomics/README.md) | 两阶段整数归约、并列最大值下标、共享桶原子直方图已通过 CPU 对照；Scan/Gather/Scatter 为语义练习 |
| 9 正确性与数值验证 | [正式第 9 章](chapters/ch09-correctness-validation/README.md) | 空输入、确定性随机数据、组合容差、NaN/Inf、舍入顺序、半精度往返和大逻辑下标已验证 |
| 10 调试工具与故障定位 | [正式第 10 章](chapters/ch10-debugging/README.md) | Release/Debug 安全模式及根回归已通过；Compute Sanitizer 与 CUDA-GDB 未安装，四类故障定位未测 |
| 11 先认识库 | [正式第 11 章](chapters/ch11-cuda-libraries/README.md) | Thrust 排序和 CUB 求和 GPU/CPU 对照已通过；cuBLAS 开发库缺失，GEMM 未测；NPP 仅作定位介绍 |
| 12 建立可信的性能基准 | [正式第 12 章](chapters/ch12-benchmarking/README.md) | 三种规模、预热、重复与多口径统计已运行且 CPU 对照通过；设备被其他任务占满，性能结论待空闲复测 |
| 13 用 Nsight 找到瓶颈 | [正式第 13 章](chapters/ch13-nsight-profiling/README.md) | 三种调用模式的 GPU/CPU 对照通过；NVTX3、Nsight Systems/Compute 缺失，时间线与指标未测 |
| 14 访存优化 | [正式第 14 章](chapters/ch14-memory-optimization/README.md) | 朴素与两种共享转置均通过 CPU 对照；GPU 有并发任务，性能结论待空闲复测，Nsight 指标未测 |
| 15 执行效率与资源取舍 | [正式第 15 章](chapters/ch15-execution-resources/README.md) | 求和与直方图 24 组均通过 CPU 对照；预测 Occupancy 非实测活跃度，性能待空闲复测 |
| 16 矩阵乘优化贯穿案例 | [正式第 16 章](chapters/ch16-matmul-case/README.md)；[pilot GEMM](pilot/ch08-ai-operators/README.md) | 三种手写版本通过 CPU 对照；cuBLAS 缺开发库未测，性能待空闲复测 |
| 17 Stream、Event 与流水线 | [正式第 17 章](chapters/ch17-stream-pipeline/README.md)；[pilot 流水线](pilot/ch07-image-pipeline/README.md) | 1/2/4 槽正确性与事件依赖通过；Nsight 缺失，实际重叠时间线未测 |
| 18 CUDA Graphs 与内存池 | [正式第 18 章](chapters/ch18-graphs-memory-pool/README.md) | Graph 与流顺序分配正确性通过；并发负载下性能待空闲复测 |
| 19 组织 CUDA C++ 工程 | [正式第 19 章](chapters/ch19-cuda-project/README.md)；[根 CMake](CMakeLists.txt) | 多文件静态库、跨文件设备链接、接口错误路径已通过；Driver/NVRTC 与动态库未实现 |
| 20 测试、部署与性能回归 | [正式第 20 章](chapters/ch20-testing-deployment/README.md)；[根 CTest](CMakeLists.txt) | 报告脚本在 zyk 实测；memcheck 缺工具而正确失败，性能基线与跨机器部署未测 |
| 21 像素、通道与布局 | [图像公共代码](common/image_support.hpp) | 多通道、布局、逐像素操作 |
| 22 均值与高斯滤波 | [现有第 5 章均值滤波](pilot/ch05-image-filtering/README.md) | 高斯滤波及对照 |
| 23 去噪与质量评价 | [现有第 5 章中值去噪](pilot/ch05-image-filtering/README.md) | 更多噪声模型与质量指标 |
| 24 插值与几何重采样 | [现有第 6 章](pilot/ch06-image-resampling/README.md) | 一般几何变换与边界策略 |
| 25 边缘、形态学、对比度 | 无 | 全章 |
| 26 图像处理流水线 | [现有第 7 章](pilot/ch07-image-pipeline/README.md) | 多算子图像质量和真实吞吐量对照 |
| 27 PyTorch 自定义算子 | 无 | 全章 |
| 28 Softmax 与归一化 | [现有第 8 章 Softmax](pilot/ch08-ai-operators/README.md) | 归一化算子与更多维度 |
| 29 Tensor Core、GEMM、CUTLASS | [现有第 8 章直接 GEMM](pilot/ch08-ai-operators/README.md) | Tensor Core 与 CUTLASS；硬件能力需核验 |
| 30 Attention 与推理 | 无 | 全章 |
| 31 线性代数与稀疏求解 | 无 | 全章 |
| 32 FFT 与频域计算 | 无 | 全章 |
| 33 Stencil、PDE、热传导 | [现有第 9 章](pilot/ch09-scientific-computing/README.md) | 更完整的稳定性、边界与误差分析 |
| 34 随机计算与粒子模拟 | 无 | 全章 |
| 35 单机多 GPU | [现有第 10 章](pilot/ch10-multi-gpu/README.md) | 真正双卡路径待具备设备时验证 |
| 36 多机通信与分布式 | 无 | 全章；需要多机环境验证 |
| 37 架构相关高级优化 | 无 | 全章；仅报告本机可测部分 |
| 38 综合项目与交付验收 | 无 | 全章 |

当前可见 GPU 只有一张，Compute Sanitizer 与 Nsight 工具尚未找到。后续章节涉及这些设备或工具时，应将未验证的路径明确列出，不写成“通过”。
