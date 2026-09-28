# CUDA C++ 开发指导书

面向有基本 C++ 知识、没有 CUDA 经验的读者。正式正文按 [38 章原始大纲](outline.md) 编写，放在 `chapters/`；此前可运行的 10 章压缩初版保留在 `pilot/`，供复用示例与结果。正文使用中文，文件与目录使用英文。

## 按原大纲编写的章节

| 章节 | 内容 | 验证状态 |
| --- | --- | --- |
| [第 1 章：认识 CUDA，准备实验环境](chapters/ch01-getting-started/README.md) | 环境、设备查询、数组加法、远程使用 | GPU 实测通过 |
| [第 2 章：从一个 GPU 线程开始](chapters/ch02-first-thread/README.md) | 写入 42、标量加法、8 个线程写编号 | CPU 对照与独立构建通过 |
| [第 3 章：从线程编号到数组计算](chapters/ch03-array-indexing/README.md) | 一维、Grid-stride、二维索引与尾块 | CPU 对照通过 |
| [第 4 章：管理数据、内存和资源](chapters/ch04-memory-resources/README.md) | 平方、多轮缓冲区复用、打包传输 | CPU 对照通过 |
| [第 5 章：执行模型与 Warp](chapters/ch05-warp-execution/README.md) | Warp/lane、统一与奇偶分支、输入分布 | CPU 对照通过 |
| [第 6 章：内存层次与数据布局](chapters/ch06-memory-layout/README.md) | 连续/跨步、行跨度、AoS/SoA、Managed | CPU 对照通过 |
| [第 7 章：共享内存与同步](chapters/ch07-shared-memory/README.md) | 块内交换、Warp 同步、分块转置与块内求和 | CPU 对照通过 |
| [第 8 章：归约、原子操作和基础并行模式](chapters/ch08-reduction-atomics/README.md) | 两阶段求和、最大值位置、16 桶直方图 | CPU 对照通过 |
| [第 9 章：正确性与数值验证](chapters/ch09-correctness-validation/README.md) | 边界、浮点容差、非有限值与大逻辑下标 | CPU 对照通过 |
| [第 10 章：调试工具与故障定位](chapters/ch10-debugging/README.md) | 安全基线、四类故障复现入口 | 安全模式通过；工具检查未测 |
| [第 11 章：先认识库，再决定实现方式](chapters/ch11-cuda-libraries/README.md) | Thrust 排序、CUB 求和、可选 cuBLAS GEMM | Thrust/CUB 通过；cuBLAS 缺依赖未测 |
| [第 12 章：建立可信的性能基准](chapters/ch12-benchmarking/README.md) | Event、传输、墙钟与 CPU 同口径测量 | 正确性通过；GPU 繁忙，性能结论待测 |
| [第 13 章：用 Nsight 找到瓶颈](chapters/ch13-nsight-profiling/README.md) | 小调用、批量调用、跨步读与分析方法 | 三种模式 CPU 对照通过；Nsight/NVTX 未测 |
| [第 14 章：访存优化，从数据排列开始](chapters/ch14-memory-optimization/README.md) | 朴素与共享内存转置、填充、Event 对照 | CPU 对照通过；性能待空闲复测 |
| [第 15 章：执行效率与资源取舍](chapters/ch15-execution-resources/README.md) | 求和、直方图、Block 与资源上限 | 24 组 CPU 对照通过；性能待空闲复测 |
| [第 16 章：矩阵乘优化贯穿案例](chapters/ch16-matmul-case/README.md) | 朴素、共享分块、双输出与可选 cuBLAS | 三个手写版本通过；cuBLAS 缺依赖未测 |
| [第 17 章：Stream、Event 与处理流水线](chapters/ch17-stream-pipeline/README.md) | 1/2/4 槽批处理、Event 依赖与合并 | CPU 对照通过；实际重叠未测 |
| [第 18 章：重复任务、CUDA Graphs 与内存池](chapters/ch18-graphs-memory-pool/README.md) | 两段 kernel 重放、Graph、流顺序分配 | CPU 对照通过；性能待空闲复测 |
| [第 19 章：组织 CUDA C++ 工程](chapters/ch19-cuda-project/README.md) | 静态库、跨文件设备链接、异步接口 | 独立及根工程构建和 CPU 对照通过 |
| [第 20 章：测试、部署与性能回归](chapters/ch20-testing-deployment/README.md) | JSON 交付报告、门禁、基准解析 | 构建/CTest/采集通过；memcheck 缺工具未测 |
| [第 21 章：像素、通道、布局与逐像素操作](chapters/ch21-image-layout/README.md) | 行跨度、ROI、RGB/BGR/RGBA、PNM | 默认及 PGM 输入路径 CPU 对照通过 |
| [第 22 章：从均值滤波到高斯滤波](chapters/ch22-mean-gaussian/README.md) | 边界、直接/可分离高斯、共享 Halo | 18 组 CPU 对照与手算像素通过 |
| [第 23 章：去噪算法与质量评价](chapters/ch23-denoising/README.md) | 椒盐/高斯噪声，中值/高斯/双边与 PSNR/SSIM8 | 六条路径 CPU 对照通过；质量值限当前样本 |
| [第 24 章：插值、缩放与几何重采样](chapters/ch24-image-resampling/README.md) | 最近邻、双线性、双三次、面积、90° 旋转 | 十条默认及 PGM 输入路径 CPU 对照通过 |
| [第 25 章：边缘、形态学与对比度增强](chapters/ch25-edges-morphology-contrast/README.md) | Sobel、膨胀/腐蚀、开闭、直方图均衡化 | 18 条默认及 PGM 输入路径 CPU 对照通过 |
| [第 26 章：完整图像处理流水线](chapters/ch26-image-pipeline/README.md) | PGM→中值→缩小→归一化，串行与双槽 | 合成与目录输入 CPU 对照通过；性能待复测 |
| [第 27 章：张量与 PyTorch 自定义算子](chapters/ch27-pytorch-custom-op/README.md) | CUDA 扩展、当前 Stream、非连续输入与梯度 | 现有 PyTorch 环境独立构建与 GPU 检查通过 |
| [第 28 章：从归约到 Softmax 与归一化](chapters/ch28-softmax-normalization/README.md) | 稳定 Softmax、LayerNorm、RMSNorm、FP16/BF16 | 15 组 FP32 与 4 组低精度 CPU 对照通过 |
| [第 29 章：Tensor Core、GEMM 与 CUTLASS](chapters/ch29-tensor-core-cutlass/README.md) | FP16 GEMM、WMMA、可选 cuBLAS/CUTLASS | 朴素/WMMA CPU 对照通过；两种可选库缺依赖未测 |
| [第 30 章：Attention 与推理专题](chapters/ch30-attention-inference/README.md) | 显式与在线前向、因果 Mask | 六组输入两条 GPU 路径 CPU 对照通过；框架/反向未测 |
| [第 31 章：线性代数与稀疏求解](chapters/ch31-linear-sparse-solvers/README.md) | 稠密 GEMV、CSR SpMV、混合式 CG、可选库求解 | 普通 CUDA 路径通过；cuBLAS/cuSPARSE/cuSOLVER 缺依赖未测 |
| [第 32 章：FFT 与频域计算](chapters/ch32-fft-frequency/README.md) | 直接 DFT、低通、线性卷积、可选 cuFFT | 普通 CUDA 教学路径通过；cuFFT 缺依赖未测 |
| [第 33 章：Stencil、PDE 与热传导](chapters/ch33-stencil-heat/README.md) | 1D/2D 显式扩散、共享内存 Halo | 三组网格 CPU 对照通过；性能未建基线 |

其余大纲章节正在编写。逐章现有素材与缺口见 [覆盖核对](coverage.md)。

## 压缩初版（pilot）导航

| 章节 | 内容 | 验证状态 |
| --- | --- | --- |
| [第 2 章：线程、下标与二维数据](pilot/ch02-thread-indexing/README.md) | 单线程写值与标量相加、一维步长循环、二维下标与边界 | CPU 对照通过 |
| [第 3 章：内存访问与块内同步](pilot/ch03-memory-and-synchronization/README.md) | 直接转置、共享内存、屏障 | CPU 对照通过 |
| [第 4 章：工程组织、错误处理与性能测量](pilot/ch04-engineering-and-timing/README.md) | RAII、错误检查、预热与两种计时 | CPU 对照通过 |
| [第 5 章：图像滤波与去噪](pilot/ch05-image-filtering/README.md) | 均值、中值、边界、效果图与 MSE | CPU 对照通过 |
| [第 6 章：插值与几何变换](pilot/ch06-image-resampling/README.md) | 最近邻、双线性、缩放、平移 | CPU 对照通过 |
| [第 7 章：图像处理流水线](pilot/ch07-image-pipeline/README.md) | 设备中间结果、算子融合、双槽多帧 | CPU 对照通过 |
| [第 8 章：AI 算子：矩阵乘法与 Softmax](pilot/ch08-ai-operators/README.md) | 直接矩阵乘法、稳定 Softmax、归约 | CPU 对照通过 |
| [第 9 章：科学计算：二维热扩散](pilot/ch09-scientific-computing/README.md) | 显式热扩散、双缓冲、稳定性与误差 | CPU 对照通过 |
| [第 10 章：多 GPU：任务划分与结果汇总](pilot/ch10-multi-gpu/README.md) | 分片、设备归属、主机汇总 | 单卡通过；双卡跳过 |

各章包含小规模手算、最小与递进示例、常见错误、练习与参考答案。完整代码位于章节 examples/，正文链接到它们；公共依赖也随仓库提供。

## 项目结构

```text
cuda-guide/
  README.md
  CMakeLists.txt
  .gitignore
  common/
    cuda_support.cuh
    image_support.hpp
  chapters/
    ch01-getting-started/
    ch02-first-thread/
    ch03-array-indexing/
    ch04-memory-resources/
    ch05-warp-execution/
    ch06-memory-layout/
    ch07-shared-memory/
    ch08-reduction-atomics/
    ch09-correctness-validation/
    ch10-debugging/
    ch11-cuda-libraries/
    ch12-benchmarking/
    ch13-nsight-profiling/
    ch14-memory-optimization/
    ch15-execution-resources/
    ch16-matmul-case/
    ch17-stream-pipeline/
    ch18-graphs-memory-pool/
    ch19-cuda-project/
    ch20-testing-deployment/
    ch21-image-layout/
    ch22-mean-gaussian/
    ch23-denoising/
    ch24-image-resampling/
    ch25-edges-morphology-contrast/
    ch26-image-pipeline/
    ch27-pytorch-custom-op/
    ch28-softmax-normalization/
    ch29-tensor-core-cutlass/
    ch30-attention-inference/
    ch31-linear-sparse-solvers/
    ch32-fft-frequency/
    ch33-stencil-heat/
    common/image_io.hpp
  pilot/
    ch02-thread-indexing/
    ...
    ch10-multi-gpu/
  scripts/
    render_images.py
  results/
    validation.md
  build/                       # Git 忽略的本机构建产物
```

## 构建当前已有源码

以下命令针对当前实测服务器；全部从项目根目录运行：

```bash
cd /data2/cuda-guide
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline -j2
ctest --test-dir build/outline --output-on-failure
```

本机 GPU 为 RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93，GCC 13.3.0，CMake 3.28.3。nvcc 未加入默认 PATH，因此使用绝对路径。其他机器需重新选择工具路径与目标架构。没有自动安装或升级驱动、CUDA 或依赖。

当前统一验证：45 项通过，1 项因没有第二块 GPU 跳过，0 项失败。不能把 CTest 的“100% tests passed”解释成双卡也通过。详情见 [全书验证记录](results/validation.md) 和各章 results/validation.md。

## 图像结果

pilot 的第 5、6、7 章保存了实际图像效果与数值验证，第 9 章也保存了温度场。PGM 为量化像素，SVG 为逐像素可视化，PNG 为便于预览的四倍最近邻显示图。CPU/GPU 误差在量化前计算。

下面按左上、右上、左下、右下显示干净参考、加噪输入、均值输出、中值输出：

![Denoising comparison](pilot/ch05-image-filtering/results/images/comparison.png)

图像来自确定性合成输入，不依赖外部图片。当前样例的噪声 MSE 为 1513.018421，均值后 282.997097，中值后 18.025412；不作为普遍算法排名。

程序的默认输出或正文命令输出位于 build/，不会覆盖归档图。需要从归档 PGM 重新生成 PNG 时，运行：

```bash
python3 scripts/render_images.py
```

该脚本仅使用 Python 标准库；编译和运行 CUDA 示例本身不需要 Python。

## 范围、来源与尚未验证的部分

原始 38 章大纲现已同步到 [outline.md](outline.md)，校验值与本机源文件一致。当前正式正文已写至第 33 章；部分工具、依赖与性能路径仍未验证；先前 10 章压缩初版保留在 pilot/，章节安排与原大纲不一致。逐章差距见 [覆盖核对](coverage.md)。当前工作目录未发现适用的 AGENTS.md。

pilot 提供可运行的入门应用示例，38 章正文仍在编写。共享内存、归约与 WMMA 已有实测示例；cuBLAS/cuDNN 集成、一般仿射旋转、P2P、NCCL/MPI 尚未实测的路径在各章分别标注。

本机未找到 Compute Sanitizer、cuda-gdb、nsys、ncu，因此没有内存/同步工具通过结论或 profiler 并发证明。只有一块可见 GPU，双卡路径、设备间通信和多卡性能未验证。第 12 章记录了多口径计时和 CPU 对照，但 GPU 被其他任务占用，可信性能基线仍待复测；第 13 章因缺工具还没有 profiler 报告。

通过 SSH 别名 zyk 连接到实际主机 ubuntu2404。环境记录保留真实主机名。

已初始化本地 Git 仓库；未建立远程仓库。已有的 .vscode/、README.html 保留，不属于本次新增教材内容。

## 技术参考

示例为本项目自行编写。特定 API 与工具说明在相关章节中链接官方资料：

- [CUDA C++ Programming Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
