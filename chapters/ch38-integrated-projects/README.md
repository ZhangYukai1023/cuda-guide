# 第 38 章：综合项目与交付验收

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch37-architecture-optimization/README.md)

本章把专题练习整理成四个可交付入口。A 与 D 复用第 26 章的完整图像流水线及其 CPU 对照；B 是多文件 Softmax→LayerNorm→GEMM 小型算子链；C 是有运行参数和输出文件的二维热传导求解器。D 将 A 的输入分给两个独立 GPU 进程，再与单卡完整批次比较。源码分别在 [图像与双卡脚本](examples/scripts/project_a.py)、[双卡入口](examples/scripts/project_d.py)、[算子库](examples/operators/operators.cuh) 和 [热传导入口](examples/heat/heat_delivery.cu)。本章已在 zyk 集成：A 复用已验证的第 26 章并完成文件验收，B/C 独立构建及 GPU/CPU 对照通过；D 因仅一张物理 GPU 以代码 77 跳过。逐项证据见[验证记录](results/validation.md)。D 需真实双卡，不能以脚本语法通过代替项目验收。

## 1. 交付清单与接口

| 项目 | 输入与命令入口 | 输出与正确性对照 | 适用限制 |
| --- | --- | --- | --- |
| A 图像批量预处理 | `project_a.py --pipeline build/ch26/image_pipeline --output DIR`；生成 6 张 17×13 P5 灰度图 | 第 26 章双槽流水线的 3×3 中值、半像素双线性缩小、8 位预览、`float32` 归一化；脚本核对 manifest、9×7 形状、逐像素 `2*v/255-1`、SHA256 | 要先构建并通过第 26 章及公共 PNM 头文件；仅 P5，输入同尺寸 |
| B 小型算子库 | `project_b_operators [--rows R --cols K --out-cols N]`；默认 7×11→7×5 | GPU 三阶段中间值和最终值分别对 CPU 双精度参考，Softmax 每行和约为 1 | `R<=128,K<=256,N<=128`，FP32、行优先；朴素 GEMM，非生产性能实现 |
| C 热传导 | `project_c_heat [--width W --height H --steps S --out DIR]`；默认 67×51、50 步 | GPU 与 CPU 双精度逐格/全场和比较，可写 `heat.f32`、`heat.pgm`、`metrics.json` | 固定零温边界、`r=0.2`，网格 3..512，步数 1..1000；写盘需小端 IEEE float32 |
| D A 的多 GPU 扩展 | `project_d.py --pipeline build/ch26/image_pipeline --output DIR --devices 0,1` | 两个进程各自运行第 26 章 CPU 对照；按源文件名合并，与单卡完整批次逐预览/张量比较 | 两张真实可见 GPU；无 P2P/跨节点，单卡返回 77/跳过 |

每个入口都只在所有检查通过时打印项目 `PASS`。A/D 先拒绝已经存在的输出目录，避免覆盖先前实验；C 若指定的目录非空也拒绝写入。输出目录应在磁盘空间充足、用户有权限的位置。A/D 的日志和 JSON 报告连同输入、预览与张量一起保存，可用哈希追踪一批数据。C 的 `heat.f32` 是行优先小端 FP32 无头数据；用 `metrics.json` 中的宽高重建，`heat.pgm` 是温度 `[0,1]` 夹紧后量化的可视化，不能反向当作求解器的精确浮点结果。

## 2. 项目 A：图像去噪、缩放与批量预处理

第 26 章原程序已经完成 P5 读取、3×3 复制边界中值去噪、半像素双线性缩放和浮点归一化，并对串行/双槽两条 GPU 路径分别核对 CPU。A 的脚本生成确定性的六帧，在同一输入目录调用该程序，然后**再次**从输出文件核验 manifest：源文件没有缺漏或重复、输出形状 9×7、PGM 预览与 `.f32` 个数匹配、所有浮点值有限且与预览像素归一化关系一致。它保存原始 pipeline stdout/stderr、文件 SHA256 和 `report.json`。这一步针对“GPU 算对了但交付文件/清单写错了”的风险；不能代替第 26 章内部的 CPU 算法参考。脚本使用 Python 标准库，无额外包。

构建第 26 章时先同步 `chapters/common/image_io.hpp`，并按该章命令完成独立与根工程测试。完成后运行：

```bash
python3 chapters/ch38-integrated-projects/examples/scripts/project_a.py \
  --pipeline build/ch26/image_pipeline --output build/ch38/project-a-run-001
```

`--output` 必须是新路径；第二次运行选另一个编号。若想处理用户自己的图片，直接使用第 26 章 CLI `image_pipeline OUTPUT_DIR INPUT_PGM_DIR`，仍要保留其 manifest、CPU 验证日志和输入参数记录。A 的固定六帧是交付验收样本，不代表所有图像质量；真实图像应另外记录噪声模型、插值边界、PSNR/SSIM 定义和视觉差异图，参见第 21—26 章。

## 3. 项目 B：Softmax、归一化与 GEMM

[operators.cuh](examples/operators/operators.cuh) 暴露三个 launch 函数，调用者传设备指针、形状和 Stream；[operators.cu](examples/operators/operators.cu) 实现逐行稳定 Softmax、逐行 LayerNorm 和行优先 GEMM。Softmax 先减行最大值，LayerNorm 使用总体方差 `Σ(x-mean)²/K` 加 `epsilon=1e-5`，GEMM 输出为 `(R×K)·(K×N)`。默认输入第一行含接近 1000 的数，检验 `exp` 稳定性；奇数 `R=7,K=11,N=5` 检验尾部维度。CPU 参考从原输入和权重用双精度重算每一阶段，分别检查误差；不只比较最终矩阵，以免阶段间误差抵消。GPU FP32 三阶段与 CPU 双精度存在舍入差异，暂定 Softmax/LayerNorm/GEMM 容差分别 `2e-5/3e-5/4e-5`，本机样本已在 zyk 通过。

接口要求输入/输出设备缓冲足够大、无不受支持的别名、当前设备与 Stream 一致；示例仅做前向传播，不提供自动求导、任意 stride、FP16/BF16 或 Tensor Core。生产算子库应补参数校验、错误传播、更多 dtype/布局与真实模型尺寸，并用第 29 章 cuBLAS/Tensor Core 路径作为性能对照。当前朴素 GEMM 的意义是明确语义和可交付多文件结构，不宣称比库更快。

## 4. 项目 C：二维热传导求解器

项目 C 使用第 33 章的显式五点格式、固定边界和双缓冲，但增加用户可设的宽高/步数和输出目录。`r=0.2` 满足等距二维显式格式的稳定约束 `r<=1/4`；它是无量纲教学参数，不能直接映射某材料的时间秒数。程序 GPU 执行完后用 CPU 双精度从相同初值重算全网格，检查每项有限且约在 `[0,1]`、最大误差 `<=5e-5`、全场和误差 `<=max(5e-3,5e-5*|CPU 全场和|)`。写出的 PGM 只用于观察扩散形状，`metrics.json` 的 `kernel_ms_single_run` 是一次 GPU 时间步循环的 event 计时，不含初始化、CPU 对照、下载与写盘，不能当作稳态性能结论。

项目 C 的 CTest 用默认 67×51 非整 tile 尺寸。交付样本可运行：

```bash
./build/ch38/project_c_heat --width 67 --height 51 --steps 50 --out build/ch38/project-c-run-001
```

若目标是物理时间，应先给出 `α,Δx,Δy,Δt` 与边界条件，按第 33 章的稳定条件计算 `rx,ry` 和步数。本例没有可变导热率、热源或收敛阈值，不应把固定 50 步称为“已经达到稳态”。

## 5. 项目 D：两 GPU 批次扩展

D 先在所选第一张 GPU 上跑六帧 A 单卡基线，再把偶数/奇数帧分到两个输入子目录，启动两个独立的 `image_pipeline` 进程。每个进程的 `CUDA_VISIBLE_DEVICES` 只指定一张物理 GPU，因此该进程内部使用逻辑设备 0；两进程各自做第 26 章 CPU 对照。脚本按**源文件名**合并两个 manifest，不依赖各子目录自己的 `frame-000` 编号，随后逐帧比较双卡与单卡的 PGM 字节和 FP32 张量（容差 `1e-6`）。这是一种无需 P2P 的任务并行；双卡不足时返回 77 且打印 `SKIP`，不能报告加速或多卡正确性。

```bash
python3 chapters/ch38-integrated-projects/examples/scripts/project_d.py \
  --pipeline build/ch26/image_pipeline --output build/ch38/project-d-run-001 --devices 0,1
```

`nvidia-smi -L` 用于检查至少两张物理 GPU，作业调度器若已设置 `CUDA_VISIBLE_DEVICES`，需先弄清可见编号与脚本的物理编号映射。D 记录单/双 GPU 的**单次**墙钟时间，包含进程启动、文件读写和每进程 CPU 参考；六张小图不能形成可信速度排名。正式性能比较要扩大批次与尺寸、固定输入、预热、重复多轮，并报告每设备工作量、单卡/双卡总时间、数据分区/合并成本和设备利用率。双卡只要一种路径失败，项目 D 就未通过。

## 6. 总构建与验收顺序

在仓库根目录构建 B/C：

```bash
cmake -S chapters/ch38-integrated-projects/examples -B build/ch38 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch38 -j
ctest --test-dir build/ch38 --output-on-failure
./build/ch38/project_b_operators --rows 7 --cols 11 --out-cols 5
./build/ch38/project_c_heat --width 67 --height 51 --steps 50
python3 -m py_compile chapters/ch38-integrated-projects/examples/scripts/project_a.py \
  chapters/ch38-integrated-projects/examples/scripts/project_d.py
```

验收按固定顺序进行：①接口与形状/参数/文件格式；②CPU 或同语义基线；③边界、极值、非有限值与误差；④构建、运行和可用时的 Compute Sanitizer 等故障检查；⑤预热重复后的性能与端到端成本；⑥环境、源代码版本、输入哈希、日志及输出目录可复现。四个项目的通过状态分别记录，不能因为 B/C 的 CTest 通过就把 A/D 也记为完成。第 35—36 章需要的多 GPU/多节点资源条件也不能从本章单卡测试推断。

## 7. 常见错误与练习答案

- A/D 输出目录复用导致旧 manifest/张量混入新批次；用全新目录保留输入与日志。
- A 只看内部 `PASS` 而没核对写盘张量 shape/归一化关系；D 只按本地帧序号合并而没按源文件名合并。
- B 把 Softmax 对大数直接 `expf(x)`，或用样本方差 `K-1` 代替接口规定的总体方差 `K`。
- B 最终 GEMM 与 CPU 差不大就忽略 Softmax/LayerNorm 中间层错误；各阶段必须单独验收。
- C 原地覆盖时间层或把零温边界改成周期边界后仍使用原参考。
- D 将单卡 `SKIP`、单次小批次时间或两张卡型号不同造成的结果差异隐藏为“已加速”。

1. A 的 17×13 输入缩小后应是什么尺寸？答：`ceil(17/2)×ceil(13/2)=9×7`；每张 FP32 张量为 `9*7*4=252` 字节。
2. B 为什么分别下载三阶段输出？答：发现阶段级错误并定位数值语义；只看最终矩阵可能出现误差抵消。
3. C 的边界一直为 0，为什么温度总和可下降？答：Dirichlet 零温边界允许热量从域内散出，不能拿封闭系统守恒去验收。
4. D 的两个子目录都输出 `frame-000.pgm`，怎样合并？答：以 manifest 的 `source` 原始文件名为全局键，每个输出路径仍留在本 GPU 的目录内，不直接复制到同名扁平目录。
5. 四个项目中 A/B/C 单卡验证通过而 D 因一张 GPU 跳过，全书第 38 章是否完全验收？答：否。只记录 A/B/C 各自通过和 D 未测；双卡资源可用后再验 D。
