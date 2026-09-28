# 第 28 章：从归约到 Softmax 与归一化

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch27-pytorch-custom-op/README.md)

本章把第 8 章的归约用于“每行一个结果域”的 AI 预处理：Softmax、LayerNorm 与 RMSNorm。示例 [row_normalization.cu](examples/row_normalization.cu) 用一个 Block 负责一行，每个线程跨步处理该行的若干列，逐步归约最大值、和、均值或平方和。CPU 参考用双精度逐行计算，再与 GPU float32 输出在明确容差内比较。[mixed_precision_softmax.cu](examples/mixed_precision_softmax.cu) 另测试 FP16/BF16 输入与输出、FP32 归约的 Softmax。本章在 zyk 使用 nvcc 12.8 独立构建并通过 GPU/CPU 对照；逐项证据见 [验证记录](results/validation.md)。

## 1. 从短行手算稳定 Softmax

对 `[1,2]`，直接计算 `e^1/(e^1+e^2)` 可得约 `[0.26894,0.73106]`。等价但更稳的做法先减行最大值 2：`[e^-1/(e^-1+1),1/(e^-1+1)]`。一整行 `[1000,1001]` 如果先对原值求指数，普通浮点会溢出；先减 1001 后仍是同样的两个概率。GPU 每行先归约最大值，再归约 `exp(x-max)` 的和，最后每个元素除以该和。测试还检查每行概率和近似 1。输入约定为**有限 float32**；整行都是 `-Inf`、含 NaN 或 mask 语义需要额外定义，不能把本例的结果当作已支持。

## 2. LayerNorm 与 RMSNorm 的统计口径

LayerNorm 对行内 `width` 个值先求算术均值，再求总体方差 `sum((x-mean)^2)/width`，输出 `(x-mean)/sqrt(variance+1e-5) * gamma[x] + beta[x]`。RMSNorm 不减均值，而求 `mean(x²)`，输出 `x/sqrt(mean(x²)+1e-5) * gamma[x]`。`gamma`、`beta` 是按列复用的参数，本例用确定性的非全一/非全零值检验下标。单列 LayerNorm 的方差为 0，输出 `beta[0]`；全零行 RMSNorm 输出全零，不会除零，因为 epsilon 正数。

GPU 在每个线程局部累加，再用 256 项共享内存树归约。统计阶段使用双精度累加以减少长行和大偏移输入的抵消误差，输出仍是 float32；这是一条侧重正确性的教学实现，不能代表最佳吞吐。LayerNorm 分两次归约：先均值，后平方偏差；不要用 `E[x²]-E[x]²` 对大偏移、很小方差的数据做草率相减。RMSNorm 只需一次平方和归约。三个 kernel 都把输入、输出与参数视为**连续行优先**数组，行跨度等于 `width`；非连续 PyTorch 张量需在接口层处理，不能直接传 `data_ptr` 进本例。

## 3. 构建与运行

在 zyk 仓库根目录执行：

```bash
cmake -S chapters/ch28-softmax-normalization/examples -B build/ch28 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch28 -j
ctest --test-dir build/ch28 --output-on-failure
./build/ch28/row_normalization
./build/ch28/mixed_precision_softmax
```

`row_normalization` 依次测试 `1×5`、`7×37`、`4×2049`、含约 ±1000 输入的 `3×37`、以及极端的 `2×1`。每个形状运行三个算子，要求 `bad=0 ... PASS`；Softmax 还要求各行概率和在 `2e-5` 内接近 1，末尾打印 `chapter 28 row normalization: PASS`。逐项容差为 `2e-5 + 2e-5*|CPU参考值|`，报告最大绝对误差。该程序的 `kernel_ms` 是单次 Event 时间，`task_ms` 包含本次 H2D、初始化、kernel、同步、D2H，不含 CPU 参考、参数缓冲分配或文件读写。它们不是稳定性能结果；按短行、长行及批量变化做结论前必须预热、重复、查看资源使用与设备负载。`mixed_precision_softmax` 只做正确性测试，当前不报告性能。

框架对照应把轴、epsilon、是否有 gamma/beta、计算精度和 dtype 设成相同值，再比较输出及梯度。是否值得用 Warp 级归约或多 Block 长行方案要以实测尺寸和硬件为准。

## 4. 混合精度的输入与参考口径

新增的 `mixed_precision_softmax` 先把确定性的 float32 输入按 round-to-nearest-even 转为 FP16 或 BF16，GPU 从**已量化的**输入读取，先转 FP32，再用行最大值与 FP32 归约计算概率，最后把概率量化回原 dtype。CPU 参考从同一批已量化输入用双精度算稳定 Softmax，再量化到相同输出 dtype。这避免拿原始 float32 logits 的参考值去错误评价低精度输入。实现使用 CUDA 的 [FP16 转换 API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html) 与 [BF16 转换 API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html)。

它测试 `1×5` 和含极大 logits、257 列尾块的 `3×257`。BF16 只在计算能力 8.0 及以上运行，否则明确打印 `SKIP`；FP16 仍运行。每个输出须有限，分别与相同 dtype 的 CPU 参考比较；FP16 最大绝对差不超过 `2e-3`、行和误差不超过 `5e-3`，BF16 分别不超过 `2e-2` 与 `3e-2`。这些教学容差在本次 zyk 样本上已通过，不是性能或通用精度保证。约 1000 的输入在 BF16 量化时可能合并为相同值，因此不能期待 BF16 与 FP16 或原始 float32 输出完全一致。当前混合精度例子仅实现 Softmax，LayerNorm/RMSNorm 的低精度路径及框架同语义对照仍是后续扩展；**不能把这些未做的部分标为已验证**。

## 5. 常见错误

- Softmax 先对未经减最大值的 logits 求指数，极大正数导致溢出。
- 每个线程只处理一列，却用 256 个线程处理 2049 列，遗漏尾部元素。
- LayerNorm 用 `E[x²]-E[x]²` 且全程低精度，对大偏移行得到负方差或明显误差。
- 把 LayerNorm 的总体方差除以 `width-1`，和框架约定不同。
- RMSNorm 误减均值，变成 LayerNorm 的另一种写法。
- 读 `gamma[row*width+x]`，而参数实际上只按列存一份。
- 用单次小批量计时推断优化收益，或不说明是否包含传输。
- 用未量化的 float32 输入计算 CPU 参考，却要求低精度输入的 GPU 输出逐位相同；先核对输入与输出的舍入位置。

## 6. 练习与参考答案

1. `[1,2]` 的 Softmax 约是多少？答：减去最大值 2 后为 `[e^-1/(1+e^-1),1/(1+e^-1)]≈[0.26894,0.73106]`。
2. `[1000,1001]` 的 Softmax 是否与 `[1,2]` 相同？答：相同；对整行加同一常数不改变 Softmax，先减最大值可避免指数溢出。
3. `[1,3]` 的 LayerNorm 在 `gamma=1,beta=0,epsilon` 很小时约是多少？答：均值 2、总体方差 1，约 `[-1,1]`。
4. 宽度为 1 的 LayerNorm 输出是什么？答：均值等于该值，中心化项为 0，所以输出 `beta[0]`。
5. 为什么测试 2049 列？答：宽度超过 256 且不整除线程数，检查每线程跨步遍历、归约和尾部索引。
6. 为什么 BF16 大数行可能与 FP16 不同？答：两种格式的尾数位数不同；原始 logits 在输入量化时可能合并，比较时要从实际量化值重新计算参考。

[下一章](../ch29-tensor-core-cutlass/README.md) 在第 16 章 GEMM 基础上研究 Tensor Core、混合精度与 CUTLASS 的选择。
