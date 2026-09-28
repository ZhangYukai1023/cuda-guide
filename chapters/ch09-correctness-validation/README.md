# 第 9 章：正确性与数值验证

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch08-reduction-atomics/README.md)

第 8 章的整数归约可以精确比较；本章给浮点结果建立明确的验收规则。示例源码见 [correctness_validation.cu](examples/correctness_validation.cu)。本章已在 zyk 用 nvcc 构建并运行，所有实际 CPU/GPU 对照及独立、根工程 CTest 均通过，详情见[验证记录](results/validation.md)。

## 1. 先写接口约定，再写测试

示例包含整数、浮点逐元素相加，四项浮点归约、半精度往返和大逻辑下标。对数组相加规定：输入 A、B 必须各有 `n` 项；`n=0` 返回空结果，不分配设备内存，也不启动 kernel；正长度才访问下标 `[0,n)`。`n` 用 `std::size_t` 表示，因此负尺寸不能直接进入此接口；若上游接收有符号尺寸，应在转换前拒绝负值。当前示例不提供任意外部尺寸解析，实际工程还须检查 `n*sizeof(T)`、网格维度是否溢出以及分配失败。

这些约定决定测试矩阵：0、1、7、1003、4097 项分别覆盖空输入、单项、不满一个 Warp、非整齐尾块和多个 Block。整数输入含正负数，用 CPU 的 `a[i]+b[i]` 逐项做**精确**比较；若任何项不同，程序退出码为 1。浮点输入由固定状态 `0x00c0ffee` 的线性同余序列生成，再映射为以 1/128 为单位的数，使相同程序可重现输入。固定种子只是复现手段，不代表测试覆盖所有可能数据。

“大下标”不要求分配 4 GiB 以上内存。示例只分配 8 个整数，但用 `2^32+123` 作为逻辑起点，在 GPU 上计算 `(base+i)%97` 并与 CPU 比较。它检验这一处索引表达式的 64 位计算；不能据此断言整套程序支持任意大数组。

## 2. 为浮点比较定义清楚的规则

绝对误差 `|actual-reference|` 适合零附近；相对误差把偏差放在参考值的尺度上。示例采用组合条件：

```text
|actual - reference| <= atol + rtol * |reference|
```

本章普通浮点相加选 `atol=1e-6`、`rtol=1e-6`，CPU 先把两个 float 输入提升为 double 再求参考值。这个容差只对应本章数据范围和单次加法，不能直接套到长归约、Softmax 或图像质量指标上。若测试失败，应打印失败下标、输入和误差并检查运算路径；不应先任意放大容差。生产验证还应统计最大绝对误差、最大相对误差和失败样本，方便定位。

`NaN` 不能用普通大小比较：任何实际 `NaN` 对有限参考值都应失败；只有测试明确预期 `NaN`，才接受 `NaN`。正、负无穷须分别匹配同号预期值。示例中 `+Inf + -Inf` 和 `NaN + 2` 明确预期 `NaN`。本章的正负零按数值相等处理；若接口关心符号位，应额外用 `std::signbit` 检查。比较器还自检，避免“差值是 NaN，却被遗漏判错”的情况。

## 3. 相同数学式可以有不同的舍入结果

对四个 float：`[1e8, 1, -1e8, 1]`，数学精确和是 2。示例按顺序在 CPU 用 float 加，预期得到 1；GPU 先加相邻两项，再合并，预期得到 0。因为 `1e8f+1` 在 float 中会舍入回 `1e8f`，两种顺序丢失的信息不同。这只是一个受控例子，不表示 GPU 总是更不准或 CPU 总是更准。

并行归约的树形顺序与串行循环可能不同，因此本章把“相同输入、明确算法、允许误差”写进验收规则；对于要求可重现的业务，应进一步固定归约算法、编译选项和平台范围，并记录结果。NVIDIA 的 [Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html) 解释了加法顺序与 fused multiply-add 对数值的影响；[CUDA 最佳实践指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/) 建议以代表性输入的参考结果持续核验修改。

## 4. 混合精度先看输入量化

`round_via_half` 把 float 转成 IEEE half 再转回 float。`1.0001f` 在 half 的 1 附近无法保留这段小差异，往返结果预期是 `1.0f`。这个例子只展示输入量化，没有执行 half 累加，更没有展示 Tensor Core 吞吐。若后续在 float 中累加 half 输入，输入转换已经丢失的信息不会因累加精度提高而恢复。工程中应分别测量输入转换误差、累加误差和最终输出误差。

## 5. 构建与运行

在仓库根目录运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target correctness_validation -j2
./build/outline/book_ch09/correctness_validation
ctest --test-dir build/outline -R '^ch09_correctness_validation$' --output-on-failure
```

独立构建可用 `cmake -S chapters/ch09-correctness-validation/examples -B build/ch09-correctness-standalone`，沿用同样的 CUDA 编译器和架构参数。独立构建与根工程命令已实际执行，输出及限制见[验证记录](results/validation.md)。

## 6. 常见故障

- 对空数组仍启动 `<<<0,128>>>`，导致无效网格；本例在 `n=0` 时直接返回。
- 在 `n` 转为无符号值后才检查负尺寸，负数已变成极大正数；须在转换之前检查。
- 把浮点结果一律用 `==` 比较，导致合法舍入差异被误判；或只检查 `abs(diff)>tol`，让 `NaN` 悄悄通过。
- 把某组输入上的容差当成通用常数，掩盖新算法的系统偏差；应按算法的误差预算和业务要求设定。
- 把半精度输入转回 float 后误以为原始精度已恢复。
- 用一组固定随机输入取代边界与极值测试；随机测试与针对性用例需要互补。

## 7. 练习与参考答案

1. 若 `reference=0`、`actual=5e-7`，本章容差是否接受？答：接受，右边是 `1e-6`；仅相对误差会在零附近失效。
2. 若 `reference=1000`、`actual=1000.0005`，是否接受？答：接受，阈值是 `0.001001`；这说明相对项随参考尺度扩大。
3. 为什么 `NaN` 对 `NaN` 要靠显式策略判断？答：IEEE 浮点比较中 `NaN==NaN` 为假；“两边都是 NaN 视为匹配”是本测试明确选定的约定。
4. 为本章再加两个边界输入。答例：整数全零与常量数组；浮点使用最大有限值与接近零的数，并单独规定溢出时是否预期 Inf。
5. 若程序公开接受 `int length`，怎样处理 `-1`？答：先检查 `length<0` 并报参数错误，再转换为 `std::size_t`；不能直接强转后分配。

下一章使用调试工具为故障保存复现输入、定位证据与修复后的验证结果。
