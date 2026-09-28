# 第 28 章验证记录

2026-09-28 在 zyk，RTX 5060 Ti（计算能力 12.0）、nvcc 12.8.93，独立配置、构建及 CTest 2/2 通过。row_normalization 的五种尺寸 × Softmax/LayerNorm/RMSNorm 共 15 组均为 bad=0、max_abs_error=0，包括 2049 列尾部、约 ±1000 大数、宽度 1。完整输出：[FP32 原始记录](run-row-2026-09-28.txt)。

mixed_precision_softmax 在同机实际运行了 FP16、BF16 各两组（1×5 与 3×257），四组均与同 dtype 的量化 CPU 参考 max_error=0；最大行和误差为 FP16 0.000286102、BF16 0.000860214。BF16 此次没有 SKIP。完整输出：[混合精度原始记录](run-mixed-2026-09-28.txt)。输入先量化、内部 FP32 归约、输出再量化。

根工程重新配置、全量构建及 CTest 41 项中 40 项通过、0 项失败；第 10 章双 GPU 用例因仅一张 GPU 跳过。耗时 3.96 秒。无稳定性能结论：章内 Event 时间是单次采样，未预热和重复统计，且设备上有并发任务。未实现低精度 LayerNorm/RMSNorm；未做与 PyTorch 同语义的框架输出/梯度对照，不能将其视为已验证。
