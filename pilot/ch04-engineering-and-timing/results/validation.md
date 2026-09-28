# 实际验证记录

验证日期：2026-09-28；主机 ubuntu2404；RTX 5060 Ti；驱动 595.84；nvcc 12.8.93；GCC 13.3.0；CMake 3.28.3。架构参数为 120，Release 构建，未使用 fast-math。

命令：`./build/all/ch04/benchmark`。工作目录为项目根目录；退出码：0。

```text
affine: n=1048576 max_abs_error=0 mismatches=0 PASS
end_to_end: n=1048576 max_abs_error=0 mismatches=0 PASS
n=1048576 warmup=5 repeats=100
kernel_sequence_mean_ms=0.007299
end_to_end_mean_ms=0.537455 (H2D+kernel+D2H, allocation excluded)
```

CPU 参考逐项核对，浮点示例同时检查非有限值。完整测试范围和容差见本章正文及源码。本机没有找到 Compute Sanitizer，因此没有 memcheck、racecheck 或 synccheck 通过结论。

以上计时为本次一次运行的测量值，包含 5 次预热及 100 次重复；不是硬件的固定性能。CUDA event 记录连续 kernel 序列的平均时间，steady_clock 记录已分配缓冲区下的 H2D+kernel+D2H 平均时间。
