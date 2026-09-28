# 实际验证记录

验证日期：2026-09-28；主机 ubuntu2404；RTX 5060 Ti；驱动 595.84；nvcc 12.8.93；GCC 13.3.0；CMake 3.28.3。架构参数为 120，Release 构建，未使用 fast-math。

命令：`./build/all/ch05/filtering pilot/ch05-image-filtering/results/images`。工作目录为项目根目录；退出码：0。

```text
box3: n=1 max_abs_error=0 mismatches=0 PASS
median3: n=1 max_abs_error=0 mismatches=0 PASS
box3: n=9 max_abs_error=8.47710503e-07 mismatches=0 PASS
center_box=37.222221
median3: n=9 max_abs_error=0 mismatches=0 PASS
center_median=10.000000
box3: n=3185 max_abs_error=6.78168402e-06 mismatches=0 PASS
box MSE_to_clean=282.997097 (noisy=1513.018421)
median3: n=3185 max_abs_error=0 mismatches=0 PASS
median MSE_to_clean=18.025412 (noisy=1513.018421)
```

CPU 参考逐项核对，浮点示例同时检查非有限值。完整测试范围和容差见本章正文及源码。本机没有找到 Compute Sanitizer，因此没有 memcheck、racecheck 或 synccheck 通过结论。
