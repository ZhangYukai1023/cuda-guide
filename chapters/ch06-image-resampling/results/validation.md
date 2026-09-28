# 实际验证记录

验证日期：2026-09-28；主机 ubuntu2404；RTX 5060 Ti；驱动 595.84；nvcc 12.8.93；GCC 13.3.0；CMake 3.28.3。架构参数为 120，Release 构建，未使用 fast-math。

命令：`./build/all/ch06/resampling chapters/ch06-image-resampling/results/images`。工作目录为项目根目录；退出码：0。

```text
tiny_bilinear: n=1 max_abs_error=0 mismatches=0 PASS
tiny_bilinear_value=12.500000
singleton: n=6 max_abs_error=0 mismatches=0 PASS
identity: n=825 max_abs_error=0 mismatches=0 PASS
nearest: n=3300 max_abs_error=0 mismatches=0 PASS
bilinear: n=3300 max_abs_error=0 mismatches=0 PASS
translated: n=825 max_abs_error=0 mismatches=0 PASS
```

CPU 参考逐项核对，浮点示例同时检查非有限值。完整测试范围和容差见本章正文及源码。本机没有找到 Compute Sanitizer，因此没有 memcheck、racecheck 或 synccheck 通过结论。
