# 实际验证记录

验证日期：2026-09-28；主机 ubuntu2404；RTX 5060 Ti；驱动 595.84；nvcc 12.8.93；GCC 13.3.0；CMake 3.28.3。架构参数为 120，Release 构建，未使用 fast-math。

命令：`./build/all/ch02/indexing`。工作目录为项目根目录；退出码：0。

```text
grid_stride: n=0 max_abs_error=0 mismatches=0 PASS
grid_stride: n=1 max_abs_error=0 mismatches=0 PASS
grid_stride: n=31 max_abs_error=0 mismatches=0 PASS
grid_stride: n=32 max_abs_error=0 mismatches=0 PASS
grid_stride: n=33 max_abs_error=0 mismatches=0 PASS
grid_stride: n=1003 max_abs_error=0 mismatches=0 PASS
image_coordinates: n=6 max_abs_error=0 mismatches=0 PASS
0 1 2 100 101 102
image_coordinates: n=703 max_abs_error=0 mismatches=0 PASS
image_coordinates: n=1 max_abs_error=0 mismatches=0 PASS
```

CPU 参考逐项核对，浮点示例同时检查非有限值。完整测试范围和容差见本章正文及源码。本机没有找到 Compute Sanitizer，因此没有 memcheck、racecheck 或 synccheck 通过结论。
