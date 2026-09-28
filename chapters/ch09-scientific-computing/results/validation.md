# 实际验证记录

验证日期：2026-09-28；主机 ubuntu2404；RTX 5060 Ti；驱动 595.84；nvcc 12.8.93；GCC 13.3.0；CMake 3.28.3。架构参数为 120，Release 构建，未使用 fast-math。

命令：`./build/all/ch09/heat chapters/ch09-scientific-computing/results/images`。工作目录为项目根目录；退出码：0。

```text
heat: n=9 max_abs_error=7.15255737e-07 mismatches=0 PASS
shape=3x3 steps=1 center=19.999998 range=[0.000000,19.999998]
heat: n=35 max_abs_error=0 mismatches=0 PASS
shape=7x5 steps=0 center=100.000000 range=[0.000000,100.000000]
heat: n=35 max_abs_error=7.15255737e-07 mismatches=0 PASS
shape=7x5 steps=1 center=19.999998 range=[0.000000,20.000000]
heat: n=35 max_abs_error=3.57627876e-07 mismatches=0 PASS
shape=7x5 steps=2 center=20.000000 range=[0.000000,20.000000]
heat: n=825 max_abs_error=1.41793282e-07 mismatches=0 PASS
shape=33x25 steps=20 center=1.952703 range=[0.000000,1.952703]
```

CPU 参考逐项核对，浮点示例同时检查非有限值。完整测试范围和容差见本章正文及源码。本机没有找到 Compute Sanitizer，因此没有 memcheck、racecheck 或 synccheck 通过结论。
