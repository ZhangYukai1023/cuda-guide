# 实际验证记录

验证日期：2026-09-28；主机 ubuntu2404；RTX 5060 Ti；驱动 595.84；nvcc 12.8.93；GCC 13.3.0；CMake 3.28.3。架构参数为 120，Release 构建，未使用 fast-math。

命令：`./build/all/ch08/operators`。工作目录为项目根目录；退出码：0。

```text
matmul: n=4 max_abs_error=0 mismatches=0 PASS
22 28 49 64
matmul: n=323 max_abs_error=0 mismatches=0 PASS
softmax: n=3 max_abs_error=0 mismatches=0 PASS
softmax cols=1 row_sums PASS
softmax: n=9 max_abs_error=7.19476123e-08 mismatches=0 PASS
softmax cols=3 row_sums PASS
softmax: n=387 max_abs_error=6.24329424e-09 mismatches=0 PASS
softmax cols=129 row_sums PASS
softmax: n=771 max_abs_error=2.31778568e-09 mismatches=0 PASS
softmax cols=257 row_sums PASS
```

CPU 参考逐项核对，浮点示例同时检查非有限值。完整测试范围和容差见本章正文及源码。本机没有找到 Compute Sanitizer，因此没有 memcheck、racecheck 或 synccheck 通过结论。
