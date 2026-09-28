# 实际验证记录

验证日期：2026-09-28；主机 ubuntu2404；RTX 5060 Ti；驱动 595.84；nvcc 12.8.93；GCC 13.3.0；CMake 3.28.3。架构参数为 120，Release 构建，未使用 fast-math。

命令：`./build/all/ch10/multi_gpu`。工作目录为项目根目录；退出码：0。

```text
sharded_transform: n=1 max_abs_error=0 mismatches=0 PASS
devices_used=1 n=1
sharded_transform: n=17 max_abs_error=0 mismatches=0 PASS
devices_used=1 n=17
sharded_transform: n=1003 max_abs_error=0 mismatches=0 PASS
devices_used=1 n=1003
Single-device fallback only; multi-device path NOT verified.
```

CPU 参考逐项核对，浮点示例同时检查非有限值。完整测试范围和容差见本章正文及源码。本机没有找到 Compute Sanitizer，因此没有 memcheck、racecheck 或 synccheck 通过结论。

只有单卡回退路径通过验证。另运行 `multi_gpu --require-two`，打印 `SKIP: two CUDA devices required`；CTest 将退出码 77 记为跳过。没有验证双卡并发、P2P 或多卡性能。
