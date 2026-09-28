# 第 6 章验证记录

SSH 别名 `zyk` 对应主机 `ubuntu2404`；GPU RTX 5060 Ti，计算能力 12.0；nvcc 12.8.93；目标 `sm_120`。环境详情见 [第 1 章记录](../../ch01-getting-started/results/environment.md)。

从项目根目录配置、构建 `memory_layout` 并运行，退出码均为 0。连续/跨步读取各测试长度 7、1003；行跨度测试宽 5、高 3、实际 stride 8；AoS/SoA 各测试长度 7、1003；显式传输与 Unified Memory 各测试长度 7。全部输出与 CPU 参考逐项一致，`mismatches=0`。实际标准输出保存在 [正文](../README.md)。

未测性能、带宽、缓存命中或页面迁移。当前环境未找到 Compute Sanitizer 或 Nsight 工具。

2026-09-28 补测：根工程 `build/outline` 重新配置和全目标构建成功；`ctest --test-dir build/outline --output-on-failure --timeout 120` 退出码 0，共 18 项，17 项通过，1 项双卡测试 `ch10_two_devices` 因单卡环境跳过，0 项失败。新增 `ch06_memory_layout` 通过。另在 `build/ch06-layout-standalone` 独立配置、构建和 CTest，1/1 通过；手动运行 `memory_layout`，上述 11 行 CPU 对照均为 `mismatches=0 PASS`。首次根 CTest 因 SSH/TCP 失联而中断，未计入通过；这里记录的是恢复后的完整重跑。
