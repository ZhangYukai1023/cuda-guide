# 第 15 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置 `chapters/ch15-execution-resources/examples` 到 `build/ch15-resources-standalone` 并构建，退出码 0；独立 CTest 1/1 通过。手动运行两次，程序每次对两种输入模式、求和/直方图、全局原子/块内聚合、Block 64/128/256 共 24 组逐项做 GPU/CPU 精确对照，均输出 `PASS`；其中一次完整 26 行 stdout 存于 [run-2026-09-28.txt](run-2026-09-28.txt)。该次最终行为 `chapter 15 resource tradeoffs: PASS`。

根 `build/outline` 重新配置、全目标构建成功；完整 CTest 共 29 项，28 项通过、0 项失败，既有 `ch10_two_devices` 因仅一块 GPU 跳过，退出码 0，总时间 2.76 秒。

资源值来自 `cudaFuncGetAttributes` 和 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。本机输出的预测 Occupancy 都为 1.000，仅表示给定配置的理论驻留上限；不是实际活跃 Warp 的 profiler 读数。求和输出包含寄存器、静态加动态共享内存与 local memory 字节；直方图输出包含寄存器、静态共享内存与 local memory 字节。

采样时 `nvidia-smi --query-compute-apps` 显示 `/data/ComfyUI/.venv/bin/python` 占用约 5508 MiB 显存。Event 时间虽来自真实运行，却受并发任务影响，不能给策略或 Block 大小做可信排名；输出清零、分配、传输也不包含在 kernel Event 中。Nsight 实际活跃度、原子争用和寄存器溢出指标未测；待设备空闲且工具可用时复测。
