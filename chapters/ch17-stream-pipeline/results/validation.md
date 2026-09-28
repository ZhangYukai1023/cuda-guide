# 第 17 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置到 `build/ch17-stream-standalone` 并构建成功，独立 CTest 1/1 通过。1、2、4 槽各处理 10 批、每批 65536 项，全部输出与 CPU 逐项精确一致（`mismatches=0`），跨 Stream 合并首项值分别为预期的 10、19、34。一次完整 stdout 存于 [run-2026-09-28.txt](run-2026-09-28.txt)。

根 `build/outline` 重新配置、全目标构建成功，完整 CTest 31 项中 30 项通过、0 项失败，1 项既有双 GPU 测试因单卡跳过，退出码 0，总时间 3.00 秒。

计时为预热后 5 次完整十批处理的主机墙钟中位数，已包括传输、kernel、Event 等待与同步，不含缓冲区分配和输入填充。测量时另有 `/data/ComfyUI/.venv/bin/python` 占用约 5508 MiB GPU 内存，当前槽数间时间差不是可信性能基线。`nsys` 不可用，未采集 H2D、kernel、D2H 的时间线，**不能声称观察到了实际重叠**。
