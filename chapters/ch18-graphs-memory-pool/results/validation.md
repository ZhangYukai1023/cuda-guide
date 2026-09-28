# 第 18 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置到 `build/ch18-graphs-standalone` 并构建成功，独立 CTest 1/1 通过。普通提交与 Graph 版各对长度 1003 的整数数组执行 100 次两段 kernel，并逐项与 CPU `2*x+1` 参考比较；流顺序 `cudaMallocAsync`、两段 kernel、`cudaFreeAsync` 路径也通过同一 CPU 对照，输出 `stream_ordered_allocator: PASS`。一次完整原始 stdout 见 [run-2026-09-28.txt](run-2026-09-28.txt)。

根 `build/outline` 重新配置、全目标构建成功，完整 CTest 32 项中 31 项通过、0 项失败，1 项既有双 GPU 用例因单卡跳过，退出码 0，总时间 3.07 秒。

本次 100 对 kernel 的普通提交墙钟中位数约 0.264 ms，100 次 Graph 启动约 0.310 ms；Graph 捕获和实例化另约 0.086 ms。它们是真实运行的原始值，但测量时 `/data/ComfyUI/.venv/bin/python` 另占约 5508 MiB GPU 内存。不能据这一组受干扰的运行判定 Graph 稳态性能，也不能推断任意业务工作负载的收益；设备空闲后应复测多轮并记录分布。内存池这里只验证基本生命周期，未做分配性能比较。
