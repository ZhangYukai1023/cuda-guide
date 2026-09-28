# 第 16 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置 `chapters/ch16-matmul-case/examples` 到 `build/ch16-matmul-standalone` 并构建成功，独立 CTest 1/1 通过。手动运行两次；三个形状 `2×2×3`、`65×37×19`、`256×256×256` 上的 `naive`、`shared_tile`、`two_register_outputs` 共 9 个结果都与 CPU double 参考符合（`mismatches=0`，当前确定性输入的 `max_abs_error=0`）。一次完整 13 行输出保存于 [run-2026-09-28.txt](run-2026-09-28.txt)。

根 `build/outline` 重新配置、全目标构建成功；完整 CTest 30 项中 29 项通过、0 项失败，既有双 GPU 用例因单卡跳过，退出码 0，总时间 2.86 秒。

CMake 报告 `cuBLAS development library unavailable`。SGEMM 分支明确 `SKIP`，未编译、未运行，未与手写版本做库性能比较；库布局映射仍是源码设计，待依赖齐备时验证。测量时另有 `/data/ComfyUI/.venv/bin/python` 占用约 5508 MiB GPU 内存，当前 Event 时间只作原始数据，不能当作稳定性能排名。数据传输、分配及 CPU 参考不计入 Event；端到端时间与 Nsight 指标均未测。
