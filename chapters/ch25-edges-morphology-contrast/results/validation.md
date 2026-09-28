# 第 25 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置到 `build/ch25-features-standalone` 并构建成功，独立 CTest 1/1 通过。5×5 手算图、7×9 常量图和 257×193 非整齐图上，Sobel、膨胀、腐蚀、开、闭与直方图均衡化共 18 条默认路径全部 `mismatches=0 PASS`（逐字节 CPU 参考）。完整 19 行 stdout 见 [run-2026-09-28.txt](run-2026-09-28.txt)。PGM、局部放大与 CPU/GPU 差异图位于忽略构建目录 `build/ch25-features-standalone/manual-images/`。程序先核对小图 Sobel 中心和亮点膨胀手算断言，再运行 GPU。

用生成的 `small-input.pgm` 作为可选 P5 输入再次运行，六条 `user-*` 路径也都 `mismatches=0 PASS`，完整 stdout 见 [run-user-pgm-2026-09-28.txt](run-user-pgm-2026-09-28.txt)。其他外部格式、大尺寸图和库实现对照未测。

根 `build/outline` 重新配置、全目标构建成功；完整 CTest 38 项中 37 项通过、0 项失败，1 项既有双 GPU 用例因单卡跳过，退出码 0，总时间 3.66 秒。单次 Event/任务墙钟只作原始观察值，未形成可靠性能基线。
