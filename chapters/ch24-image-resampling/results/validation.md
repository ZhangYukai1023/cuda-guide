# 第 24 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置到 `build/ch24-resample-standalone` 并构建成功，独立 CTest 1/1 通过。4×4→7×5 最近邻/双线性/双三次、4×4 旋转、8×8→3×3 棋盘格最近邻/面积缩小、常量图四种采样共十条默认路径均 `max_error=0 beyond_tolerance=0 PASS`。一次完整原始 stdout 见 [run-2026-09-28.txt](run-2026-09-28.txt)。程序在 GPU 前先核对小图中心最近邻 140、双线性 110 和旋转四角手算值；实际 GPU 对照通过。生成的 PGM、差异及局部放大图在忽略构建目录 `build/ch24-resample-standalone/manual-images/`。

另将本章生成的 `checker-8x8.pgm` 作为可选输入，执行 2 倍双线性放大，输出 `user-bilinear-2x 8x8 -> 16x16 max_error=0 beyond_tolerance=0 ... PASS`，完整 stdout 见 [run-user-pgm-2026-09-28.txt](run-user-pgm-2026-09-28.txt)。其他外部文件格式、较大输入和 NPP/OpenCV 同语义比较未测。

根 `build/outline` 重新配置、全目标构建成功；完整 CTest 37 项中 36 项通过、0 项失败，1 项既有双 GPU 用例因单卡跳过，退出码 0，总时间 3.58 秒。Event/任务计时均为单次运行原始观察值，未建立可靠性能基线。
