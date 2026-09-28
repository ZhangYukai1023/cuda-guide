# 第 23 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置到 `build/ch23-denoise-standalone`、构建成功，独立 CTest 1/1 通过。固定种子生成的椒盐与高斯噪声，各经过 3×3 中值、5×5 高斯和 5×5 双边滤波，共六条 GPU 路径全部 `beyond_tolerance=0 PASS`；本次双边的 `CPU_max_error=0`，中值和高斯也逐字节一致。默认运行完整九行 stdout 见 [run-2026-09-28.txt](run-2026-09-28.txt)。

本次固定干净图上，椒盐输入 PSNR 16.375 dB、SSIM8 0.318765；中值输出为 31.169 dB、0.916011。高斯噪声输入为 23.157 dB、0.649635；本参数下双边输出为 27.455 dB、0.769630。其余指标见原始 stdout；这些数字只对当前参考图、噪声种子、SSIM8 定义和参数有效。

以默认运行生成的 `clean.pgm` 作为可选 P5 输入再次运行，六条路径也均通过，完整 stdout 见 [run-user-pgm-2026-09-28.txt](run-user-pgm-2026-09-28.txt)。默认运行在忽略构建目录 `build/ch23-denoise-standalone/manual-images/` 写出 27 张 PGM（输入、输出、局部放大和相对干净图差异）；这些差异图是质量残差，不是 CPU/GPU 差异。外部图像格式、其它尺寸和无参考场景未测。

根 `build/outline` 重新配置、全目标构建成功；完整 CTest 36 项中 35 项通过、0 项失败，1 项既有双 GPU 用例因单卡跳过，退出码 0，总时间 3.46 秒。单次 Event/任务耗时仅为原始观察值，未形成稳定性能基线。
