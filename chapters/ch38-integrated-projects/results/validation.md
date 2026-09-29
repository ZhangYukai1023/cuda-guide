# 第 38 章验证记录

2026-09-28/29 在 zyk，RTX 5060 Ti（计算能力 12.0）、nvcc 12.8.93。项目 B/C 独立 CMake 构建与 CTest 2/2 通过。B 默认 7×11→7×5 三阶段 GPU 对 CPU double 参考最大误差依次为 Softmax 3.94312e-08、LayerNorm 3.44437e-07、GEMM 7.64627e-08，Softmax 最大行和误差 7.46222e-08，完整输出见[run-b-2026-09-28.txt](run-b-2026-09-28.txt)。C 默认 67×51、50 步 max_abs_error=1.34929e-07、sum_error=1.57983e-06，命令带 --out 写出了 heat.f32、heat.pgm、metrics.json，见[默认输出](run-c-2026-09-28.txt)与[文件输出](run-c-files-2026-09-28.txt)。

项目 A 用根工程第 26 章 image_pipeline 验收六帧 17×13 输入，输出 9×7 预览与 .f32、manifest、SHA256 均通过，归一化最大误差 5.94e-08；构建目录保留 report.json、原始输入和输出，stdout 见[run-a-2026-09-28.txt](run-a-2026-09-28.txt)。项目 D 检测到只有一张物理 GPU，打印 SKIP 并以 77 退出，未运行双进程或双卡对照，见[run-d-2026-09-28.txt](run-d-2026-09-28.txt)。A/D Python 文件语法检查通过；不能据此把 D 标为通过。

根工程重新配置、全目标构建及 CTest 53 项中 51 项通过、0 项失败，第 10、35 章双 GPU 用例各 1 项跳过，总时间 17.38 秒。A/D 是章外脚本验收，未计入根 CTest 53 项。GPU 有并发负载，B/C 单次计时与 D 未执行均不构成性能结论。Compute Sanitizer、真实双卡、其他机器部署及跨环境兼容性仍未测。
