# 第 26 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置到 `build/ch26-pipeline-standalone`、构建成功，独立 CTest 1/1 通过。默认六帧 65×49 确定性图像在串行与双槽路径共 12 帧处理记录均 `preview_mismatches=0 norm_mismatches=0 norm_max_error=0 PASS`，输出尺寸 33×25，完整 stdout 见 [run-2026-09-28.txt](run-2026-09-28.txt)。

另将第 23 章生成的 `clean.pgm`、`gaussian-noise.pgm`、`salt-pepper.pgm` 组成 64×64 三帧输入目录，串行与双槽共六条记录也均通过 CPU 参考，输出 32×32，完整 stdout 见 [run-user-pgm-2026-09-28.txt](run-user-pgm-2026-09-28.txt)。`build/ch26-pipeline-standalone/user-output/manifest.tsv` 列出三帧顺序和 `.pgm`、`.f32` 文件名；三个 `.f32` 各 4096 字节，按小端 float32 读取各 1024 项，实测取值均在 [-1,1]。该构建目录共生成 7 个输出文件（3 预览、3 张量、1 清单）。

根 `build/outline` 重新配置、全目标构建成功；完整 CTest 39 项中 38 项通过、0 项失败，1 项既有双 GPU 用例因单卡跳过，退出码 0，总时间 3.76 秒。

本次 `serial_batch_ms` 与 `double_slot_batch_ms` 是一次运行的真实原始值，测量时 GPU 有其他任务，未预热或重复构成稳定性能基线，也没有 Nsight 时间线证明传输/计算重叠。输入目录只用本书生成的 P5 测试，外部真实图像、长视频或 NPP 同语义比较未测。
