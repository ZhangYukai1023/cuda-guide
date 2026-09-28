# 第 21 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

先将公共头文件 `chapters/common/image_io.hpp` 与本章一同同步。独立配置到 `build/ch21-image-standalone`、构建成功，独立 CTest 1/1 通过。手动运行的六项默认操作均输出 `mismatches=0 PASS`：灰度 ROI 反色、亮度、阈值、RGB 转灰度、BGR 转灰度、RGBA 亮度且 alpha 保留；每项比较了全部有效像素及行填充字节。终行 `chapter 21 image layout: PASS`。

以本章生成的 `gray-input.pgm` 再运行可选用户 P5 输入路径，输出 `user_pgm width=5 height=3 kernel_ms=0.0060 task_ms=0.0246 file_read_ms=0.0116 file_write_ms=0.0103 mismatches=0 PASS`。这些计时是一次小图运行的原始数字，不作性能结论。输出分别保存在忽略构建目录 `build/ch21-image-standalone/manual-images/`（22 个 PGM/PPM 文件）和 `user-images/`（23 个文件，含可选反色结果）。程序写出 CPU/GPU 差异图，默认测试逐字节差异为零。仅测试了本章生成的 P5 文件；外部格式兼容性未测。

根 `build/outline` 重新配置并全目标构建成功；完整 CTest 34 项中 33 项通过、0 项失败，既有双 GPU 用例因单卡跳过，退出码 0，总时间 3.28 秒。
