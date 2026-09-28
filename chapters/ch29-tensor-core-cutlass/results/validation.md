# 第 29 章验证记录

2026-09-28 在 zyk，RTX 5060 Ti（计算能力 12.0）、nvcc 12.8.93，独立 CMake 配置、构建及 CTest 1/1 通过。三个形状（16×16×16、19×23×37、64×64×64）的朴素 FP16 输入/FP32 累加及 WMMA 两条路径，合计六组 GPU 输出与量化后输入的 CPU double 参考比较，均为 bad=0。非整齐尺寸补齐为 32×32×48。最大绝对误差：朴素 7.152557e-07，WMMA 4.768372e-07。完整原始输出见 [run-2026-09-28.txt](run-2026-09-28.txt)。

本机 CUDA Toolkit 缺 cuBLAS 开发头文件与库，GEMMEx 分支按预期报告 SKIP，未编译或运行。未找到现有 CUTLASS 头文件，CUTLASS 可选 target 未构建或运行。没有 cuobjdump/nvdisasm 或 Nsight 供实际指令确认，因此只能确认 WMMA API 路径的数值正确性，不能声称已观察硬件指令或稳定加速。

根工程重新配置、全目标构建、CTest 42 项中 41 项通过、0 项失败；第 10 章双 GPU 用例因单卡跳过。总时间 4.03 秒。章内 kernel_ms 是单次 Event 时间，未预热与重复统计，GPU 上存在并发负载；不据此给出性能结论。Compute Sanitizer memcheck 未测。
