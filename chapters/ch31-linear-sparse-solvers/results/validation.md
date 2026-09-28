# 第 31 章验证记录

2026-09-28 在 zyk，RTX 5060 Ti（计算能力 12.0）、nvcc 12.8.93，独立 CMake 配置、构建及 CTest 1/1 通过。普通 CUDA 教学路径：非对称 3×3 稠密 GEMV 精确得到 (14,23,27)；32 阶、94 非零 CSR SpMV 对 CPU 参考最大绝对误差 3.52859497e-07；GPU SpMV + CPU 标量递推的混合式 CG 在 11 轮收敛，CPU 重新计算的真相对残差 3.16530871e-07，最大解误差 6.79972156e-07。原始输出见[run-reference-2026-09-28.txt](run-reference-2026-09-28.txt)。

zyk Toolkit 未提供 cuBLAS、cuSPARSE、cuSOLVER 开发头文件和库，原库示例 linear_solvers target 未构建、未运行；LU、库 SpMV 和库 CG 不能声称通过。混合式 CG 每轮有 H2D/D2H 往返，不代表高性能 GPU 求解。根工程重新配置、全目标构建及 CTest 44 项中 43 项通过、0 项失败；第 10 章双 GPU 用例单卡跳过，总时间 4.24 秒。未做性能基准，也未测试不规则 CSR 分布、非 SPD 失败路径或库版本对照。
