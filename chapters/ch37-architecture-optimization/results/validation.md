# 第 37 章验证记录

2026-09-28 在 zyk，RTX 5060 Ti（计算能力 12.0）、nvcc 12.8.93，以 sm_120 编译，独立 CMake 构建和 CTest 1/1 通过。运行时显示 async_binary_version=120，cp.async 路径实际执行而非 SKIP。N=1024 与 262144 两种规模，通用与 cp.async 输出相对 CPU 参考最大绝对误差均为 5.96046e-08，两条 GPU 路径逐项差异 0。完整单次实验输出见[run-2026-09-28.txt](run-2026-09-28.txt)。

ptxas 报告通用 kernel 用 14 寄存器、0 barrier；cp.async kernel 用 14 寄存器、1 barrier、1040 字节共享内存；均无 spill，原始报告见[build-2026-09-28.txt](build-2026-09-28.txt)。本机缺 cuobjdump/nvdisasm/Nsight，未核查最终 SASS 或 profiler 指标。单轮 100 次 Event 平均时间：N=1024 通用 0.00083776 ms、异步 0.00084736 ms；N=262144 通用 0.00228128 ms、异步 0.00275488 ms。本次异步版没有更快，GPU 有并发负载，也未做多轮独立测量，不据此给出稳定性能排名。TMA、集群、多级流水线均未实现或测量。

根工程重新配置、全目标构建及 CTest 51 项中 49 项通过、0 项失败，第 10、35 章双 GPU 用例各 1 项跳过，总时间 17.42 秒。
