# 第 13 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

## 构建与正确性

独立配置 `chapters/ch13-nsight-profiling/examples` 到 `build/ch13-nsight-standalone`，指定 nvcc 和架构 120；构建退出码 0。独立 CTest 3/3 通过，退出码 0。手动运行输出：

```text
SKIP NVTX ranges: nvtx3/nvtx3.hpp unavailable
profile case=tiny n=16384: PASS
SKIP NVTX ranges: nvtx3/nvtx3.hpp unavailable
profile case=batch n=16384: PASS
SKIP NVTX ranges: nvtx3/nvtx3.hpp unavailable
profile case=stride n=16384: PASS
```

程序内 `verify` 对三种模式的全部 16384 个 float 输出逐项与 CPU 参考比较。通过只说明当前输入和计算结果正确，不说明性能诊断通过。

根 `build/outline` 重新配置并全目标构建，退出码均为 0。完整 `ctest --test-dir build/outline --output-on-failure --timeout 120` 退出码 0：27 项中 26 项通过、0 项失败、1 项既有双 GPU 用例因单卡跳过；本章新增 `ch13_profile_tiny`、`ch13_profile_batch`、`ch13_profile_stride` 均通过，总时间 2.57 秒。

## 分析工具限制

`nvtx3/nvtx3.hpp` 当前头文件树中不可用，源码采用无标记的替代实现并明确打印 `SKIP`；NVTX 范围并未实测。`command -v nsys` 与 `command -v ncu` 均无输出，Nsight Systems/Compute 报告和 profiler 权限未检查，故时间线、CPU 提交空隙、kernel 指标、访存瓶颈假设均**未验证**。需在工具齐备的环境重编译并按正文命令采集。没有将 CTest 的进程时间用于性能结论。
