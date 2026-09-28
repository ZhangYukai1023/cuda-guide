# 第 5 章验证记录

在 SSH 别名 `zyk` 所连接的 `ubuntu2404` 上运行；GPU 为 RTX 5060 Ti，计算能力 12.0；nvcc 12.8.93；目标架构 `sm_120`。更完整环境见 [第 1 章记录](../../ch01-getting-started/results/environment.md)。

从项目根目录配置、构建 `warp_paths` 并运行，退出码均为 0。设备属性查询返回 `warp_size=32 sm_count=36`。64 个线程的 Warp 与 lane 编号、统一数据、奇偶路径、正负交替与分组输入，以及 1003 项输入的每块 64/128 线程版本，均与 CPU 参考逐项一致（`mismatches=0`）。标准输出全文见 [正文](../README.md)。

`predicate_mixed_warps` 由 Host 对输入符号分布建模，未测量 GPU 分支指令、吞吐量或速度。没有 profiler 采集；当前环境也未找到 Compute Sanitizer。全书 CTest 中 `ch05_warp_paths` 通过；本章使用新目录 `build/ch05-warp-standalone` 独立配置、构建和测试，1/1 通过。旧目录 `build/ch05-standalone` 属于 pilot 图像章节，CMake 缓存不能跨源码目录复用。
