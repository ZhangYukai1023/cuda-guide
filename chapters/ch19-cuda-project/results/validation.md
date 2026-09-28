# 第 19 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

独立配置到 `build/ch19-project-standalone`、编译 `cuda_array_ops` 静态库、完成 `Linking CUDA device code ... cmake_device_link.o` 和最终可执行文件链接，退出码 0。独立 CTest `ch19_array_module` 1/1 通过，手动运行输出 `chapter 19 reusable array module: PASS`。

示例用页锁定 host 缓冲对长度 1003 和 7 的仿射输出逐项做 CPU 对照，并验证未等待时重复入队和超容量请求均被拒绝；`count=0` 按接口约定是无操作。异步错误由 `wait()` 传播，本次没有故意执行非法 GPU 访存或工具定位。

根 `build/outline` 重新配置并全目标构建成功；完整 CTest 33 项中 32 项通过、0 项失败，1 项既有双 GPU 用例因单卡跳过，退出码 0，总时间 3.20 秒。Driver API、NVRTC 与动态库没有在本章实现或测试。
