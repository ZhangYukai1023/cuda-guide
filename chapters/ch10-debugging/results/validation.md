# 第 10 章验证记录

2026-09-28（Asia/Shanghai）在 zyk 的 `codex/align-outline-38` 分支验证。RTX 5060 Ti，计算能力 12.0；nvcc 12.8.93，目标 `sm_120`。从本机草稿按章同步前远程目录不存在，`.vscode/` 与 `README.html` 未触及。

Release 独立构建目录 `build/ch10-debug-standalone`：配置、构建成功，CTest `ch10_debug_safe` 1/1 通过；手动执行 `debug_cases safe` 输出 `safe cases: PASS (index, shared, initialized, warp mask)`。Debug 目录 `build/ch10-debug-gdb` 使用 `-g -G` 构建；起初与 `-lineinfo` 同时启用导致 nvcc/ptxas 警告，已修改 CMake 使 Debug 只传 `-g -G`、Release 只传 `-lineinfo`，重构建无该警告，Debug CTest 1/1 通过。

根 `build/outline` 重新配置、全目标构建成功；`ctest --test-dir build/outline --output-on-failure --timeout 120` 退出码 0：22 项中 21 项通过、0 项失败、1 项旧双卡测试因仅一张 GPU 跳过；新增 `ch10_debug_safe` 通过，总时间 3.78 秒。

**工具验收未完成**：`command -v`、Toolkit `bin/` 和已检查的 CUDA 安装路径均未找到 `compute-sanitizer` 或 `cuda-gdb`。没有运行 `oob`/memcheck、`race`/racecheck、`init`/initcheck、`sync`/synccheck，也没有宣称坏例子必然产生某个报告。Debug 构建通过不等于已完成断点调试。当前不安装或升级 Toolkit；若后续环境提供工具，再保存每项原始报告、退出码和修复验证。
