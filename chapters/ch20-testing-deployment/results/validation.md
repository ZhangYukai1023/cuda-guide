# 第 20 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。脚本运行时仓库 HEAD 为 `9af9358`；第 20 章尚未提交，因此报告里的 `git status` 如实包含本章目录。

`python3 -m py_compile chapters/ch20-testing-deployment/examples/release_check.py` 成功。对 `build/outline` 运行脚本并请求第 12 章基准，报告写入忽略构建目录 `build/reports/ch20-correctness-benchmark.json`，命令退出码 0、`ok=true`。JSON 记录 `cmake --build`、全量 CTest、基准程序、GPU 和 nvcc 探测返回码均为 0。CTest 共 33 项：32 项通过、0 项失败，既有 `ch10_two_devices` 因单卡跳过，总时间 3.18 秒。基准解析出长度 7/4096/1048576 的 `wall_ms` 分别为 0.0088/0.0205/0.4655 ms；这只是受干扰的原始采集值，不是合格性能基线。

另请求 `--sanitizer-binary build/outline/book_ch10/debug_cases`，报告写入 `build/reports/ch20-required-memcheck.json`。构建与 CTest 均返回 0，但 `compute-sanitizer` 不可用，`memcheck.returncode=null`、`stderr=compute-sanitizer unavailable`，脚本退出码 1、`ok=false`，与必需门禁语义一致。没有执行 memcheck，也没有工具定位结论。

两份完整 JSON 保存在 zyk 的 `build/reports/` 中，未提交大体积原始日志。测量期间另有 `/data/ComfyUI/.venv/bin/python` 占用约 5508 MiB GPU 内存，尚未建立可信基线或进行性能回归比较。没有第二目标环境，跨机器安装与架构兼容性未测。
