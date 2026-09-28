# 第 8 章验证记录

2026-09-28（Asia/Shanghai）在 zyk 的 `codex/align-outline-38` 分支验证。设备 RTX 5060 Ti（计算能力 12.0），nvcc 12.8.93，目标 `sm_120`。从本机持久草稿按章同步前，远程目标目录不存在；`.vscode/` 和 `README.html` 未触及。

独立构建：`cmake -S chapters/ch08-reduction-atomics/examples -B build/ch08-reduction-standalone -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release`；构建退出码 0，CTest `ch08_reduction_atomic` 1/1 通过。手动运行程序输出：

```text
sum: n=1 max_abs_error=0 mismatches=0 PASS
max_index: n=1 value=-5 index=0 PASS
sum: n=1 max_abs_error=0 mismatches=0 PASS
max_index: n=7 value=1 index=6 PASS
sum: n=1 max_abs_error=0 mismatches=0 PASS
max_index: n=128 value=5 index=10 PASS
sum: n=1 max_abs_error=0 mismatches=0 PASS
max_index: n=1003 value=5 index=10 PASS
histogram_cycle: n=16 max_abs_error=0 mismatches=0 PASS
histogram_all_zero: n=16 max_abs_error=0 mismatches=0 PASS
```

`sum: n=1` 是第二阶段输出标量的长度，不是原输入长度；四组原输入长度依次为 1、7、128、1003。最大值并列时按最早下标比较。直方图两组分别覆盖循环桶和 1003 个零值的集中竞争。

根 `build/outline` 重新配置、全目标构建成功；`ctest --test-dir build/outline --output-on-failure --timeout 120` 退出码 0：20 项中 19 项通过、0 项失败、1 项旧双卡测试因仅一张 GPU 跳过；新增 `ch08_reduction_atomic` 通过，总时间 3.53 秒。

当前没有可用的 Compute Sanitizer，故未对故障模式运行 memcheck/racecheck/synccheck。Scan、Gather、Scatter 在正文中是语义练习，尚无本章 GPU 实现或设备测试；本章也未测原子竞争耗时。
