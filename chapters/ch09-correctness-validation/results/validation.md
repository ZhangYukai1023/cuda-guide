# 第 9 章验证记录

2026-09-28（Asia/Shanghai）在 zyk 的 `codex/align-outline-38` 分支验证。RTX 5060 Ti，计算能力 12.0；nvcc 12.8.93，目标 `sm_120`。第 9 章本机草稿同步前远程目标目录不存在，原有 `.vscode/` 与 `README.html` 未触及。

独立构建：`cmake -S chapters/ch09-correctness-validation/examples -B build/ch09-correctness-standalone -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release`，构建退出码 0，CTest `ch09_correctness_validation` 1/1 通过。手动运行输出：

```text
integer n=0: PASS (documented empty result; no kernel launch)
float n=0: PASS (documented empty result; no kernel launch)
integer n=1: PASS mismatches=0
float n=1 seed=0x00c0ffee: PASS mismatches=0
integer n=7: PASS mismatches=0
float n=7 seed=0x00c0ffee: PASS mismatches=0
integer n=1003: PASS mismatches=0
float n=1003 seed=0x00c0ffee: PASS mismatches=0
integer n=4097: PASS mismatches=0
float n=4097 seed=0x00c0ffee: PASS mismatches=0
special NaN/Inf: PASS mismatches=0
sum order: sequential=1 adjacent_pairwise=0 double_reference=2
half roundtrip 1.0001 -> 1: PASS
logical index base=4294967419 n=8: PASS mismatches=0
chapter 9 validation: PASS
```

根 `build/outline` 重新配置和全目标构建成功；`ctest --test-dir build/outline --output-on-failure --timeout 120` 退出码 0：21 项中 20 项通过、0 项失败、1 项旧双卡测试因仅一张 GPU 跳过；新增 `ch09_correctness_validation` 通过，总时间 3.61 秒。

这些结果只证明本章已覆盖的输入及明确定义的比较规则。大逻辑下标检查只分配 8 项，不代表实际支持 4 GiB 以上数组；半精度往返没有执行半精度累加；没有测试所有 NaN payload 或跨编译器重现性，也没有可用的 Compute Sanitizer 记录。
