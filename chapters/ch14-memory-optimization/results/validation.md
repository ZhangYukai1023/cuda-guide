# 第 14 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

## 构建与正确性

独立配置到 `build/ch14-memory-standalone`、构建和 CTest 退出码均为 0，`ch14_transpose_compare` 1/1 通过。手动运行 `./build/ch14-memory-standalone/transpose_compare` 的原始输出：

```text
width=16 height=16 variant=naive median_ms=0.0031 effective_GBps=0.653 mismatches=0 PASS
width=16 height=16 variant=shared_32x32 median_ms=0.0031 effective_GBps=0.670 mismatches=0 PASS
width=16 height=16 variant=shared_32x33 median_ms=0.0032 effective_GBps=0.650 mismatches=0 PASS
width=35 height=19 variant=naive median_ms=0.0030 effective_GBps=1.788 mismatches=0 PASS
width=35 height=19 variant=shared_32x32 median_ms=0.0031 effective_GBps=1.714 mismatches=0 PASS
width=35 height=19 variant=shared_32x33 median_ms=0.0031 effective_GBps=1.723 mismatches=0 PASS
width=1024 height=1024 variant=naive median_ms=0.0524 effective_GBps=159.990 mismatches=0 PASS
width=1024 height=1024 variant=shared_32x32 median_ms=0.0261 effective_GBps=321.058 mismatches=0 PASS
width=1024 height=1024 variant=shared_32x33 median_ms=0.0156 effective_GBps=539.391 mismatches=0 PASS
chapter 14 transpose comparison: PASS
```

每种形状的三个版本完整输出均与 CPU 转置结果逐项精确一致，`mismatches=0`。根 `build/outline` 重新配置并全目标构建成功；完整 CTest 28 项中 27 项通过、0 项失败、1 项既有双 GPU 测试因单卡跳过，退出码 0，总时间 2.66 秒。

## 性能解释限制

程序先预热 3 次，再记录 20 次 CUDA Event 样本并取中位数；上表只是一次进程运行的结果。测量期间 `nvidia-smi --query-compute-apps` 报告 `/data/ComfyUI/.venv/bin/python` 占用约 5508 MiB GPU 内存；因此不能把上述时间、有效 GB/s 或三个版本的相对顺序当作可信的空闲设备基线。尚未使用 Nsight Compute 测访存事务、bank 冲突和资源指标，也未测包含传输的端到端性能。需设备空闲后重复整组实验并保存分布与负载条件。
