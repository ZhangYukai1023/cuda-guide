# 第 7 章验证记录

2026-09-28（Asia/Shanghai）在 zyk 的 `codex/align-outline-38` 分支验证。设备为 RTX 5060 Ti（计算能力 12.0），nvcc 12.8.93，目标 `sm_120`。本章源码来自先前未验证的本机草稿，先核对远程目标目录不存在，再单章同步；未覆盖 `.vscode/` 或 `README.html`。

独立构建：`cmake -S chapters/ch07-shared-memory/examples -B build/ch07-shared-standalone -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release`，随后 `cmake --build build/ch07-shared-standalone -j2`，退出码均为 0。独立 CTest `ch07_shared_memory` 1/1 通过。手动运行程序的 CPU 对照输出：

```text
reverse_block: n=8 max_abs_error=0 mismatches=0 PASS
reverse_warp: n=32 max_abs_error=0 mismatches=0 PASS
transpose_tiled: n=6 max_abs_error=0 mismatches=0 PASS
transpose_tiled: n=1 max_abs_error=0 mismatches=0 PASS
transpose_tiled: n=527 max_abs_error=0 mismatches=0 PASS
block_sums: n=1 max_abs_error=0 mismatches=0 PASS
block_sums: n=1 max_abs_error=0 mismatches=0 PASS
block_sums: n=1 max_abs_error=0 mismatches=0 PASS
block_sums: n=8 max_abs_error=0 mismatches=0 PASS
```

根工程 `build/outline` 重新配置、全目标构建成功；`ctest --test-dir build/outline --output-on-failure --timeout 120` 退出码 0，19 项中 18 项通过、0 项失败、1 项旧双卡测试因仅一张 GPU 跳过。新增 `ch07_shared_memory` 通过，总时间 3.39 秒。

未找到 `compute-sanitizer`，因此未运行 memcheck/racecheck/synccheck；没有主动运行存在数据竞争或错误屏障的坏例子。共享内存 bank 冲突、带宽、占用率和速度收益也未测，不能由本章正确性 PASS 推断性能。
