# 第 2 章验证记录

在 SSH 别名 `zyk` 所连接的 `ubuntu2404` 上执行。GPU：RTX 5060 Ti，计算能力 12.0；nvcc：12.8.93；目标架构：`sm_120`。环境详情见 [第 1 章记录](../../ch01-getting-started/results/environment.md)。

```bash
cmake -S . -B build/outline -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target first_threads -j2
./build/outline/book_ch02/first_threads
```

配置、编译和运行退出码均为 0。实际程序输出：

```text
write_42: got=42 expected=42 PASS
add_scalars: got=12 expected=12 PASS
write_ids: got=0,1,2,3,4,5,6,7 PASS
```

前两例与 CPU 手算值 42 和 `7+5=12` 对照，第三例与 CPU 数组 `0..7` 逐项比较。测试输出不是性能测量。本机未找到 Compute Sanitizer，故未运行其 memcheck、racecheck 或 synccheck。全书 CTest 中 `ch02_first_threads` 通过；本章单独配置、构建、CTest 也通过（1/1）。
