# 第 3 章验证记录

主机 `ubuntu2404`（本机 SSH 别名 `zyk`）；RTX 5060 Ti，计算能力 12.0；nvcc 12.8.93；构建目标 `sm_120`。完整环境见 [第 1 章环境记录](../../ch01-getting-started/results/environment.md)。

从项目根目录执行：

```bash
cmake -S . -B build/outline -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target array_indexing -j2
./build/outline/book_ch03/array_indexing
```

上述配置、编译、运行退出码均为 0。程序用 CPU 整数参考逐项核对 GPU 输出：`multiply` 与 `vector_add` 各测试长度 1、7、256、1003，全部 `mismatches=0`；`coordinates` 测试 3×2、1×1、37×19、19×37，全部 `mismatches=0`。实际标准输出与本章正文的输出区块一致。

未测性能；未找到 Compute Sanitizer，因此没有内存工具检查结论。全书 CTest 中 `ch03_array_indexing` 通过；本章单独配置、构建、CTest 也通过（1/1）。
