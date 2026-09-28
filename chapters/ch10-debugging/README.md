# 第 10 章：调试工具与故障定位

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch09-correctness-validation/README.md)

第 9 章能判断答案错了，本章要定位为什么错。示例 [debug_cases.cu](examples/debug_cases.cu) 提供一个正常模式，以及四种**必须显式选择**的故障模式。常规 CTest 只运行正常模式；故障模式留给 Compute Sanitizer 单独检查。本章已在 zyk 构建并验证 `safe` 模式；Release、Debug 和根工程 CTest 通过。Compute Sanitizer 与 CUDA-GDB 在已检查的环境中不可用，四种故障模式的工具定位**未测**，详见[验证记录](results/validation.md)。

## 1. 把故障缩到最小

先记录输入长度、启动配置、随机种子、GPU/驱动与编译命令，再把失败缩成一个 Block、一个 Warp 或一个越界下标。本章每个坏例子只保留一个目标故障：

| 模式 | 故障点 | 对照修复 | 主检查工具 |
| --- | --- | --- | --- |
| `oob` | 8 项缓冲区写 `out[8]` | `safe_index` 用 `i<n` 守卫 | memcheck |
| `race` | 32 线程向同一个共享地址写 | `safe_shared_sum` 每线程写自己的槽并在读取前同步 | racecheck |
| `init` | 从分配后未写入的全局数组读取 | 先 `cudaMemset`，再用 `safe_initialized_read` | initcheck |
| `sync` | 线程 16 调用 `__syncwarp(0x0000ffff)`，但掩码不含自身 | `safe_warp_mask` 由参与条件生成掩码 | synccheck |

源程序在每次 kernel 后先调用 `cudaGetLastError()` 捕获启动配置错误，再调用 `cudaDeviceSynchronize()` 捕获异步执行错误。后者报错时，根因仍可能在之前的 kernel；因此保留两处检查及源码行号很重要。四个坏例子只为教学使用，不应作为生产程序的一部分。

## 2. 先验证正常基线

`safe` 模式验证四件事：8 项索引输出应为 `[0,2,...,14]`；共享内存求 `0+...+31` 应为 496；先清零的输入使输出为 `[0,...,7]`；Warp 掩码例子写出 1。任何一项不符或 CUDA 调用失败，进程退出码为 1。这使工具环境自身先有一个可检查的基线。

在根工程完成本章集成后运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target debug_cases -j2
ctest --test-dir build/outline -R '^ch10_debug_safe$' --output-on-failure
```

独立构建可用 `cmake -S chapters/ch10-debugging/examples -B build/ch10-debug-standalone` 并指定相同编译器与架构。上述安全模式命令已实际运行；故障模式没有纳入普通 CTest。

## 3. 分别检查四类错误

工具可用时，程序位于 `build/outline/book_ch10/debug_cases`，可逐项运行：

```bash
compute-sanitizer --tool memcheck --error-exitcode 86 \
  ./build/outline/book_ch10/debug_cases oob
compute-sanitizer --tool racecheck --error-exitcode 86 \
  ./build/outline/book_ch10/debug_cases race
compute-sanitizer --tool initcheck --error-exitcode 86 \
  ./build/outline/book_ch10/debug_cases init
compute-sanitizer --tool synccheck --error-exitcode 86 \
  ./build/outline/book_ch10/debug_cases sync
```

这些命令的目标是**发现错误**，所以返回非零或打印报告属于预期调查线索，不能把它们加入普通“必须通过”的 CTest。`--error-exitcode 86` 只在被测程序本身返回 0 而工具发现错误时指定工具退出码；被测程序自己因 CUDA 错误返回 1 时，应结合两种输出判断。不要只凭退出码推断是哪一行。每个工具有检查范围：memcheck 查越界/错位等访存问题；racecheck 重点查共享内存访问危险；initcheck 默认查未初始化的设备全局内存读取；synccheck 查同步原语误用。后面三个工具不能代替 memcheck。范围以 [Compute Sanitizer 官方文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) 为准。

`sync` 用了一个明确的错误掩码：线程 0—16 都走到 `__syncwarp`，而掩码只声明线程 0—15。它对应官方文档所述的 Invalid arguments 类型。坏例子可能表现出未定义行为；在共享 GPU 上先单独运行，不要与长作业并行，必要时用外层超时控制并保存报告。修复方法是让执行者与掩码描述的参与者一致，而非删掉同步指令。

## 4. 需要单步时再用 CUDA-GDB

先用 `-lineinfo` 让优化构建中的工具报告能定位到源码。若需要看线程、变量和逐行执行，另建 Debug 目录；本章 CMake 在 Debug 模式为 CUDA 源添加 `-g -G`：

```bash
cmake -S chapters/ch10-debugging/examples -B build/ch10-debug-gdb \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Debug
cmake --build build/ch10-debug-gdb -j2
cuda-gdb --args ./build/ch10-debug-gdb/debug_cases safe
```

Debug 构建及安全模式 CTest 已通过，但当前没有 `cuda-gdb`，未执行单步。工具可用时，在调试器里先断到 `safe_index` 等 kernel，再检查 `threadIdx.x`、`n` 和数组下标。`-G` 关闭许多设备端优化，调试构建的耗时不能当性能结果。具体版本、支持的设备和用法以 [CUDA-GDB 官方文档](https://docs.nvidia.com/cuda/cuda-gdb/) 为准。

## 5. 一条完整的故障记录

以越界模式为模板：

1. **现象**：正常 `safe` 通过，`oob` 在 memcheck 下报告越界写；保留工具版本、退出码和首条相关报告。
2. **复现输入**：1 个 Block、32 线程，目标缓冲区 8 个 `int`，只有线程 0 写下标 8。
3. **定位证据**：工具指向 `bad_oob` 的写入行；`0..7` 才是合法范围。
4. **修复**：加 `i<n` 保护，并让实际启动规模与 `n` 的约定一致。
5. **验证**：正常模式 CPU 对照全通过；memcheck 正常模式无该错误；再运行根工程回归。

这个记录模板同样适用于共享内存竞争、未初始化读取和同步掩码错误。工具报告是证据，修复后的对照测试才能说明问题已被解决。

## 6. 常见误区

- `cudaGetLastError()` 返回成功便断言 kernel 已正确完成；异步执行错误需要同步点才能观察。
- `racecheck` 没报告便断言所有全局内存并发写都安全；应了解它的共享内存检测范围。
- 只运行 `initcheck` 而不先排除越界读取，误把非法地址当成初始化问题。
- 在含部分线程的条件分支里放 `__syncthreads()`，让同一个 Block 不能一致到达屏障。
- 调试构建与 Release 构建结果或耗时不同，却不记录编译选项。
- 把故意失败的故障模式加入常规成功门禁，导致每次回归都红。

## 7. 练习与参考答案

1. 8 项缓冲区合法下标范围是什么？答：0—7；下标 8 是越界。
2. 32 个线程都写共享内存 `slot`，即使最后只由线程 0 读取，为什么仍不安全？答：多个写之间没有确定顺序，读到哪次写的结果不可作为正确性约定。
3. `cudaMalloc` 后立刻读取值为 0，能否认为内存自动清零？答：不能；应显式初始化或把输入复制到设备。
4. 线程 16 调用 `__syncwarp(0x0000ffff)` 哪里不一致？答：掩码只包括 0—15，未包含当前调用线程 16。
5. Sanitizer 给出越界写行号后，下一步怎样验证修复？答：保存最小复现，修边界守卫，正常模式与 CPU 对照，再在 memcheck 下运行正常模式并执行工程回归。

[下一章](../ch11-cuda-libraries/README.md)以成熟 CUDA 库为正确性和工程实现的对照，再判断何时值得自己写 kernel。
