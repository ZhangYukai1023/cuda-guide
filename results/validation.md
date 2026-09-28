# 全书验证记录

日期：2026-09-28（Asia/Shanghai）。工作目录：/data2/cuda-guide。主机名：ubuntu2404。

## 环境

RTX 5060 Ti，计算能力 12.0，驱动 595.84；CUDA nvcc 12.8.93（/home/zhangyukai/.local/cuda/bin/nvcc）；GCC 13.3.0；CMake 3.28.3。初始详细记录见 [第 1 章环境记录](../chapters/ch01-getting-started/results/environment.md)。

没有安装或升级工具。Compute Sanitizer、cuda-gdb、nsys、ncu 在已检查路径中未找到。只有一块可见 GPU。

## 构建和测试命令

```bash
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all -j2
ctest --test-dir build/all --output-on-failure
```

首次配置和编译成功。以下为加入单线程示例之前的构建与测试输出尾段（早期构建进度省略）：

```text
[ 50%] Linking CUDA executable benchmark
[ 50%] Built target benchmark
[ 54%] Building CUDA object ch06/CMakeFiles/resampling.dir/resampling.cu.o
[ 59%] Linking CUDA executable filtering
[ 59%] Built target filtering
[ 63%] Building CUDA object ch07/CMakeFiles/pipeline.dir/pipeline.cu.o
[ 68%] Linking CUDA executable resampling
[ 68%] Built target resampling
[ 72%] Building CUDA object ch08/CMakeFiles/operators.dir/operators.cu.o
[ 77%] Linking CUDA executable pipeline
[ 77%] Built target pipeline
[ 81%] Building CUDA object ch09/CMakeFiles/heat.dir/heat.cu.o
[ 86%] Linking CUDA executable operators
[ 86%] Built target operators
[ 90%] Building CUDA object ch10/CMakeFiles/multi_gpu.dir/multi_gpu.cu.o
[ 95%] Linking CUDA executable heat
[ 95%] Built target heat
[100%] Linking CUDA executable multi_gpu
[100%] Built target multi_gpu
Internal ctest changing into directory: /data2/cuda-guide/build/all
Test project /data2/cuda-guide/build/all
      Start  1: device_info
 1/12 Test  #1: device_info ......................   Passed    0.12 sec
      Start  2: vector_add
 2/12 Test  #2: vector_add .......................   Passed    0.21 sec
      Start  3: ch2_indexing
 3/12 Test  #3: ch2_indexing .....................   Passed    0.19 sec
      Start  4: ch3_transpose
 4/12 Test  #4: ch3_transpose ....................   Passed    0.21 sec
      Start  5: ch4_benchmark
 5/12 Test  #5: ch4_benchmark ....................   Passed    0.24 sec
      Start  6: ch5_filtering
 6/12 Test  #6: ch5_filtering ....................   Passed    0.20 sec
      Start  7: ch6_resampling
 7/12 Test  #7: ch6_resampling ...................   Passed    0.18 sec
      Start  8: ch7_pipeline
 8/12 Test  #8: ch7_pipeline .....................   Passed    0.19 sec
      Start  9: ch8_operators
 9/12 Test  #9: ch8_operators ....................   Passed    0.19 sec
      Start 10: ch9_heat
10/12 Test #10: ch9_heat .........................   Passed    0.19 sec
      Start 11: ch10_multi_gpu
11/12 Test #11: ch10_multi_gpu ...................   Passed    0.18 sec
      Start 12: ch10_two_devices
12/12 Test #12: ch10_two_devices .................***Skipped   0.08 sec

100% tests passed, 0 tests failed out of 12

Total Test time (real) =   2.18 sec

The following tests did not run:
	 12 - ch10_two_devices (Skipped)
```

初始记录统计为 11 项通过、1 项跳过、0 项失败。双卡跳过不代表已验证。单独运行：

```text
$ ./build/all/ch10/multi_gpu --require-two
SKIP: two CUDA devices required
exit code: 77
```

## 独立构建检查

另外按第 5 章提供的独立构建方式配置 build/ch05-standalone，成功构建并通过该章 CTest。没有声称每个独立目录都单独配置过；全书统一构建覆盖全部目标。

## 数值与图像

各章 results/validation.md 保留独立运行的输出，均为实际结果。图像章节的输入和输出由程序生成，PGM 与 SVG 已归档。scripts/render_images.py 从归档 PGM 生成 12 张 PNG 及 1 张去噪对照图；仅使用 Python 标准库。对照图已人工查看，左上参考、右上加噪、左下均值、右下中值，与程序数值结果一致。

CPU 参考覆盖下标、转置、仿射计算、均值与中值、最近邻与双线性、平移、双缓冲多帧、矩阵乘法、Softmax、热扩散及单卡分片。浮点示例检查绝对/相对误差和非有限值，细节见各章。

只有第 4 章测量性能：5 次预热、100 次重复，分别记录 CUDA event 序列均值与已分配缓冲区下 H2D+kernel+D2H 的端到端均值。其他章节及 CTest 的进程时间不作为性能结论。

## 限制

- 未执行 Compute Sanitizer、CUDA 专用断点调试或 profiler。
- 未验证真实双卡、P2P、NCCL/MPI 或多卡性能。
- 原始 38 章大纲现已同步到仓库的 outline.md。当前 10 章为压缩初版，章节安排与原大纲不一致。
- Git 仓库已初始化，尚未建立远程仓库。

## 单线程示例加入后的重新验证

加入第 2 章 `single_thread` 目标后，重新配置、构建全书并执行 CTest，命令退出码均为 0。当前总数为 13 项：12 项通过，1 项双卡测试因设备不足跳过。新增测试输出：

```text
Start  4: ch2_single_thread
4/13 Test  #4: ch2_single_thread ................   Passed
100% tests passed, 0 tests failed out of 13
The following tests did not run:
  13 - ch10_two_devices (Skipped)
```

第 2 章的 [验证记录](../pilot/ch02-thread-indexing/results/validation.md) 保存了三项 CPU 对照的实际输出。

## 按原大纲整理后的验证

正式章节已建立第 1、2 章，原压缩版示例保存在 `pilot/`。使用新的 `build/outline` 目录完成配置与全目标构建，退出码均为 0。`ctest --test-dir build/outline --output-on-failure` 实测 14 项：13 项通过，1 项双 GPU 测试因仅一块可见 GPU 跳过，0 项失败。正式第 2 章的 `ch02_first_threads` 通过。

另按 `chapters/ch02-first-thread/examples` 独立配置、构建并运行 CTest，1/1 通过。第 2 章程序与 CPU 预期结果的详细输出见 [该章验证记录](../chapters/ch02-first-thread/results/validation.md)。

## 第 3 章加入后的验证

重新构建并运行 `ctest --test-dir build/outline --output-on-failure`，实测 15 项：14 项通过，1 项双 GPU 测试跳过，0 项失败。新增 `ch03_array_indexing` 测试通过。另按 `chapters/ch03-array-indexing/examples` 独立配置、构建并运行 CTest，1/1 通过。第 3 章的整数 CPU 对照记录见 [该章验证记录](../chapters/ch03-array-indexing/results/validation.md)。

## 第 4 章加入后的验证

重新构建并运行 `ctest --test-dir build/outline --output-on-failure`，实测 16 项：15 项通过，1 项双 GPU 测试因只有一块可见 GPU 跳过，0 项失败。新增 `ch04_memory_resources` 测试通过。另按 `chapters/ch04-memory-resources/examples` 独立配置、构建并运行 CTest，1/1 通过。计时口径和该次测量值见 [第 4 章验证记录](../chapters/ch04-memory-resources/results/validation.md)。

## 第 5 章加入后的验证

重新构建并运行 `ctest --test-dir build/outline --output-on-failure`，实测 17 项：16 项通过，1 项双 GPU 测试因只有一块可见 GPU 跳过，0 项失败。新增 `ch05_warp_paths` 测试通过。独立构建目录为 `build/ch05-warp-standalone`，该章 CTest 1/1 通过。`build/ch05-standalone` 原先用于 pilot 图像章节，不能复用其 CMake 缓存；没有删除旧构建产物。见 [第 5 章验证记录](../chapters/ch05-warp-execution/results/validation.md)。

## 第 6 章加入后的验证

2026-09-28 在 `codex/align-outline-38` 分支上重新配置并构建 `build/outline` 全目标，命令退出码均为 0。完整运行 `ctest --test-dir build/outline --output-on-failure --timeout 120`，退出码 0：18 项测试中 17 项通过、0 项失败、1 项 `ch10_two_devices` 因仅一张 GPU 跳过；新增 `ch06_memory_layout` 通过，总时间 3.28 秒。此前一次 CTest 在第 2 项期间因 SSH/TCP 失联中断，未用作通过证据。

第 6 章另在 `build/ch06-layout-standalone` 独立配置、构建、运行 CTest，1/1 通过；手动运行可执行文件，连续/跨步、行跨度、AoS/SoA、显式传输和 Managed 11 行 CPU 对照均为 `mismatches=0 PASS`，详情见[本章验证记录](../chapters/ch06-memory-layout/results/validation.md)。没有测量缓存、带宽或页面迁移；没有安装新工具。

## 第 7 章加入后的验证

2026-09-28 同步第 7 章草稿后，在 `build/ch07-shared-standalone` 独立配置、构建并运行 CTest，1/1 通过；手动运行 8 线程/Warp 反转、3×2/1×1/31×17 分块转置和四种长度的块内求和，均与 CPU 参考完全一致（`mismatches=0`）。[章节记录](../chapters/ch07-shared-memory/results/validation.md)保留输出与限制。

根 `build/outline` 重新配置并全目标构建，完整 CTest 19 项中 18 项通过、1 项双卡测试因设备不足跳过、0 项失败，退出码 0，总时间 3.39 秒。当前未找到 Compute Sanitizer，坏例子工具检查与性能分析均未测。

## 第 8 章加入后的验证

2026-09-28 同步第 8 章后，在 `build/ch08-reduction-standalone` 独立配置、构建、CTest，1/1 通过；手动运行长度 1/7/128/1003 的两阶段求和与最大值下标、16 桶循环和全零直方图，均与 CPU 参考一致，详见[章节记录](../chapters/ch08-reduction-atomics/results/validation.md)。

根 `build/outline` 重新配置、全目标构建，完整 CTest 共 20 项：19 项通过、0 项失败、1 项双卡测试因设备不足跳过；退出码 0，总时间 3.53 秒。Compute Sanitizer 仍不可用，Scan/Gather/Scatter 的 GPU 版本和原子竞争性能均未测。

## 第 9 章加入后的验证

2026-09-28 同步第 9 章后，`build/ch09-correctness-standalone` 独立配置、构建、CTest 1/1 通过；手动运行空输入、长度 1/7/1003/4097、固定种子浮点、NaN/Inf、舍入顺序、半精度往返与大逻辑下标检查，结果见[章节记录](../chapters/ch09-correctness-validation/results/validation.md)。

根 `build/outline` 重新配置、全目标构建及完整 CTest 通过：21 项中 20 项通过、0 项失败、1 项双卡测试因设备不足跳过，退出码 0，总时间 3.61 秒。章节容差只用于当前数据与运算，不宣称一般数值精度保证。

## 第 10 章加入后的验证

2026-09-28 第 10 章 Release/Debug 独立配置、构建及安全模式 CTest 分别 1/1 通过；Debug CMake 已修复 `-G` 与 `-lineinfo` 冲突。根 `build/outline` 重新配置、全目标构建和完整 CTest 通过：22 项中 21 项通过、0 项失败、1 项双卡测试因设备不足跳过，退出码 0，总时间 3.78 秒。

当前环境没有可用的 Compute Sanitizer 或 CUDA-GDB，故障模式与单步定位**未测**。常规回归只包含 `safe` 模式，详见[章节记录](../chapters/ch10-debugging/results/validation.md)。

## 第 11 章加入后的验证

2026-09-28 第 11 章独立目录 `build/ch11-library-standalone` 配置、构建与 CTest 1/1 通过；Thrust 排序和 CUB 两个整数求和在 GPU 上通过 CPU 对照。当前环境无 cuBLAS 开发库，SGEMM 分支明确 `SKIP`、未编译也未运行；初始缺依赖配置和动态运行库加载失败已修复为可选 cuBLAS 与静态 CUDA 运行库链接，细节见[章节记录](../chapters/ch11-cuda-libraries/results/validation.md)。

根 `build/outline` 重新配置、全目标构建和完整 CTest 通过：23 项中 22 项通过、0 项失败、1 项双卡测试因设备不足跳过，退出码 0，总时间 4.05 秒。`ch11_library_baselines` 的通过仅覆盖 Thrust/CUB；不可把 cuBLAS `SKIP` 计入完成。
