# 第 5 章：执行模型与 Warp

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch04-memory-resources/README.md)

前四章已经能把数组结果算对。本章追问：线程如何组成 Warp，为什么相邻线程走不同分支值得关注，以及 Block 大小能改变什么。示例只验证计算结果与分组模型，不测加速比；资源占用率和调优留到第四篇。完整源码见 [warp_paths.cu](examples/warp_paths.cu)。

## 1. 从 64 个线程画出两个 Warp

一个 Block 含 64 个一维线程。当前服务器查询 `cudaDeviceProp.warpSize` 得到 32，因此块内线程 0—31 属于第一个 Warp，32—63 属于第二个。线程 37 的块内 Warp 编号是 `37/32=1`，在该 Warp 中的位置（lane）是 `37%32=5`。源码让每个线程写出自己的 `threadIdx.x/warpSize` 和 `threadIdx.x%warpSize`，再由 CPU 对照全部 64 项。

| 块内线程编号 | 0 | 1 | 31 | 32 | 37 | 63 |
| --- | --- | --- | --- | --- | --- | --- |
| Warp 编号 | 0 | 0 | 0 | 1 | 1 | 1 |
| lane 编号 | 0 | 1 | 31 | 0 | 5 | 31 |

本机还查询到 36 个 SM（Streaming Multiprocessor）。Block 会被分配到可用的 SM；一个 SM 可以容纳多个 Block，具体取决于资源限制。Block 之间必须可以独立运行，**不能假定块 0 先于块 1 完成**。本例通过每个线程写确定的数组下标来保证正确性，而不是依赖执行次序。NVIDIA 的 [CUDA 12.8 编程指南：硬件实现与 SIMT](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#hardware-implementation) 说明了 Warp 分组、分支路径与 Block 调度。

SIMT（Single Instruction, Multiple Threads）让程序员写每个线程自己的标量逻辑，硬件以 Warp 为单位安排指令。`threadIdx.x` 是线程的逻辑编号；一个 Warp 不是“一个独占 CPU 核”。本章不需要根据线程编号推算它会在哪个 SM 上运行。

## 2. 最小示例：所有线程走同一条件路径

`classify` 根据输入值选择公式：非负数算 `2*x+1`，负数算 `3*x-1`。第一组 64 个输入全是 5，所以 CPU 手算每项应为 11：

```cpp
if (input[i] >= 0) output[i] = 2 * input[i] + 1;
else output[i] = 3 * input[i] - 1;
```

所有有效线程的谓词都是“非负”。程序打印 `uniform ... PASS`，表示 GPU 输出与 CPU 参考逐项一致；另打印 `predicate_mixed_warps=0`，这是 Host 根据输入符号和查询到的 Warp 大小计算的分组模型，**不是** GPU 分支指令的实测计数。

## 3. 递进：奇偶线程与数据条件

奇偶示例根据全局下标决定结果：偶数位置 `input[i]+10`，奇数位置 `input[i]-10`。输入从 0 到 63，前四项手算为 `[10,-9,12,-7]`，实际 GPU 结果与 CPU 完全一致。这个条件让相邻 lane 的布尔值交替。

再比较两组同样含 32 个 `+5`、32 个 `-5` 的输入：

```text
交替：+5,-5,+5,-5,...       每个 32 元素 Warp 都含两种符号
分组：前 32 项 +5，后 32 项 -5  第一个 Warp 只含正值，第二个只含负值
```

两种输入的正确输出值都是正数位置 11、负数位置 -16。Host 模型分别数出 2 个和 0 个“同时含两种谓词的 Warp”。如果编译器生成实际控制流分支，同 Warp 内不同线程走不同路径可能使路径分段执行；不同 Warp 可以独立选择路径。**这个输出不能证明实际机器码一定发生了分支分化**：短条件可能被编译器改成谓词指令，需在后续性能分析章节用合适工具查看。不能把模型数值直接当成性能测量。

## 4. Block 大小、任务规模与边界

同一组 64 个“分组”输入分别用每块 64、128 个线程运行，结果相同。128 线程的 Block 中只有前 64 个线程对应有效元素，其余线程被 `i<n` 排除。这个例子说明启动容量可以大于数据长度，但不代表多启动的线程提供了有用工作。

接着用长度 1003 的确定性输入 `i%7-3`，分别以每块 64、128 线程运行。两者都覆盖全部元素并与 CPU 完全一致；Host 模型都数出 32 个含两种谓词的 Warp。改变 Block 大小改变 Grid 中 Block 数、边界线程和资源需求，**本章没有测量哪种更快**。决定 Block 尺寸时，先确认正确性和设备上限，再到第四篇用可重复基准分析资源占用与执行效率。

## 5. 构建、运行与实测输出

在服务器项目根目录执行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target warp_paths -j2
./build/outline/book_ch05/warp_paths
ctest --test-dir build/outline -R '^ch05_warp_paths$' --output-on-failure
```

也可用 `-S chapters/ch05-warp-execution/examples -B build/ch05-warp-standalone` 独立配置。当前服务器实际输出：

```text
warp_size=32 sm_count=36
warp_ids: n=64 max_abs_error=0 mismatches=0 PASS
lane_ids: n=64 max_abs_error=0 mismatches=0 PASS
uniform: n=64 max_abs_error=0 mismatches=0 PASS
uniform: predicate_mixed_warps=0 (host model, not measured branch instructions)
parity: n=64 max_abs_error=0 mismatches=0 PASS
data_alternating: n=64 max_abs_error=0 mismatches=0 PASS
data_alternating: predicate_mixed_warps=2 (host model, not measured branch instructions)
data_grouped: n=64 max_abs_error=0 mismatches=0 PASS
data_grouped: predicate_mixed_warps=0 (host model, not measured branch instructions)
data_grouped_block128: n=64 max_abs_error=0 mismatches=0 PASS
data_grouped_block128: predicate_mixed_warps=0 (host model, not measured branch instructions)
irregular_block64: n=1003 max_abs_error=0 mismatches=0 PASS
irregular_block64: predicate_mixed_warps=32 (host model, not measured branch instructions)
irregular_block128: n=1003 max_abs_error=0 mismatches=0 PASS
irregular_block128: predicate_mixed_warps=32 (host model, not measured branch instructions)
```

详细环境、验证范围与未测部分见 [验证记录](results/validation.md)。没有进行 GPU 性能计时或 profiler 采集。

## 6. 常见错误

- 把“同一 Warp 的线程通常协同执行”误用为无同步的数据交换保证。现代架构支持独立线程调度；线程协作需使用明确的同步原语，后续章节讲解。
- 让 Block 1 等待 Block 0 写某个普通内存位置。普通 Grid 不保证 Block 顺序，这种设计可能挂起或读到错误数据；可拆成分阶段 kernel。
- 看到 `predicate_mixed_warps=2` 就宣称硬件分支指令一定分化，或宣称程序变慢。这个数只描述输入谓词分布，既不是机器码，也不是性能指标。
- 只用单一 Block 大小、单一输入分布做一次测量后推广到所有任务。本章未做性能结论。
- 将块内 Warp 编号 `threadIdx.x/warpSize` 当作整个 Grid 的唯一编号；不同 Block 可以有相同的块内 Warp 编号。

## 7. 练习与参考答案

1. Warp 大小为 32 时，块内线程 70 属于哪个 Warp、哪个 lane？答：Warp 2，lane 6。
2. 64 项全为正值，按本章谓词模型有几个混合 Warp？答：0。
3. 64 项中正负交替，按 32 项一个 Warp 有几个混合 Warp？答：2；这仍不等于实测机器分支数。
4. 把分组输入改为前 16 项正、后 48 项负，两个 Warp 中几个同时含正负？答：第一个 Warp 混合，第二个全负，共 1 个。
5. 长度 1003、每块 64 线程至少需要几个 Block？改为每块 128 呢？答：分别是 16 个和 8 个。两种都必须保留 `i<n` 边界检查。

下一章将把线程与实际内存位置对应起来，比较连续、跨步和 AoS/SoA 数据布局。
