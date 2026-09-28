# 第 15 章：执行效率与资源取舍

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch14-memory-optimization/README.md)

第 14 章改变访存路径，本章在**相同输入与输出定义**下改变并行组织方式。示例 [resource_tradeoffs.cu](examples/resource_tradeoffs.cu) 用两类整数任务：求和、16 桶直方图。每类任务分别有“所有线程直接更新全局结果”和“Block 内先聚合再更新全局结果”两个版本，Block 大小取 64、128、256。每组都与 CPU 精确比较，并记录 kernel Event 时间、编译器报告的资源及占用上限估计。已在 zyk 构建运行，24 组均通过 GPU/CPU 对照；同时有其他 GPU 任务，当前计时只作原始记录。

## 1. 两种归约策略

求和输入为 65539 项，长度刻意不是任一 Block 大小的整数倍。直接版让每个有效线程 `atomicAdd(output,input[i])`；块内版先把本块数据放入共享内存，越界线程放 0，逐轮折半求出块部分和，再由线程 0 原子加到全局输出。块内版把原本最多 65539 次全局原子更新减少为每块一次，但增加了共享内存读写与屏障。到底谁更快取决于竞争、块数、资源和设备，不从原子次数单独推断。

所有线程到达每轮 `__syncthreads()`；尾块越界线程不能提前返回。输出在每次测量前清零，清零动作**不计入**本章打印的 kernel Event 时间。若要比较整个归约任务，应把初始化输出也纳入第 12 章定义的任务边界。整数和的范围在本例中不会溢出 `int`；处理任意输入时须重新定义溢出约定。

## 2. 直方图的局部聚合

直方图直接版对 `bins[input[i]]` 做全局 `atomicAdd`。块内版先清零共享内存中的 16 个桶，所有线程同步，再在本块共享桶内原子计数，第二次同步后由前 16 个线程把局部结果合并到全局桶。输入一组循环取 0—15，另一组全为 0；全零会把更新集中到一个桶，便于观察竞争情形。CPU 为两个输入各算 16 桶参考数组，逐项精确比较。

“局部聚合”仍然使用原子操作：本块内对同一共享桶的更新需要原子性，块与块合并到同一全局桶也需要原子性。它减少全局争用，但可能增加共享内存争用与同步成本。跨步循环、更多每线程工作量或私有桶是其它策略，应单独改动并验证。

## 3. Occupancy 只是上限估计

示例用 `cudaFuncGetAttributes` 读取每线程寄存器数、静态共享内存与 local memory 字节；用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 在给定 Block 大小与动态共享内存条件下求可驻留 Block 上限，再以 `active_blocks*block/maxThreadsPerSM` 打印 `predicted_occupancy`。这只是**资源约束下的潜在驻留比例**，不是 profiler 测到的实际活跃 Warp，也不直接说明耗时。

增加 Block 大小可能减少块数量，却提高每块资源占用；更多寄存器或共享内存可能降低可驻留 Block 数。较高 Occupancy 有助于隐藏部分延迟，但不保证时间更短，可能已经达到足够的延迟隐藏。还应检查分支分化、原子竞争、指令级并行、寄存器溢出和缓存行为。循环展开可能减少循环控制，却增加寄存器压力；本章未把任何展开策略称为已验证优化。[CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)对 Occupancy 与资源取舍给出解释。

## 4. 如何记录一次有效比较

示例每组预热 3 次、采样 10 次，输出中位数。输入和设备缓冲区准备在计时外；每次结果清零在 Event 起点之前。实验表至少应保存以下列：

| 列 | 意义 |
| --- | --- |
| 输入模式 | 循环桶或全零；影响原子竞争 |
| 任务与策略 | 求和/直方图，直接全局更新/块内聚合 |
| Block 大小 | 64、128、256；同时记录实际 Grid |
| 正确性 | CPU 对照应逐项一致 |
| 时间 | 多次样本与中位数；说明是否包含输出清零 |
| 资源 | 每线程寄存器、共享内存、local memory、预测 Occupancy |
| 环境 | GPU、驱动、nvcc、优化级别、其他负载 |

如果“资源占用改善但程序没有更快”，应如实记录：可能原瓶颈是原子竞争、指令数或数据规模，或者时间差小于波动。第 13 章的 Nsight 指标可以帮助否定或支持具体假设。不能为了提高 Occupancy 就无依据地强制减少寄存器；溢出可能使 local memory 流量增加。

## 5. 构建与验证

在仓库根目录：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target resource_tradeoffs -j2
./build/outline/book_ch15/resource_tradeoffs
ctest --test-dir build/outline -R '^ch15_resource_tradeoffs$' --output-on-failure
```

也可独立构建 `chapters/ch15-execution-resources/examples`。[验证记录](results/validation.md)说明测试结果和限制，[完整运行输出](results/run-2026-09-28.txt)保留 24 组原始行。当前 GPU 有并发任务，不能依据这一次输出给版本或 Block 大小做稳定排名。

## 6. 常见错误

- 全局输出不清零，后一次测量累积前一次结果。
- 用普通 `bins[value]++` 代替原子更新，产生丢计数。
- 尾块在 `__syncthreads()` 前提前退出，让参与同步的线程不一致。
- 两个版本处理不同输入或不同长度，却比较耗时。
- 将 API 的预测 Occupancy 当成 profiler 实测值，或把它当成唯一优化目标。
- 强制限制寄存器后只看 Occupancy，忽略 local memory 溢出和实际时间。
- 把输出清零排除在 kernel Event 外，却声称结果是完整任务耗时。

## 7. 练习与参考答案

1. `n=65539`、Block 128 时 Grid 需要多少块？答：`ceil(65539/128)=513`，最后一块只有 3 个有效项。
2. 全零直方图的参考输出是什么？答：桶 0 为 65539，其他 15 桶为 0。
3. 块内求和为何在尾块给越界线程写 0？答：0 是加法中性元，保证所有线程都参与同步但不改变结果。
4. 若 Block 128 预测 Occupancy 高于 256，而中位时间相同，能否说 128 一定更优？答：不能；应看样本波动和完整任务时间，两者可能无可辨别差异。
5. 如何验证减少全局原子更新确实有用？答：固定输入、Block、编译环境，比较直接版与块内版的正确结果、时间分布，并采集原子竞争/内存相关指标；把额外同步与共享资源也计入解释。

[下一章：矩阵乘优化贯穿案例](../ch16-matmul-case/README.md)用矩阵乘串起正确性、数据重用、分块和可选库对照。
