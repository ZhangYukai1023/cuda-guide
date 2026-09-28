# 第 14 章：访存优化，从数据排列开始

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch13-nsight-profiling/README.md)

第 13 章建立了用 profiler 找瓶颈的方法；当前服务器没有 Nsight 工具，因此这里先用转置作可复现的访存实验。本章用矩阵转置研究地址布局：相同输入、相同输出定义，依次运行朴素全局内存转置、`32×32` 共享内存分块、`32×33` 填充共享内存分块。源码见 [transpose_compare.cu](examples/transpose_compare.cu)。三种实现已在 zyk 上编译并通过 GPU/CPU 逐项对照；当前 GPU 同时运行其他任务，性能差异尚未获得空闲设备复测。

## 1. 先确认索引语义

输入是行主序矩阵，宽 `W`、高 `H`，元素 `(y,x)` 位于 `input[y*W+x]`。转置输出的宽是 `H`、高是 `W`，同一值位于 `output[x*H+y]`。例如：

```text
输入 H=2, W=3       转置 H=3, W=2
[1 2 3]              [1 4]
[4 5 6]              [2 5]
                     [3 6]
```

每个版本都与 CPU 按这个公式产生的完整数组逐项比较。输入取整数值但存为 float，转置只搬运、不做算术，因此应精确相等。测试形状 `16×16`、`35×19`、`1024×1024` 分别覆盖小于一块、两个方向都不整齐以及较大整齐矩阵。尾块线程先检查输入坐标，块内所有线程仍到达 `__syncthreads()`；转置后再检查输出坐标，才读取共享内存并写出。

## 2. 三种访问路径

朴素版本按线程的 x 坐标连续读取 `input[y*W+x]`，但同一组线程写出 `output[x*H+y]` 时，写地址相隔 `H` 项。对于较大 `H`，相邻线程的写入不连续，可能形成较多全局内存事务。实际合并情况还受硬件、对齐、缓存与尺寸影响，应看第 13 章的访存指标，而非只看源码推断倍数。

共享内存版本先让 Block 按行把一个 `32×32` tile 读入 `tile[threadIdx.y][threadIdx.x]`，整块同步后交换线程坐标再读 `tile[threadIdx.x][threadIdx.y]`，使输出写入也按连续方向排列。`tile[32][32]` 在读“列”时可能让一个 Warp 的多个地址映射到相同 bank。把第二维改为 33，行跨度增加一项，可打散这一典型模式中的 bank 冲突。两版的全局输入输出和计算语义完全一致，差别只有共享数组的行跨度。[CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)给出合并访存与共享内存填充的原理和示例；特定设备上的收益仍要实测。

## 3. 别让一个优化引出另一个瓶颈

对齐有利于让访问落在更少事务中，但不能简单把任意指针强转成 `float4*`：地址需满足类型对齐、长度需处理尾部，输入输出布局和结果不能改变。本章保持标量 float 读写，把**向量化读取**作为单独的后续实验，不把它冒充已验证优化。

缓存可能让反复运行同一矩阵的 kernel 比一次性处理更快；本章 Event 中位数是设备常驻数据的重复执行时间，不含 H2D/D2H 和分配。共享内存分块也消耗每块资源，可能改变并行驻留；更大的 tile、更多寄存器暂存或循环展开未必更快，严重时还会出现寄存器溢出到 local memory。应结合 Nsight Compute 的资源和访存指标，再与第 12 章相同口径的时间核对。

## 4. 测量和验证

每个形状、每个版本先预热 3 次，再用 CUDA Event 测 20 次 kernel，打印中位时间。`effective_GBps` 仅按逻辑上读一次、写一次的 `2*W*H*sizeof(float)` 计算，不等于物理 DRAM 带宽。小矩阵的 Event 值可能受计时分辨率和启动成本支配，不能仅凭 `16×16` 的微小差异排序算法。

在仓库根目录构建并运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target transpose_compare -j2
./build/outline/book_ch14/transpose_compare
ctest --test-dir build/outline -R '^ch14_transpose_compare$' --output-on-failure
```

独立构建源目录为 `chapters/ch14-memory-optimization/examples`。对性能作结论前，应重复整组实验并检查 GPU 上是否有其他任务、时钟/功耗状态及 Nsight 指标。[验证记录](results/validation.md)保存一次实测原始值和并发负载；不能把这组值当作稳定性能排名。

## 5. 常见错误

- 写输出时仍用 `y*W+x`，结果只是复制或乱序，并非矩阵转置。
- 对尾块越界线程提前 `return`，使同一 Block 只有部分线程到达 `__syncthreads()`。
- 把 `tile[32][33]` 中的 33 错写进全局矩阵索引；填充只影响共享内存行跨度。
- 以单次执行或某一尺寸上的波动宣称填充版必然更快。
- 向量化读取不检查对齐和长度尾部，或让不同版本处理的元素数量不同。
- 将重复读同一数据的 Event 时间直接视为端到端图像处理时间。

## 6. 练习与参考答案

1. 宽 3、高 2 的输入 `[1,2,3,4,5,6]` 转置后的线性数组是什么？答：`[1,4,2,5,3,6]`。
2. `35×19` 使用 `32×32` tile，需要多少 Block？答：x 方向 `ceil(35/32)=2`，y 方向 `ceil(19/32)=1`，共 2 个。
3. 为什么尾块中越界线程不能在同步前直接退出？答：同一个 Block 的屏障要求线程按约定参与；可让越界线程跳过读写但继续到达屏障。
4. 若把 tile 宽改为 16，是否仍可直接用固定 `32×33` 的 bank 分析？答：不能；Warp 内线程坐标、行跨度和访问分组都会变，应重新分析并实测。
5. 若填充版共享内存 bank 冲突减少但总时间不变，下一步看什么？答：检查全局访存、启动成本、占用与资源使用、缓存和测量波动，判断原瓶颈是否在 bank 冲突。

[下一章：执行效率与资源取舍](../ch15-execution-resources/README.md)分析 Block 大小、资源和 Occupancy 的取舍。
