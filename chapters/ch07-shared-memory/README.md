# 第 7 章：共享内存与同步

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch06-memory-layout/README.md)

第 6 章的线程主要独立读取数据。本章让一个 Block 的线程交换数据：先反转 8 个整数，再转置二维小矩阵，最后对每块的输入求和。重点是判断谁写入、谁读取、哪些线程必须到达同步点。完整源码见 [shared_memory.cu](examples/shared_memory.cu)。本章已在 zyk 用 nvcc 编译，并在 RTX 5060 Ti 上运行；下面的 CPU 对照与 CTest 结果见[验证记录](results/validation.md)。

## 1. 最小示例：8 个线程互换数据

输入 `[0,1,2,3,4,5,6,7]`，要求输出 `[7,6,5,4,3,2,1,0]`。每个线程先把自己的输入写入块内共享数组，然后读取另一线程写入的位置：

```cpp
__shared__ int tile[8];
int t = threadIdx.x;
tile[t] = input[t];
__syncthreads();
output[t] = tile[7 - t];
```

线程 0 想读 `tile[7]`，而它由线程 7 写入。若缺少屏障，线程 0 可能在写入发生前读取。`__syncthreads()` 让同一个 Block 中到达这里的线程等待所有应参加的线程，并协调屏障前后的共享数据访问。本例启动恰好 1 个 Block、8 个线程，所有线程都执行屏障。

共享内存属于 Block。另一个 Block 不能用这段普通 `__shared__` 数组来交换数据，`__syncthreads()` 也不保证其他 Block 已完成。GPU kernel 与 Host 的 `cudaDeviceSynchronize()` 是不同范围的同步：前者是设备内块内会合点，后者等待设备工作以便 Host 检查结果。

## 2. Warp 内同步与参与线程

第二个例子启动 1 个完整 Warp（本机 Warp 大小为 32），同样用共享数组反转 0 到 31。这次使用 `__syncwarp()`：

```cpp
tile[threadIdx.x] = input[threadIdx.x];
__syncwarp();
output[threadIdx.x] = tile[31 - threadIdx.x];
```

默认掩码要求这个 Warp 的 32 个线程都参与。本例正好启动 32 个线程，且没有提前返回，所以参与范围明确。`__syncwarp()` 只协调所指定的 Warp 线程，不能代替跨 Warp 的 `__syncthreads()`。不要仅凭“同一 Warp 看起来同步执行”就删掉必要的同步；现代 CUDA 硬件支持独立线程调度。更复杂的部分参与掩码和 Warp 数据交换留在进阶章节。

## 3. 递进：16×16 分块转置

输入为 2 行、3 列：

```text
输入：0 1 2      转置：0 100
      100 101 102         1 101
                          2 102
```

输入行优先偏移为 `y*width+x`；转置输出有 `width` 行、每行 `height` 列，偏移为 `x*height+y`。每个 16×16 的 Block 把输入块加载到 `tile[16][17]`，在 `__syncthreads()` 后交换读取的线程坐标和 Block 坐标，写到输出。第 17 列是常见的行跨度填充写法；本章不测 bank 冲突或加速效果。

```cpp
if (x < width && y < height)
    tile[threadIdx.y][threadIdx.x] = input[y * width + x];
__syncthreads();
if (output_x < height && output_y < width)
    output[output_y * height + output_x] = tile[threadIdx.x][threadIdx.y];
```

屏障在输入边界条件之外。对于 31×17 这类不完整 Tile，一部分线程没有有效输入，但仍须到达屏障。有效输出读取的共享元素一定对应一个有效输入位置：输出边界条件同时保证相应生产线程的输入坐标有效。源码对 3×2、1×1、31×17 分别与 CPU 转置逐项比较。

## 4. 块内求和：先填零再同步

每个 Block 固定 128 个线程，读取最多 128 个整数。最后一块若不足 128 项，多余线程先把自己的共享位置写成 0，再和其他线程一起参加屏障。之后按 64、32、16、8、4、2、1 的步长做树形求和：

```cpp
values[t] = i < n ? input[i] : 0;
__syncthreads();
for (int offset = 64; offset > 0; offset /= 2) {
    if (t < offset) values[t] += values[t + offset];
    __syncthreads();
}
if (t == 0) partial[blockIdx.x] = values[0];
```

这里的 `if (t < offset)` 只控制加法，屏障仍由 Block 内所有线程执行。线程不能因 `i>=n` 就在屏障前 `return`。长度 1、7、128、1003 分别会得到 1、1、1、8 个块的部分和；Host 为每块计算参考值并逐项比较。本章只得到“每块部分和”，还没有把不同 Block 的结果合并成全局总和。跨 Block 的分阶段归约将在第 8 章继续。

## 5. 构建与验证

在服务器项目根目录执行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target shared_memory -j2
./build/outline/book_ch07/shared_memory
ctest --test-dir build/outline -R '^ch07_shared_memory$' --output-on-failure
```

也可用 `-S chapters/ch07-shared-memory/examples -B build/ch07-shared-standalone` 独立配置。独立构建和根工程 CTest 均已运行，CPU 对照逐项通过。Compute Sanitizer 当前不可用，因此同步错误故障模式的工具检查仍未测。实际验证状态见 [验证记录](results/validation.md)。

## 6. 常见故障与定位

- 删除 `reverse_block` 的屏障：读取别的线程写入的共享位置可能形成数据竞争。结果偶尔正确也不能证明安全。
- 在 `transpose_tiled` 的 `if(x<width && y<height)` 中调用屏障，或让边界线程提前返回：一个 Block 内的线程可能不一致地到达屏障，行为不符合要求。
- 用 `__syncwarp()` 试图协调两个不同 Warp，或用 `__syncthreads()` 试图协调不同 Block：同步作用域不匹配。
- 分块转置只交换共享数组下标，不交换 Block 坐标；大于一个 Tile 的矩阵会写错位置。
- 块内求和未把越界线程的共享位置设为 0，导致最后一个 Block 读到未定义的旧值。

若服务器上可用 Compute Sanitizer，可对程序运行 memcheck、racecheck 和 synccheck；当前环境此前未找到该工具，不能写成“工具检查通过”。工具类别与适用范围见 [Compute Sanitizer 官方文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

## 7. 练习与参考答案

1. 输入 `[2,4,6,8,10,12,14,16]`，8 线程反转后的结果？答：`[16,14,12,10,8,6,4,2]`。
2. 3×2 输入按行展开为 `[0,1,2,100,101,102]`，转置后按行展开是什么？答：`[0,100,1,101,2,102]`。
3. 长度 130、每块 128 项的块内求和，需要几个部分和？答：2 个；第二块只有前 2 个线程装载输入，其余 126 个线程必须写 0 并参加每次屏障。
4. 线程 0 写 `tile[0]`，线程 7 读 `tile[0]`，只在读线程调用 `__syncthreads()` 可以吗？答：不可以，同一 Block 的参与线程必须按要求一致地到达该屏障。
5. 本章的 8 个块部分和能用一个普通 `__syncthreads()` 直接合并吗？答：不能；它只协调一个 Block。可再启动一个 kernel 或使用其他明确支持的跨 Block 方案。

下一章从这些部分和出发，学习归约、原子操作与直方图等基础并行模式。
