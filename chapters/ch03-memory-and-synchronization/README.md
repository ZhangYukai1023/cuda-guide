# 第 3 章：内存访问与块内同步

[全书导航](../../README.md) · [上一章](../ch02-thread-indexing/README.md) · [下一章](../ch04-engineering-and-timing/README.md)

## 1. 从一张小表的转置开始

输入矩阵有 2 行、3 列，转置后变成 3 行、2 列：

```text
输入：0 1 2      输出：0 3
      3 4 5            1 4
                       2 5
```

输入按行展开为 [0,1,2,3,4,5]；输出为 [0,3,1,4,2,5]。最小实现让线程 (x,y) 读取输入的 a[y*w+x]，写到输出 b[x*h+y]。输出宽度变成了原来的高度 h，这里不能继续用 w 做输出行跨度。

输入和输出必须分开。本例不是原地转置算法；让读写共享同一数组，会使某些线程读到已被覆盖的数据。

## 2. 最小示例：直接从全局内存读写

```cpp
if (x < w && y < h)
    b[x*h+y] = a[y*w+x];
```

它的优点是容易证明正确。相邻 x 线程读取连续位置，但写入位置相隔 h 个元素。访问模式会影响访存效率，不过本章只做正确性测试，不宣称某个版本更快。

需要先区分三种存储：

| 存储 | 本章例子 | 谁能使用 |
| --- | --- | --- |
| 全局内存 | a、b 数组 | 不同块都可寻址，正确性仍需管理依赖 |
| 共享内存 | tile 小矩阵 | 同一个线程块内的线程 |
| 线程局部值 | x、y、临时值 | 当前线程；不能假定一定驻留寄存器 |

不要把 C++ 局部数组自动理解为“共享”。只有显式的共享存储才供块内线程协作。寄存器压力、局部内存溢出等属于进一步分析的内容。

## 3. 递进示例：先读小块，再交换访问方向

每块使用 16×16 线程和一块共享数组：

```cpp
__shared__ int tile[16][17];
if (x<w && y<h) tile[threadIdx.y][threadIdx.x]=a[y*w+x];
__syncthreads();
int ox=blockIdx.y*16+threadIdx.x;
int oy=blockIdx.x*16+threadIdx.y;
if (ox<h && oy<w) b[oy*h+ox]=tile[threadIdx.x][threadIdx.y];
```

先以原方向读取，等块中所有线程完成写入后，再交换下标读取共享内存并写出。这里有两个交换：线程在 tile 内的坐标交换，线程块在整个矩阵中的坐标也交换。

为什么共享数组有 17 列？多出的列用于改变相邻行的地址间隔，是降低某些列访问 bank 冲突的常见技巧。本例 block 为 16×16，不能直接套用其他 block 配置的 bank 冲突结论；这里只演示 padding 写法，没有用 profiler 证明冲突数量，也没有测量加速比。

为什么边缘没有初始化整块 tile 也能正确？一个满足输出条件的线程读取 tile[tx][ty]，对应的生产线程原本加载输入坐标 (blockIdx.x*16+ty, blockIdx.y*16+tx)。输出条件恰好保证该坐标有效。因此有效输出不会读取未填充的位置。

## 4. 屏障的位置决定正确性

__syncthreads 是块内的会合点。本例把它放在边界判断之后、独立的一行，所有线程都会到达。错误写法是把它放进 if(x<w && y<h)，让部分线程参加而其他线程跳过。

屏障不能同步两个普通线程块，也不能替代 CPU 的 cudaDeviceSynchronize。前者协调同块线程读写共享数组；后者等待设备工作，供主机检查结果。官方语义可查阅 [线程层次与同步说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#thread-hierarchy)。

源码对 3×2、1×1、31×17、64×48 都运行直接版和共享内存版。31×17 同时覆盖两个方向的不完整 tile；所有输出与 CPU 转置结果逐项完全一致。

## 5. 常见错误

- 在访问 tile 之前漏掉屏障，出现线程间读写竞争。
- 在屏障之前让边界线程 return，使块内控制流无法共同到达同步点。
- 输出仍按原宽度寻址；长方形输入比正方形更容易暴露问题。
- 只交换 tile 内坐标，忘记交换块坐标。
- 因为测试通过就断言不存在竞争。本机没有 Compute Sanitizer，racecheck、synccheck 尚未运行。

## 6. 练习与参考答案

1. 4×2 输入 [0,1,2,3,4,5,6,7] 转置后的扁平数组是什么？
   答：[0,4,1,5,2,6,3,7]，输出宽度为 2。
2. 删除边界 if，测试 31×17 会发生什么？
   答：存在越界读写风险，错误不一定稳定表现为崩溃。不要把“没崩溃”当作正确。
3. 能把 __syncthreads 换成 cudaDeviceSynchronize 吗？
   答：不能。这段代码需要设备函数中的块内屏障；本书的主机同步调用不承担这一职责。
4. 共享版一定比直接版快吗？
   答：不能据本章判断。需要在同一输入、相同预热与重复条件下分别计时，尤其小矩阵容易由启动成本主导。

## 构建、运行与实测输出

完整源码：[transpose.cu](examples/transpose.cu)；构建定义：[CMakeLists.txt](examples/CMakeLists.txt)。公共依赖也在本仓库，不需要复制未提供的代码。

在项目根目录执行（首次配置后可重复构建）：

```bash
cd /data2/cuda-guide
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all --target transpose -j2
./build/all/ch03/transpose
```

也可单独配置本章：把上面 -S 改为 chapters/ch03-memory-and-synchronization/examples，-B 改为 build/ch03-standalone，其他选项不变；程序位于该独立构建目录下。无需第三方图像或数学库。

本次实际输出如下。除计时数据可能随运行波动外，同一固定输入应得到这些结果或在注明容差内一致；最终错误数量应为 0：

```text
naive_transpose: n=6 max_abs_error=0 mismatches=0 PASS
tiled_transpose: n=6 max_abs_error=0 mismatches=0 PASS
0 3 1 4 2 5
naive_transpose: n=1 max_abs_error=0 mismatches=0 PASS
tiled_transpose: n=1 max_abs_error=0 mismatches=0 PASS
naive_transpose: n=527 max_abs_error=0 mismatches=0 PASS
tiled_transpose: n=527 max_abs_error=0 mismatches=0 PASS
naive_transpose: n=3072 max_abs_error=0 mismatches=0 PASS
tiled_transpose: n=3072 max_abs_error=0 mismatches=0 PASS
```

更多环境与限制见 [validation.md](results/validation.md)。本机没有 Compute Sanitizer，不能把 CPU 对照通过等同于内存或同步工具检查通过。
