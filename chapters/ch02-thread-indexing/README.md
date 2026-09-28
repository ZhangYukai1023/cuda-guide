# 第 2 章：线程、下标与二维数据

[全书导航](../../README.md) · [上一章](../ch01-getting-started/README.md) · [下一章](../ch03-memory-and-synchronization/README.md)

## 1. 从一个 GPU 线程开始

先看三个可以手算的任务：一个线程写入 42；一个线程把 7 和 5 相加得到 12；然后让同一线程块的 8 个线程各写自己的编号，结果是 `[0,1,2,3,4,5,6,7]`。前两步只有一个 GPU 线程，不需要先理解 Grid-stride loop。

完整源码在 [single_thread.cu](examples/single_thread.cu)。`__global__` 声明由 CPU 发起、在 GPU 上执行的函数。`<<<1, 1>>>` 表示启动一个 Block，其中只有一个线程；函数参数里的设备指针指向 GPU 内存。CPU 不能把普通主机指针直接传给本例 kernel 并期待它可用。

```cpp
__global__ void write_42(int* out) { out[0] = 42; }
// 主机代码：
DeviceBuffer<int> one(1);              // 内部调用 cudaMalloc
write_42<<<1, 1>>>(one.data);          // 提交 kernel
CUDA_CHECK(cudaGetLastError());        // 检查提交错误
CUDA_CHECK(cudaDeviceSynchronize());   // 等待执行并检查异步错误
verify("write_42", one.download(), std::vector<int>{42}); // 内部调用 cudaMemcpy
// 离开作用域时 DeviceBuffer 调用 cudaFree
```

这些辅助函数的完整实现放在 [cuda_support.cuh](../../common/cuda_support.cuh)。按顺序可看到 `cudaMalloc` 申请设备空间、`cudaMemcpy` 取回结果、`cudaFree` 释放资源。`cudaGetLastError` 只能检查启动时已经报告的错误；kernel 执行中的错误还需要在同步点检查。`verify` 与 CPU 手算值逐项比较，出错时程序返回非零。

递进的 `add_scalars<<<1,1>>>(7,5,one.data)` 仍只有一个线程；`write_ids<<<1,8>>>` 才开始让多个线程各写一个位置。这里的 `threadIdx.x` 是块内编号 0 到 7，只有一个 Block 时也等于本例的全局编号。后文再处理多个 Block。

在项目根目录执行：

```bash
cmake --build build/all --target single_thread -j2
./build/all/ch02/single_thread
```

实际输出：

```text
write_42: n=1 max_abs_error=0 mismatches=0 PASS
add_scalars: n=1 max_abs_error=0 mismatches=0 PASS
write_ids: n=8 max_abs_error=0 mismatches=0 PASS
```

练习：把 `add_scalars` 的输入改成 `-3` 和 `9`，先预测输出再编译运行。参考答案为 `6`；同时把 CPU 参考值改为 `6`，否则正确性检查会按旧预期报错。

## 2. 先给每个元素找到负责它的线程

第 1 章用 128 个线程处理 8 个数，但线程不总是只处理一个元素。设有 10 个元素、只启动 4 个线程，可以这样分工：

| 线程的初始下标 | 它负责的元素下标 |
| --- | --- |
| 0 | 0、4、8 |
| 1 | 1、5、9 |
| 2 | 2、6 |
| 3 | 3、7 |

每次向前走的步长是总线程数。这种循环称为 grid-stride loop。它把数据规模与启动线程数分开，后续处理较长向量时也能复用同一段代码。

最小示例计算 out[i]=3i-7。例如 n=4 时结果为 [-7,-4,-1,2]。完整程序还测试 0、1、31、32、33、1003 个元素。

```cpp
for (int i = blockIdx.x*blockDim.x + threadIdx.x;
     i < n; i += blockDim.x*gridDim.x)
    out[i] = 3*i - 7;
```

本例实际固定启动 2×32=64 个线程。长度 1003 时，线程 0 负责 0、64、128……960，线程 43 最后负责 1003 吗？不会，因为下标等于长度时已经越界，循环条件是小于 n。

n=0 在主机端直接跳过启动。CUDA 不需要为一个空任务启动零个块；清楚处理空输入比依赖特殊启动参数更可靠。源码使用 int 下标，因为测试数据很小。处理超过 int 范围的数据时，应同时检查索引、乘积、字节数和网格尺寸的类型。

## 3. 从一维数组走到二维图像

设图像宽 3、高 2：

```text
坐标值 100*y+x：
  0    1    2
100  101  102
内存：[0,1,2,100,101,102]
```

二维坐标与内存偏移不是同一个东西。连续行优先数组的偏移是 y*width+x，不是 x*height+y。网格负责覆盖二维空间：

```cpp
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;
if (x < width && y < height)
    out[y*width+x] = 100*y+x;
```

递进示例每块 16×8 个线程，对 37×19 图像启动 3×3 个块。边缘块会有多余线程，x 与 y 两个方向都要检查。程序还测试 1×1 图像，避免仅在“大而规则”的数据上通过。

二维 block 的 x 方向在线性线程编号中变化最快，因此把 x 对应连续像素也是后续分析内存访问的自然起点。线程块的大小不等于图像大小，块的实际执行顺序也不应成为算法的正确性前提。

## 4. 如何阅读本章源码

本章开始把重复代码放入 [cuda_support.cuh](../../common/cuda_support.cuh)。它也是完整示例的一部分，不能只复制单个 .cu 到别处编译。DeviceBuffer 申请设备数组，upload/download 传输数据；CUDA_CHECK 报告调用位置；verify 对照 CPU 参考并在不一致时使程序失败。其 C++ 资源管理细节在第 4 章讲解，初读只需把它们当作有明确职责的小工具。

核函数与 CPU 参考实现分别生成结果，逐项比较整数，要求误差为 0。长度 0 的 PASS 仅代表空输入处理正确，不表示 GPU 执行了计算。二维小例实际打印 0 1 2 100 101 102。

## 5. 常见错误

- 把 blockIdx 当作全局元素下标，导致不同线程重复写同一位置。
- 把二维偏移写成 y*height+x；方形图像可能掩盖这个错误，所以要测试 37×19。
- 在 CPU 上按 printf 的先后顺序推断 GPU 调度。需要确定性结果时，让线程写入指定位置，再由 CPU 有序打印。
- 向上取整计算块数后忘记边界检查。启动参数覆盖数据，不意味着每个线程都有有效元素。
- 认为某个线程块一定先完成，再让其他块读取它的中间结果。普通启动没有这种保证。

## 6. 练习与参考答案

1. 5 个线程用步长循环处理 12 项，线程 1 处理哪些下标？
   答：1、6、11；线程 2 处理 2、7，不能访问 12。
2. 7×5 图像用 4×4 的块，需要怎样的 grid？
   答：2×2 个块，共 64 个线程，只有 35 个对应有效像素。
3. 像素 (x=2,y=3)，图像宽 7，偏移是多少？
   答：3×7+2=23。宽度是行跨度；高度只决定行数。
4. 将二维公式改成 y*width+x，CPU 参考也同步修改，然后重编译。对 3×2 图像应输出什么？
   答：[0,1,2,3,4,5]；这是练习预期，不属于当前日志的原始结果。

完成本章后，你应能手算任意一个线程的工作范围。下一章在这个基础上讨论线程如何协作交换数据。

## 构建、运行与实测输出

下标与二维示例源码：[indexing.cu](examples/indexing.cu)；单线程示例：[single_thread.cu](examples/single_thread.cu)；构建定义：[CMakeLists.txt](examples/CMakeLists.txt)。公共依赖也在本仓库，不需要复制未提供的代码。

在项目根目录执行（首次配置后可重复构建）：

```bash
cd /data2/cuda-guide
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all --target indexing -j2
./build/all/ch02/indexing
```

也可单独配置本章：把上面 -S 改为 chapters/ch02-thread-indexing/examples，-B 改为 build/ch02-standalone，其他选项不变；程序位于该独立构建目录下。无需第三方图像或数学库。

本次实际输出如下。除计时数据可能随运行波动外，同一固定输入应得到这些结果或在注明容差内一致；最终错误数量应为 0：

```text
grid_stride: n=0 max_abs_error=0 mismatches=0 PASS
grid_stride: n=1 max_abs_error=0 mismatches=0 PASS
grid_stride: n=31 max_abs_error=0 mismatches=0 PASS
grid_stride: n=32 max_abs_error=0 mismatches=0 PASS
grid_stride: n=33 max_abs_error=0 mismatches=0 PASS
grid_stride: n=1003 max_abs_error=0 mismatches=0 PASS
image_coordinates: n=6 max_abs_error=0 mismatches=0 PASS
0 1 2 100 101 102
image_coordinates: n=703 max_abs_error=0 mismatches=0 PASS
image_coordinates: n=1 max_abs_error=0 mismatches=0 PASS
```

更多环境与限制见 [validation.md](results/validation.md)。本机没有 Compute Sanitizer，不能把 CPU 对照通过等同于内存或同步工具检查通过。
