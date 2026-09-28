# 第 3 章：从线程编号到数组计算

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch02-first-thread/README.md)

第 2 章只有一个线程块。本章让多个 Block 处理一维数组，再让二维线程块处理一张小图。目标是能为任意元素算出负责它的线程，并在尺寸不能整除 Block 大小时仍写对结果。完整源码见 [array_indexing.cu](examples/array_indexing.cu)。

## 1. 手算一个七元素数组

输入 `a=[-6,-5,-4,-3,-2,-1,0]`，每项乘 3 后应得到 `[-18,-15,-12,-9,-6,-3,0]`。每个位置只依赖自身，适合让一个线程负责一个元素：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) output[i] = input[i] * factor;
```

`threadIdx.x` 是块内编号，`blockIdx.x` 是块编号，`blockDim.x` 是每块线程数。因此每块 128 线程时，块 0 负责起始下标 0，块 1 负责起始下标 128。`i` 是全局下标，而不是只看 `threadIdx.x`。一个长度 7 的数组仍可启动 128 个线程，只有 `i=0..6` 的线程访问内存；其余线程因 `i<n` 为假而不读写。

Host 使用 `(n+127)/128` 个块，向上取整覆盖末尾。例如长度 256 恰好需要 2 块；长度 1003 需要 8 块，启动 1024 个线程，最后 21 个线程越过末尾。`if(i<n)` 是防止越界的必要条件，不能因为只多出少数线程就省去。长度 0 的任务需要单独约定，不能启动 0 个块；本章按大纲测试长度 1、7、256、1003，空输入处理在资源管理章节展开。

## 2. 同一索引公式做向量加法

设 `a=[-6,-5,-4]`、`b=[1,3,5]`，CPU 手算 `a+b=[-5,-2,1]`。GPU 可以让每个线程只处理一个下标，也可以采用 Grid-stride loop：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (; i < n; i += stride) output[i] = a[i] + b[i];
```

本例固定启动 2 个 Block，每块 128 个线程，所以总步长为 256。线程 0 处理 `0,256,512,768`，再尝试 1024 时停止，适用于长度 1003。线程 1 处理 `1,257,513,769`。每个下标只由一个线程负责；把步长误写成 `blockDim.x` 会让不同 Block 访问相同位置。

两个 kernel 都用 CPU 整数计算的参考数组逐项核对，容差为 0。本章的 [公共辅助头文件](../../common/cuda_support.cuh) 完整提供设备内存申请、上传、下载、错误检查与比较；第 2 章已用原始 CUDA API 展示过这些步骤。这里复用辅助类，把注意力放在下标公式。

## 3. 二维索引与图像行跨度

宽 3、高 2 的图像可以按行存成六个整数。如果像素值由 `100*y+x` 定义，坐标与线性内存对应如下：

```text
(x,y)  (0,0) (1,0) (2,0)   (0,1) (1,1) (2,1)
value      0     1     2     100   101   102
index      0     1     2       3     4     5
```

行优先内存偏移是 `y*width+x`。宽度表示一行有多少元素，不能写成 `y*height+x`；方形图像有时会掩盖这一错误。CUDA 线程也可使用二维 Block 和 Grid：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) output[y * width + x] = 100 * y + x;
```

本例 Block 为 16×8，Grid 的两个方向分别向上取整。除 3×2 和 1×1 外，还实测了 37×19 与 19×37：宽、高互换后总像素数相同，但行跨度和边界 Block 不同，两种都与 CPU 参考完全一致。多个 Block 没有必须按编号依次完成的保证；让每个线程写到确定的位置，CPU 再按顺序读取结果即可。

## 4. 构建与实测输出

在服务器项目根目录执行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target array_indexing -j2
./build/outline/book_ch03/array_indexing
ctest --test-dir build/outline -R '^ch03_array_indexing$' --output-on-failure
```

也可将 `-S` 改为 `chapters/ch03-array-indexing/examples`，将 `-B` 改为 `build/ch03-standalone` 独立配置。完整源码和 [CMake 构建文件](examples/CMakeLists.txt) 都在本章目录。本机选择 `120` 是因实际 GPU 计算能力为 12.0，其他机器必须重新核对。

本次程序实测输出如下；整数结果均与 CPU 逐项完全相同：

```text
multiply: n=1 max_abs_error=0 mismatches=0 PASS
vector_add: n=1 max_abs_error=0 mismatches=0 PASS
multiply: n=7 max_abs_error=0 mismatches=0 PASS
vector_add: n=7 max_abs_error=0 mismatches=0 PASS
multiply: n=256 max_abs_error=0 mismatches=0 PASS
vector_add: n=256 max_abs_error=0 mismatches=0 PASS
multiply: n=1003 max_abs_error=0 mismatches=0 PASS
vector_add: n=1003 max_abs_error=0 mismatches=0 PASS
coordinates: n=6 max_abs_error=0 mismatches=0 PASS
coordinates: n=1 max_abs_error=0 mismatches=0 PASS
coordinates: n=703 max_abs_error=0 mismatches=0 PASS
coordinates: n=703 max_abs_error=0 mismatches=0 PASS
```

更多构建与限制见 [验证记录](results/validation.md)。本章只验证正确性，没有性能比较，也没有在缺失的 Compute Sanitizer 下声称内存检查通过。

## 5. 常见错误与定位

- 只把 `threadIdx.x` 当数组下标：多个 Block 会重复写前 128 项，后续位置无人处理。
- 向上取整启动后不写 `i<n`：尾块可能越界。用长度 1003 比只用 256 更容易暴露问题。
- Grid-stride loop 的步长只取 `blockDim.x`：不同 Block 工作重叠；应是整个 Grid 的线程数。
- 二维内存偏移使用高度代替宽度：用 37×19 与 19×37 分别检验。
- 以 `printf` 顺序推断线程执行顺序：调度与输出顺序不构成正确性依据，应比较最终数组。

## 6. 练习与参考答案

1. `<<<3,4>>>` 中块 2 的线程 1 对应全局一维下标是多少？答：`2*4+1=9`。
2. 长度 7、每块 4 个线程需要几个 Block？最后一块有几个有效线程？答：2 个 Block；最后一块有效下标为 4、5、6，共 3 个线程。
3. 总线程数为 6 时，Grid-stride loop 的线程 2 处理长度 17 数组的哪些下标？答：2、8、14。
4. 宽 7、高 5 的图像，坐标 `(x=2,y=3)` 的行优先偏移是多少？答：`3*7+2=23`。
5. 宽 37、高 19，用 16×8 的 Block 需要几个二维 Block？答：x 方向向上取整为 3，y 方向也为 3，即 3×3 个 Block；每个边界线程仍要检查两个坐标。

本章完成了从单线程到一维、二维数组的工作分配。下一章将进一步管理输入、输出和中间缓冲区，并区分 GPU 工作与 CPU/GPU 数据传输的成本。
