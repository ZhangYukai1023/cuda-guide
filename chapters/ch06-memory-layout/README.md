# 第 6 章：内存层次与数据布局

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch05-warp-execution/README.md)

同一项计算若让相邻线程读取不同位置，结果仍可能完全正确，但访问模式会改变。先用小数组手算连续与跨步读取，再处理带行跨度的二维数据，最后比较粒子的 AoS/SoA 布局及 Unified Memory。完整源码见 [memory_layout.cu](examples/memory_layout.cu)。本章验证地址与结果，不做性能排名。

## 1. 哪些内存由谁使用

| 名称 | 初学时的理解 | 本章使用情况 |
| --- | --- | --- |
| Register | 编译器常用来保存线程自己的标量临时值；C++ 局部变量不保证一定留在寄存器 | `i`、`x`、`y` 等是线程私有值，实际放置由编译器决定 |
| Local memory | 线程私有的可寻址空间，名字叫 Local，物理上通常走设备内存层次 | 未刻意使用；寄存器溢出等情况留到性能分析 |
| Shared memory | 同一个 Block 的线程可共同访问，生命周期与 Block 相关 | 第 7 章实测；本章先辨认作用域 |
| Global memory | Grid 内线程可访问的设备存储，显式 `cudaMalloc` 数据在此 | 所有数组示例 |
| Constant memory | 设备代码只读的常量区域，适合特定访问模式 | 只介绍用途，未编写常量内存 kernel |
| Cache | 硬件可能缓存某些读取，具体效果依访问模式和架构而变 | 本章未采集缓存指标 |

“Local”并不意味着一定在快速的片上存储，“Global”也不意味着每次访问都绕过缓存。存储空间的作用域与实际访问开销要分开理解。概念与限制依据 [CUDA 12.8 编程指南：内存层次](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#memory-hierarchy) 及 [最佳实践：设备内存空间](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/#device-memory-spaces)。

## 2. 最小示例：连续读取与跨步读取

输入为 `0,1,2,...`，7 个线程每人输出读到值的两倍。连续版读取 `input[i]`，前 7 项为 `[0,2,4,6,8,10,12]`。跨步版读取 `input[4*i]`，前 7 项为 `[0,8,16,24,32,40,48]`。

```cpp
// 相邻线程 i、i+1 读取相邻元素
output[i] = 2 * input[i];
// 相邻线程读取相隔 4 个 int 的元素
output[i] = 2 * input[i * 4];
```

每个 `int` 在本例占 4 字节，因此两个相邻线程的源地址分别相差 4 与 16 字节。源码为跨步版申请 `n*4` 个元素，避免最后一个线程越界；两版分别与 CPU 参考数组逐项比较。长度 7 和 1003 都实际通过。相邻地址更容易形成合并访问，但缓存、事务数与最终速度需要后续用分析工具和可信基准测量，不能从本章 PASS 输出推断速度。

## 3. 二维行跨度与填充

宽 5、高 3 的小图若每行实际占 8 个整数，后 3 个位置可作填充。像素 `(x=3,y=2)` 在带填充输入中的偏移是 `2*8+3=19`，复制到紧凑输出后偏移是 `2*5+3=13`。核函数明确使用两个不同的行跨度：

```cpp
compact[y * width + x] = padded[y * row_stride + x];
```

源码把填充值设为 -1，有效像素设为 `100*y+x`，再将 5×3 的有效区域复制为连续输出，与 CPU 参考全部 15 项一致。真实图像接口可能把 stride 以字节而非元素为单位给出；调用时必须先确认单位，不能直接把“宽度”当作实际行跨度。

## 4. AoS 与 SoA：同一粒子的两种排列

两个粒子分别为 `(x=0,y=1)`、`(x=1,y=4)`，计算 `x+y` 应得到 `[1,5]`。AoS（Array of Structures）把同一粒子的字段放在一起；SoA（Structure of Arrays）把同一字段排成连续数组：

```text
AoS: [x0,y0 | x1,y1 | x2,y2 | ...]
SoA: x=[x0,x1,x2,...]，y=[y0,y1,y2,...]
```

完整示例令 `x=i`、`y=3*i+1`，每项参考结果为 `4*i+1`。AoS 核函数读 `input[i].x` 和 `input[i].y`；SoA 核函数读 `x[i]` 和 `y[i]`。长度 7 和 1003 都与 CPU 参考完全一致。AoS 便于把单个粒子的字段一起管理，SoA 让相邻线程读取同一字段时地址更连续；具体工作负载和缓存行为决定性能，本章只确认布局和索引正确。

## 5. Unified Memory 与显式传输

最后一组输入为 `[-3,-2,-1,0,1,2,3]`，每项加 1 后应为 `[-2,-1,0,1,2,3,4]`。显式版用 `cudaMalloc`、H2D 上传、kernel、同步、D2H 下载；Managed 版用 `cudaMallocManaged` 得到 CPU 与 GPU 都能寻址的数组，由 CPU 填值，kernel 修改，同步完成后 CPU 读取。两版都与同一 CPU 参考完全一致。

Managed 指针可由 Host 和 Device 使用，但这不表示访问没有数据移动成本。运行时可能在 CPU/GPU 间迁移或管理页面；本章没有测量迁移量或速度。Host 读取 kernel 修改后的 Managed 数据之前要先完成合适的同步。本章用 `cudaDeviceSynchronize`。Managed 内存仍须 `cudaFree`；源码在正常路径和异常路径都执行释放。

## 6. 构建、运行与实测输出

从服务器项目根目录运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target memory_layout -j2
./build/outline/book_ch06/memory_layout
ctest --test-dir build/outline -R '^ch06_memory_layout$' --output-on-failure
```

也可用 `-S chapters/ch06-memory-layout/examples -B build/ch06-layout-standalone` 独立配置。实际输出：

```text
contiguous: n=7 max_abs_error=0 mismatches=0 PASS
strided: n=7 max_abs_error=0 mismatches=0 PASS
contiguous: n=1003 max_abs_error=0 mismatches=0 PASS
strided: n=1003 max_abs_error=0 mismatches=0 PASS
row_stride: n=15 max_abs_error=0 mismatches=0 PASS
aos: n=7 max_abs_error=0 mismatches=0 PASS
soa: n=7 max_abs_error=0 mismatches=0 PASS
aos: n=1003 max_abs_error=0 mismatches=0 PASS
soa: n=1003 max_abs_error=0 mismatches=0 PASS
explicit_transfer: n=7 max_abs_error=0 mismatches=0 PASS
managed_memory: n=7 max_abs_error=0 mismatches=0 PASS
```

记录与工具限制见 [验证结果](results/validation.md)。所有 PASS 都是正确性验证，没有测量带宽、缓存命中或加速比。

## 7. 常见错误

- 为跨步读取只分配 `n` 个输入元素，却访问 `input[4*i]`；长度 7 时最后会试图读取第 24 项。
- 把图像有效宽度与实际行跨度混用，或者混淆字节 stride 和元素 stride。
- 将 AoS 的 `Particle` 结构体数组当作 SoA 两个连续 `int` 数组传给 kernel。两种布局要与对应索引公式配套。
- 认为 Managed 内存无需同步，kernel 发起后立即在 Host 读取。同步前结果不能作为已完成计算使用。
- 把变量名为“local”或“shared”误当作存储空间保证。CUDA 存储类别由声明、作用域与编译器共同决定。

## 8. 练习与参考答案

1. 连续输入 `[0,1,2,...]`，步长为 3，线程 0、1、2 各读哪个下标？答：0、3、6；若计算两倍，输出为 0、6、12。
2. 宽 5、行跨度 8，像素 `(x=4,y=1)` 的输入与紧凑输出偏移分别是多少？答：12 与 9。
3. 三个粒子 `(0,1),(1,4),(2,7)` 的 AoS 与 SoA 排列如何写？答：AoS 为 `[(0,1),(1,4),(2,7)]`；SoA 为 `x=[0,1,2]`、`y=[1,4,7]`；两者 `x+y` 均为 `[1,5,9]`。
4. Managed 版省去了显式 `cudaMemcpy`，是否能据此说它一定更快？答：不能；数据迁移与页面管理仍有成本，必须测量同一任务的完整范围。
5. 为什么本章不能宣称跨步版一定慢 4 倍？答：步长 4 只描述逻辑地址间隔，实际性能还受缓存、事务、任务规模、编译与硬件影响，本章没有计时。

下一章通过共享内存交换线程数据，学习块内同步的必要性和范围。
