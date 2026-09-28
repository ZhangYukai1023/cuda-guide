# 第 11 章：先认识库，再决定实现方式

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch10-debugging/README.md)

第 8 章手写了归约，第 9、10 章建立了验证与调试方法。现在先找可用的库实现作为正确性和性能基线。本章示例 [library_baselines.cu](examples/library_baselines.cu) 使用 Thrust 排序、CUB 整数组求和，并在有 cuBLAS 开发库时编译 2×2 矩阵乘法；已运行的路径各自与 CPU 结果比较。本章在 zyk 实测了 Thrust 与 CUB；当前 CUDA 安装没有 cuBLAS 开发库，`cublasSgemm` 路径跳过且**未编译、未运行**，不能把整章库对照记为完成。详见[验证记录](results/validation.md)。

## 1. 三层常用工具各做什么

- **Thrust** 提供接近 C++ 算法的接口；本章用 `device_vector` 与 `thrust::sort`。它适合先表达排序、扫描、变换等常见数据处理。
- **CUB** 提供面向线程、Warp、Block 与整个设备的并行原语；本章用设备级 `DeviceReduce::Sum`。调用者须理解输入输出和临时工作区。
- **cuBLAS** 提供线性代数例程；本章用 `cublasSgemm`。矩阵尺寸、布局、转置、leading dimension、数据类型和计算精度必须与接口约定一致。

图像处理篇还会遇到 [NPP](https://docs.nvidia.com/cuda/npp/index.html)，它提供图像与信号处理操作。NPP 的图像行步长、ROI 和边界约定需要单独核对；本章只介绍定位，不把它塞进前三个小示例。

## 2. 排序：先与 CPU 的同一输入对照

输入 `[7,1,7,-2,5,0,-2]`，`std::sort` 与 `thrust::sort` 应都输出 `[-2,-2,0,1,5,7,7]`。本章只比较值序列，未要求重复键的原始相对顺序；若业务要求稳定排序，应另选稳定算法，并把“键相同怎么办”写入测试。示例把 host vector 构造为 `device_vector`，排序后再复制回主机。这些分配与传输都是完整任务的一部分，不能只看排序调用就推断端到端收益。[Thrust 文档](https://nvidia.github.io/cccl/thrust/)可用于核对具体版本 API。

## 3. CUB 归约：工作区也是接口的一部分

对 `n=7` 和 `n=1003`，输入定义为 `i%17-8`，CPU 用 `std::accumulate` 得到精确整数参考值。CUB 调用分两步：

```cpp
std::size_t bytes = 0;
cub::DeviceReduce::Sum(nullptr, bytes, d_in, d_out, n); // 查询工作区
// 分配至少 bytes 个设备字节；本例 RAII 对象负责释放
cub::DeviceReduce::Sum(workspace, bytes, d_in, d_out, n); // 实际归约
```

查询调用不能当成已经算出总和；第二次调用才写输出。本章让 `d_in`、`d_out` 与工作区都活到设备同步以后，并比较 CPU 与 GPU 的整数结果。较大系统可复用工作区，但要保证容量足够、并发流之间不在未同步时覆盖同一片空间。工作区需求可能随类型、长度或实现改变，不能把这次查询值写死。[CUB DeviceReduce 文档](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html)给出两阶段用法。

## 4. cuBLAS：先把矩阵布局画出来

本章矩阵在逻辑上是：

```text
A = [1 2]    B = [5 6]    A×B = [19 22]
    [3 4]        [7 8]          [43 50]
```

`cublasSgemm` 的普通接口按**列主序**解释连续数组，因此源码中 `A={1,3,2,4}`、`B={5,7,6,8}`，预期输出连续数组是 `{19,43,22,50}`。调用参数 `m=n=k=2`，`lda=ldb=ldc=2`，`alpha=1`、`beta=0`。即使数值看似“差不多”，若把行主序数组原样传给列主序接口，计算的就是另一组矩阵；应先核对布局和 leading dimension。示例用 `1e-5` 检查小矩阵浮点结果，容差只用于该输入。[cuBLAS 官方文档](https://docs.nvidia.com/cuda/cublas/index.html)说明 GEMM 的矩阵约定。

## 5. 选库还是自己写

先用库得到可信结果，再观察完整任务和数据流。标准排序、求和、GEMM 通常先考虑库；手写 kernel 适合表达库没有的融合逻辑、特殊布局或实测瓶颈。组合实现可能把库运算与一个小的转换 kernel 接起来。三者都要核查接口、资源生命周期、正确性和实际耗时。库的 kernel 快不保证整条链路快：主机到设备复制、临时空间、调用频率及输出回传都计入端到端时间。第 12 章将建立同一口径的测量。

## 6. 构建与验证

在仓库根目录运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target library_baselines -j2
./build/outline/book_ch11/library_baselines
ctest --test-dir build/outline -R '^ch11_library_baselines$' --output-on-failure
```

独立构建可用 `cmake -S chapters/ch11-cuda-libraries/examples -B build/ch11-library-standalone`，需同样指定 nvcc 与目标架构。CMake 在有 `CUDA::cublas` 时编译并链接 SGEMM；当前 zyk 缺少 cuBLAS 开发库，所以程序明确打印 `SKIP cuBLAS SGEMM`，只把 Thrust/CUB 记为通过。独立构建与根工程输出见[验证记录](results/validation.md)。CMake 链接已有的静态 CUDA 运行库，避免该环境缺少动态库搜索路径造成程序在启动前失败。

## 7. 常见错误

- CUB 查询工作区大小后忘记第二次调用，读取了未写入的输出。
- 临时工作区在异步计算完成前释放，或并发调用共享同一工作区却没有流顺序约束。
- 把 cuBLAS 列主序参数按行主序理解，或者把 `lda` 误当总元素数。
- `beta` 非 0 却没有初始化输出矩阵 C。
- 在每个 `device_vector` 元素上做一次主机访问，意外产生大量小传输。
- 只测 kernel 或库调用本身，就声称整个业务链路加速。

## 8. 练习与参考答案

1. 本章 Thrust 排序后的数组是什么？答：`[-2,-2,0,1,5,7,7]`。
2. 若 CUB 查询工作区是 4096 字节，第二次调用前需做什么？答：分配至少 4096 字节设备可访问工作区，保持到调用完成；还须有有效输入与输出缓冲区。
3. 上述 2×2 乘法中结果 `C(1,0)` 是多少？答：43，列主序连续数组下标 1。
4. 把第 8 章手写整数归约与本章 CUB 归约比较时，必须保持什么一致？答：相同输入、数据类型、溢出约定、输出定义和同步后读取；再分别记录构建配置与时间口径。
5. 哪些因素可能让库内核很快但整体很慢？答例：频繁主机设备拷贝、每次调用重新分配工作区、过小任务的调用开销、串行同步。

下一章统一计时范围、预热、重复次数和统计方式，再讨论优化是否真实有效。
