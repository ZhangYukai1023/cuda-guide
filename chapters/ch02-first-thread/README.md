# 第 2 章：从一个 GPU 线程开始

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch01-getting-started/README.md)

本章目标是读懂并运行一次完整的 GPU 计算：在 CPU 上申请设备内存、启动一个 GPU 线程、等待它完成、取回结果、核对并释放内存。先修知识只需 C++ 函数、指针和数组。完整源码是 [first_threads.cu](examples/first_threads.cu)。

## 1. 用一个线程写入 42

先不处理数组。如果一个线程只做 `out[0] = 42`，CPU 预期结果就是 42。CUDA 中由 CPU 执行的代码称为 Host 代码，由 GPU 执行的代码称为 Device 代码。用 `__global__` 声明的函数是从 Host 启动的 kernel：

```cpp
__global__ void write_42(int* out) { out[0] = 42; }
// 在 main 调用的 run 中：
write_42<<<1, 1>>>(device);
```

`<<<1, 1>>>` 前面的 1 是线程块数量，后面的 1 是每块线程数量。总共只启动 1 个线程。三重尖括号是 CUDA C++ 的启动语法，普通 C++ 编译器不能独立解析。NVIDIA 的 [CUDA C++ 入门说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html) 和 [Runtime API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/index.html) 是本章语法与 API 的依据。

这个线程写的是 GPU 内存地址。CPU 变量的地址和本例设备指针不是一回事；需要先申请设备内存，再把地址交给 kernel。

## 2. 一次计算的六个步骤

| 顺序 | 本章源码 | 要解决的问题 |
| --- | --- | --- |
| 1 | `cudaMalloc(&device, 8*sizeof(int))` | 申请能装 8 个整数的 GPU 空间；前两例只用第一个位置 |
| 2 | `write_42<<<1,1>>>(device)` | 将工作提交到 GPU |
| 3 | `cudaGetLastError()` | 检查已经报告的启动错误 |
| 4 | `cudaDeviceSynchronize()` | 等待 GPU 工作完成，并检查异步执行错误 |
| 5 | `cudaMemcpy(host,device,sizeof(int),cudaMemcpyDeviceToHost)` | 将一个结果复制回 CPU |
| 6 | `cudaFree(device)` | 用完后释放设备内存，即使前面的验证失败也会调用 |

`cudaMalloc` 的参数是设备指针的地址，调用成功后会将分配结果写入 `device`。`8*sizeof(int)` 按字节计算；写成 `8` 只申请 8 个字节，不能容纳 8 个整数。Host 上的 `host[8]` 是另一块内存。`cudaMemcpyDeviceToHost` 明确指出传输方向。

kernel 启动一般对 Host 异步，因此只检查启动表达式还不能证明计算已经完成。本例在每次启动后依次检查 `cudaGetLastError`、`cudaDeviceSynchronize`，再拷贝输出。`check` 把失败的 API 和 CUDA 错误字符串写到 stderr 并返回 false；`main` 最终返回非零，CTest 因而会报告失败。`run` 返回后仍会执行 `cudaFree`，避免失败路径跳过释放。第一章引入的 RAII 封装是工程中的另一种写法，本章先把原始 API 顺序完整展示出来。

## 3. 从一个线程递进到八个线程

第一步把 42 写到 `device[0]`；第二步仍只启用一个线程，将 `7 + 5` 写到同一位置。CPU 参考值是 12。这两步复用同一块设备内存，不需要重新 `cudaMalloc`。

第三步启动一个包含 8 个线程的块：

```cpp
__global__ void write_ids(int* out) { out[threadIdx.x] = threadIdx.x; }
write_ids<<<1, 8>>>(device);
```

可以手算线程与数组位置的对应关系：

| `threadIdx.x` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `out` 下标 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 写入值 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

`threadIdx.x` 在本章只表示**块内**编号。因为只有一个块，它恰好也是数组下标；多个块时需要把 `blockIdx.x` 纳入公式，这是第 3 章的主题。源码将 8 个整数全部取回，与 CPU 上的 `0..7` 逐项比较；任意一项不相同便打印下标、实际值和预期值并返回失败。

## 4. 构建、运行与实测结果

在服务器 `/data2/cuda-guide` 执行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target first_threads -j2
./build/outline/book_ch02/first_threads
ctest --test-dir build/outline -R '^ch02_first_threads$' --output-on-failure
```

本机是计算能力 12.0 的 RTX 5060 Ti，Toolkit 的 `nvcc` 支持 `sm_120`，因此 CMake 架构值为 `120`。换设备时应重新核对，不要把这个值当作通用参数。也可把 `-S` 改为 `chapters/ch02-first-thread/examples`，将 `-B` 改成单独的构建目录来独立配置本章。

程序实际输出：

```text
write_42: got=42 expected=42 PASS
add_scalars: got=12 expected=12 PASS
write_ids: got=0,1,2,3,4,5,6,7 PASS
```

以上三项均有 CPU 参考值。运行没有测量性能，不能从它们推断 GPU 比 CPU 快。编译、CTest 与环境细节见 [验证记录](results/validation.md)。

## 5. 常见错误与定位

- 将 Host 数组指针直接传给当前 kernel。这个示例使用显式设备分配，应传 `cudaMalloc` 返回的指针；发生错误时先检查设备地址与传输方向。
- `cudaMalloc` 按元素数而不是字节数申请。对于 8 个 `int`，要计算 `8*sizeof(int)`。
- 把 `<<<1,8>>>` 理解为 8 个 Block、每块 1 个线程。启动参数顺序是 Block 数、每块线程数。
- 启动后只检查 `cudaGetLastError` 就认为计算成功。还需要在同步点检查执行状态、取回数据并比较。
- 只打印 PASS 而不让错误返回非零。本例每个 API、三个 CPU 对照和释放结果都会影响退出码。

本机尚未找到 Compute Sanitizer，因此没有声称内存检查或竞争检查通过。这里的三个 kernel 没有共享内存或线程间数据依赖；同步故障将留到后续章节。

## 6. 练习与参考答案

1. 把 `add_scalars` 的参数改为 `-3, 9`，预期 `host[0]` 是多少？答：6。还要把 CPU 参考表达式改为 `-3+9`，否则验证会按旧值 12 失败。
2. 如果 `write_ids<<<1,8>>>` 改为 `<<<1,4>>>`，剩余四项能否保证是 4、5、6、7？答：不能。只启动了 4 个线程，它们只写下标 0 到 3；剩余位置没有由本次 kernel 定义。
3. 若要写 16 个编号，除了 `<<<1,16>>>` 还要改什么？答：把设备申请与 Host 数组容量改为 16 个 `int`，把复制字节数和 CPU 验证循环范围都改为 16；不能只改启动参数。
4. 为什么本章可以在 `run` 失败后释放设备指针？答：`main` 先调用 `run(device)` 保存结果，再无条件调用 `cudaFree(device)`，最后根据两项状态决定退出码。

本章完成了从 Host 到单线程 Device、再回到 Host 的完整闭环。下一章会用 `blockIdx.x * blockDim.x + threadIdx.x` 让多个块共同处理任意长度的数组。
