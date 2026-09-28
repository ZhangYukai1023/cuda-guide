# 第 1 章：认识 CUDA，准备实验环境

本章目标：分清 CPU、GPU、驱动和编译器的职责，查询一块 CUDA 设备，并亲手完成一次可以用 CPU 核对的 GPU 计算。读者只需了解 C++ 函数、数组、指针和基本循环。暂时不需要理解共享内存、warp、异步流水线或多 GPU。

本章示例在当前服务器实际验证。原始大纲未能取得，经用户授权直接编写；用户后续已授权连续编写，全书导航见 [项目首页](../../README.md)。

## 1. 从八次加法开始

假设有两个数组：

```text
a = [1,  2,  3,  4,  5,  6,  7,  8]
b = [10, 20, 30, 40, 50, 60, 70, 80]
c = [11, 22, 33, 44, 55, 66, 77, 88]
```

普通 C++ 可以用一个循环计算：`c[i] = a[i] + b[i]`。第 0 项只依赖 a[0] 和 b[0]，不需要等待第 1 项。因此可以把八个位置交给八个工作者，同时完成。CUDA 让我们把这样的工作交给 NVIDIA GPU。

这里的“工作者”是 CUDA 线程，不等于一颗独占的物理处理器。线程如何映射到硬件，后面再讲。本章只用“一个线程负责一个下标”理解代码。

八个数太少，不能因为使用 GPU 就期待更快。准备 GPU 工作、分配内存和传输数据都有成本。本章验证的是计算正确性，不进行性能比较。

再看一个图像方向的小问题。一张 2×3 的灰度图可以按行存成六个整数：

```text
图像：                按行展开：
  0   30   60         [0, 30, 60, 90, 120, 250]
 90  120  250
```

如果每个像素加 10，且超过 255 时截断，手算结果是 `[10, 40, 70, 100, 130, 255]`。每个像素独立，和数组加法有相似之处。但去噪、插值与滤波往往还要读取邻居；不能直接把所有图像操作当成独立加法。这个小图只作手算说明，实际 GPU 示例仍使用整数数组。

## 2. 认识实验环境的几个组成部分

| 名称 | 本章中的作用 |
| --- | --- |
| CPU / host（主机） | 运行 main，准备数据，发起 GPU 工作，检查结果 |
| GPU / device（设备） | 执行并行计算函数 |
| NVIDIA 驱动 | 让操作系统和程序能够使用 GPU |
| CUDA Toolkit | 提供 CUDA 编译、头文件、运行库等开发组件 |
| nvcc | 编译包含 CUDA 扩展语法的 .cu 文件，并协调主机端编译 |
| C++ 编译器 | 编译主机端 C++ 代码 |
| CMake | 组织构建过程，本身不是 CUDA 编译器 |

CUDA 不是把普通 C++ 文件换个后缀就自动并行。我们需要明确指定哪些函数在 GPU 上运行，以及数据如何送到 GPU、如何取回。

本章采用显式设备内存：CPU 数组与 GPU 数组分别申请，数据通过复制传递。不要在 CPU 上直接解引用设备指针。内存申请、复制和释放 API 的定义可查阅 [NVIDIA CUDA Runtime 内存接口文档](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html)。

## 3. 本机检查结果与编译参数

实测主机名为 `ubuntu2404`，工作目录为 `/data2`；这里保留实际主机名，不把它改写成 zyk。初始环境记录保存在 [environment.md](results/environment.md)，后续验证保存在 [validation.md](results/validation.md)。

| 项目 | 本机实测 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| 显存 | nvidia-smi 显示 16311 MiB |
| 驱动版本 | 595.84 |
| GPU 计算能力 | 12.0 |
| nvcc | CUDA 12.8，V12.8.93 |
| C++ 编译器 | GCC 13.3.0 |
| CMake | 3.28.3 |
| 主机调试器 | GDB 15.0.50.20240403 |
| CUDA 调试、分析工具 | 检查范围内未找到 Compute Sanitizer、cuda-gdb、nsys、ncu |

可重复执行的只读检查：

```bash
hostname
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
/home/zhangyukai/.local/cuda/bin/nvcc --version
/home/zhangyukai/.local/cuda/bin/nvcc --list-gpu-code
c++ --version
cmake --version
command -v compute-sanitizer cuda-gdb nsys ncu gdb
```

最后一行用于查找命令，部分命令不存在时可能返回非零退出码；它不是示例测试。

有三个容易混淆的数字：驱动版本 595.84、Toolkit 版本 12.8、计算能力 12.0。它们分别描述驱动、开发工具和 GPU 架构。另一个数字是 `nvidia-smi` 顶部显示的 CUDA Version 13.2；它表示驱动支持的 CUDA 版本信息，不能用来证明已经安装 Toolkit 13.2。本机 Toolkit 版本依据是实际 nvcc 输出。

本机 nvcc 不在默认 PATH 中，使用绝对路径即可，无需修改系统配置。GPU 查询显示计算能力 12.0，nvcc 支持列表包含 `sm_120`，所以选择 `-arch=sm_120`。这项选择也已通过真实构建和运行验证。换一台 GPU 后，应重新查询架构，不要照抄 120。

本章参数还有：

- `-std=c++17`：选择 C++17。
- `-O2`：启用编译优化，不代表我们已测量性能。
- `-lineinfo`：保留设备代码行号信息，便于后续工具定位。
- `-o`：指定输出程序路径。

## 4. 最小示例：让程序报告 GPU

先不写计算函数。这个程序查询 CUDA 可见设备，打印名称和计算能力。这样可以先判断运行库是否能连接到设备。设备查询成功仍不能替代一次真正的 kernel 运行，因此后面还有数组加法。

完整文件：[device_info.cu](examples/device_info.cu)。

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return EXIT_FAILURE;
    }
    std::printf("CUDA devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, i));
        std::printf("Device %d: %s, compute capability %d.%d\n",
                    i, prop.name, prop.major, prop.minor);
    }
    return EXIT_SUCCESS;
}
```

在项目根目录执行以下命令。后文所有命令也以此目录为起点：

```bash
cd /data2/cuda-guide
mkdir -p build/ch01-direct
/home/zhangyukai/.local/cuda/bin/nvcc -std=c++17 -O2 -lineinfo -arch=sm_120 \
  chapters/ch01-getting-started/examples/device_info.cu \
  -o build/ch01-direct/device_info
./build/ch01-direct/device_info
```

本机实际输出，也是相同环境下的预期结果：

```text
CUDA devices: 1
Device 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0
```

其他机器的设备数和名称可以不同。若设置了 CUDA_VISIBLE_DEVICES，可见设备还可能只是物理设备的子集。

`cudaGetDeviceCount` 写入设备数；`cudaGetDeviceProperties` 填充属性结构。这里用 `check` 检查每次 CUDA 调用的返回值。一旦失败，程序打印错误并以非零状态退出，避免继续使用无效结果。GDB 用于主机代码调试；本机没有找到 CUDA 专用调试器，因此本章不声称完成设备断点调试。

## 5. 递进示例：把八次加法交给 GPU

先在 CPU 生成 a 和 b，再在 CPU 上算出参考数组。随后申请三个 GPU 数组、复制两个输入、启动计算、等待完成、取回结果，最后逐项比较。只有 GPU 与 CPU 的每个元素一致，程序才报告 PASS。

GPU 函数通常称为 kernel（核函数）。本例的核心只有：

```cpp
__global__ void add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

`__global__` 表示这个函数由主机启动、在设备执行。`<<<blocks, threads>>>` 是启动配置：使用多少个线程块，每块有多少个线程。先把 block 当作线程分组即可。

举例：如果每组 4 个线程，第 0 组负责下标 0、1、2、3，第 1 组负责 4、5、6、7。上面的公式就是“组号 × 每组人数 + 组内编号”。正式源码使用每组 128 个线程，这只是一个简单的实验配置，不是最优参数结论。

当 n=129 时，`(129 + 128 - 1) / 128 = 2`，会启动 256 个线程。下标 129 到 255 超出数组，必须被 `if (i < n)` 排除。n=8 时同样有多余线程，这不影响正确性。

完整文件：[vector_add.cu](examples/vector_add.cu)。

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 本章使用立即失败的错误处理，适合独立的小程序。
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

bool run_case(int n, bool print_values) {
    std::vector<int> a(n), b(n), result(n), reference(n);
    for (int i = 0; i < n; ++i) {
        a[i] = i + 1;
        b[i] = 10 * (i + 1);
        reference[i] = a[i] + b[i];
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    int *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    check(cudaMalloc(&device_a, bytes));
    check(cudaMalloc(&device_b, bytes));
    check(cudaMalloc(&device_c, bytes));
    check(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(device_a, device_b, device_c, n);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_a));
    check(cudaFree(device_b));
    check(cudaFree(device_c));
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (result[i] != reference[i]) ++mismatches;
    }
    if (print_values) {
        std::printf("result:");
        for (int value : result) std::printf(" %d", value);
        std::printf("\n");
    }
    std::printf("n=%d, mismatches=%d, %s\n", n, mismatches,
                mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int main() {
    bool passed = true;
    // 1：单元素；8：可手算；129：跨 block 且非整块；1000：稍大规模。
    for (int n : {1, 8, 129, 1000}) {
        if (!run_case(n, n == 8)) passed = false;
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

构建与运行：

```bash
/home/zhangyukai/.local/cuda/bin/nvcc -std=c++17 -O2 -lineinfo -arch=sm_120 \
  chapters/ch01-getting-started/examples/vector_add.cu \
  -o build/ch01-direct/vector_add
./build/ch01-direct/vector_add
```

本机实际输出，也是预期结果：

```text
n=1, mismatches=0, PASS
result: 11 22 33 44 55 66 77 88
n=8, mismatches=0, PASS
n=129, mismatches=0, PASS
n=1000, mismatches=0, PASS
```

`mismatches` 是不相等的元素数量。整数在本例的数值范围内不会溢出，可以逐项精确比较。后续浮点数计算需要讨论误差容限，不能一概采用完全相等。

读代码时特别注意五处：

1. `bytes = n * sizeof(int)` 是字节数。数组有 8 个元素，不意味着只需要 8 字节。
2. `cudaMalloc` 申请设备内存；输入向量仍保留在 CPU 上。
3. 两个输入使用 HostToDevice，输出使用 DeviceToHost，方向不能写反。
4. 启动 kernel 后立即检查 `cudaGetLastError()`，再用 `cudaDeviceSynchronize()` 等待并检查执行错误。仅仅成功提交工作，不等于工作已经完成。
5. 取回结果后释放设备内存，最后与 CPU 参考数组比较。

此程序固定测试四个正数长度，未提供任意长度输入接口，也没有宣称支持 n=0 或超大数组。错误处理为教学用的立即退出；较大项目中需要用资源管理机制保证中途失败时也能有序清理资源，后续工程章节再展开。

本例各线程只写自己的输出位置，没有线程间共享数据，不需要 `__syncthreads()`。主机等待 GPU 完成与线程块内部同步是不同的事，本章先掌握前者。关于 kernel 启动和线程组织，可参见 [CUDA C++ Programming Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)。

## 6. 用 CMake 重复构建与测试

直接 nvcc 命令适合理解编译过程。文件增多后可以使用 [CMakeLists.txt](examples/CMakeLists.txt)，避免手动重复拼写源文件。

```bash
cmake -S chapters/ch01-getting-started/examples -B build/ch01-cmake \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch01-cmake -j2
ctest --test-dir build/ch01-cmake --output-on-failure
```

`-S` 是源码目录，`-B` 是生成文件目录。CMake 的架构写法是 120，而 nvcc 的写法是 sm_120。构建目录被 Git 忽略。源码和运行记录则保留在版本控制范围内。

本机实际通过两项测试：device_info 与 vector_add。CTest 显示 `100% tests passed, 0 tests failed out of 2`。测试通过依据是退出状态；vector_add 只有逐项比较通过才返回成功。

CTest 的时间包括进程启动等成本，不是 kernel 耗时。本章没有性能实验。未来性能章节会单独设置预热、重复测量和同步，并分别报告 kernel 耗时与端到端耗时；这里不从测试时长推导吞吐率。

## 7. 内存检查的状态

本机 PATH 和已发现 CUDA 的 bin 目录中没有 Compute Sanitizer；对用户 .local 目录的有限深度搜索也没有找到它。因此本次未执行 memcheck 或 synccheck。CPU 对照通过不等于内存检查通过。

如果后续环境已经具备该工具，可以在项目根目录运行：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/ch01-direct/vector_add
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/ch01-direct/vector_add
```

这些是待执行命令，不是本次成功记录。正常目标是没有工具报告的错误，实际结论应以工具日志为准。本例没有块内同步原语，synccheck 的覆盖价值有限；更复杂的同步示例将在后续章节检查。本次没有自动安装任何工具。

## 8. 常见错误与排查顺序

| 现象或错误 | 先检查什么 |
| --- | --- |
| nvcc: command not found | 本机编译器位于用户目录，使用本章绝对路径；不要据此直接认定未安装 CUDA |
| nvidia-smi 正常却不能编译 | 驱动与开发工具不同，检查 nvcc、头文件、运行库和主机编译器 |
| 不支持 sm_120 | 检查实际调用的 nvcc 版本及其 --list-gpu-code 输出 |
| no kernel image is available | 核对运行设备与编译目标架构，不要只查看驱动版本 |
| 计算结果错或非法内存访问 | 核对字节数、复制方向、下标范围以及是否检查 CUDA 错误 |
| 129 个元素只有前 128 个正确 | 检查块数是否使用向上取整，并保留越界判断 |
| 在 CPU 上读取 device_a[0] | 设备指针不是本例 CPU 可直接读取的数组，先复制回主机 |
| 把 GPU 提交时间当计算时间 | 提交可能先返回；本章通过同步检查完成情况，没有做性能测量 |

不要为了让命令“成功”而忽略返回值。示例的目标是遇到问题时及时暴露错误，而不是无条件打印 PASS。

## 9. 练习与参考答案

### 练习 1：手算与分工

设 a=[2,4,6,8,10]，b=[1,1,1,1,1]，每块 4 个线程。需要几个块？哪些线程不该写输出？

**参考答案：** c=[3,5,7,9,11]；需要 2 个块，共 8 个线程。全局下标 5、6、7 应被边界判断排除。第 1 块的第 0 个线程全局下标为 4。

### 练习 2：增加一个块边界测试

在 main 的长度列表中增加 128，重新运行上述 nvcc 构建命令和程序。预期新增输出是什么？

**参考答案：** 应新增 `n=128, mismatches=0, PASS`。这是练习的预期，不属于已记录的四组实测。128 恰好只需要一个块，与 129 的跨块情况配合检查边界。

### 练习 3：改成减法

将 kernel 改成 a[i]-b[i]。仅修改 kernel 足够吗？

**参考答案：** 不够，还需修改 CPU 参考公式 `reference[i] = a[i] - b[i]`，然后重新构建运行。8 个元素的数学预期是 [-9,-18,-27,-36,-45,-54,-63,-72]。如果不修改参考公式，验证应失败；这也说明 CPU 参考必须代表想解决的问题。

### 练习 4：判断环境版本

nvidia-smi 显示 CUDA 13.2，而 nvcc 显示 12.8，应在环境记录里写什么？

**参考答案：** 分别记录驱动相关显示与实际 Toolkit 编译器版本，不把两者合并成一个“CUDA 版本”。本机编译器是 12.8.93。

### 练习 5：图像亮度的溢出

像素值 250 加 10，8 位灰度结果应是多少？为什么不能先把和转换成 8 位无符号数再截断？

**参考答案：** 应为 255。先用足够宽的整数得到 260，再截断到 255；过早转成 8 位无符号数会回绕成 4，丢失本来应截断的信息。本题仅为手算练习，后续图像专篇会提供完整图像、效果图与数值验证。

## 10. 本章完成范围

已完成设备查询、显式内存复制、整数数组 kernel、CPU 对照、直接编译和 CMake 测试。未执行 Compute Sanitizer，也未开展性能实验。

下一章解释线程下标与二维布局，再进入内存、工程实践及应用。继续阅读 [第 2 章](../ch02-thread-indexing/README.md)。
