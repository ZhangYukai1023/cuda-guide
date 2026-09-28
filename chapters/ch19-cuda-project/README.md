# 第 19 章：组织 CUDA C++ 工程

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch18-graphs-memory-pool/README.md)

前面的章节主要用单文件示例说明一个概念。本章把一个数组仿射变换 `output[i]=input[i]*scale+bias` 拆成可复用的静态库：公开接口、资源管理、kernel、跨文件设备函数和调用示例分开。完整工程在 [examples](examples/)；静态库及最终程序已在 zyk 完成跨文件设备链接，并通过 CPU 对照和接口错误检查。

## 1. 文件边界与构建目标

| 文件 | 职责 |
| --- | --- |
| `include/array_ops.hpp` | `ArrayTransformer` 的公开接口与所有权约定 |
| `src/array_ops.cu` | 设备缓冲、Stream、异步 H2D/kernel/D2H 与错误传播 |
| `src/array_kernel.cu` | Grid/Block、边界守卫和 kernel 启动 |
| `src/device_math.cu` | 被另一 CUDA 翻译单元调用的 `__device__` 运算 |
| `src/main.cu` | 页锁定 host 输入输出、CPU 对照和错误接口检查 |
| `CMakeLists.txt` | `cuda_array_ops` 静态库、`array_module_demo` 程序和 CTest |

`array_kernel.cu` 调用 `device_math.cu` 定义的设备函数，因此 CMake 为库和最终程序启用 `CUDA_SEPARABLE_COMPILATION`，由构建系统处理跨文件设备符号的链接。若把设备函数改为头文件内联，可能不再需要这个跨文件符号；是否采用哪种结构应服从代码复用与实际构建需求，而不是为“多文件”增加无谓复杂度。示例先用静态库；动态库还涉及导出符号、运行时搜索路径、ABI 与部署版本，本章说明方向但没有把动态库标为已实现。

## 2. 明确谁拥有哪份资源

`ArrayTransformer(capacity)` 拥有两块最多 `capacity` 项的设备数组和一条非默认 Stream，构造时分配、析构时同步并释放。调用者拥有页锁定 host 输入/输出，并保证它们从 `enqueue()` 开始直到 `wait()` 返回都有效且内容不被改写。`enqueue()` 只排队 H2D、kernel、D2H，返回并不意味着结果已可读；`wait()` 同步并传播异步执行错误。本接口一次只允许一个在途请求：未 `wait()` 再 `enqueue()` 会抛 `logic_error`。对象没有线程安全承诺。

`count=0` 是无操作；`count>capacity`、host 指针为空会报参数错误。构造函数在部分分配失败时清理已取得的资源。析构函数不能抛异常，因此真正关心执行错误的调用者必须显式调用 `wait()`。示例在等待后比较 1003 项和 7 项结果；还检查重复入队与超容量请求被拒绝。更完整的生产接口应在大小转换前检查乘法和 Grid 溢出、定义取消与异步完成句柄，并处理多 Stream 调用；本章小示例的容量固定为 1003。

## 3. 错误在哪里出现

`cudaMalloc` 等同步 API 可能在调用处失败。kernel 启动后立即用 `cudaGetLastError()` 查启动配置；执行中的非法访存等异步错误往往在 `wait()` 的 `cudaStreamSynchronize` 才显现。`enqueue()` 若排队中途失败，会先把可能已经排进 Stream 的工作排干，再让对象回到可复用状态；析构时也等待剩余工作后才释放设备缓冲。日志保留失败 API 与 CUDA 错误字符串，调用者接收 C++ 异常。生产库可改用错误码或 `expected` 风格，但必须保持“调用失败”与“异步执行失败”的明确边界。

## 4. Runtime、Driver 与 NVRTC 的位置

本章使用 **CUDA Runtime API**，便于在 CMake 编译好的 CUDA C++ 工程里启动 kernel、管理 Stream 与内存。**Driver API** 适合需要更显式控制 context、module 或动态加载的系统；**NVRTC** 用于运行时编译 CUDA 源码，例如按动态形状生成代码。这两者只介绍适用方向，示例没有加载 Driver module，也没有调用 NVRTC，不能宣称已验证。需要时应把编译缓存、错误日志、模块生命周期和驱动兼容性一并设计。[CUDA 编程指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)与对应 Runtime/Driver/NVRTC 官方文档可作接口依据。

## 5. 构建与测试

独立构建本章：

```bash
cmake -S chapters/ch19-cuda-project/examples -B build/ch19-project-standalone \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch19-project-standalone -j2
ctest --test-dir build/ch19-project-standalone --output-on-failure
```

根工程目标为 `array_module_demo`，CTest 为 `ch19_array_module`。实测构建包含 `Linking CUDA device code` 阶段，退出码 0；记录见[验证记录](results/validation.md)。其他工具链仍需按同样步骤验证。

## 6. 常见错误

- 把 `enqueue()` 返回当作计算完成，马上读取 host 输出或覆盖 host 输入。
- 在上一次异步 D2H 尚未完成时复用同一设备或页锁定缓冲区。
- 构造函数第二次分配失败时漏掉第一次分配和 Stream 的清理。
- 析构函数抛异常或完全不等待，在 GPU 仍使用缓冲时释放资源。
- 跨 `.cu` 文件引用 `__device__` 函数，却没有启用必要的分离编译与设备链接。
- 静态库能编译就认为最终可执行文件一定能链接；两阶段都要验证。
- 把 CPU 参考和 GPU kernel 共同依赖同一错误索引逻辑，失去独立对照。

## 7. 练习与参考答案

1. `input=[1,-2,3]`、`scale=2`、`bias=1` 的输出？答：`[3,-3,7]`。
2. 调用 `enqueue()` 后谁负责保持 host 输入有效？答：调用者，直到 `wait()` 返回；模块拥有自己的设备缓冲与 Stream。
3. 如果用户忘了 `wait()` 就析构，对执行错误还能可靠反馈吗？答：析构会同步并清理资源，但不能抛异常；调用者应显式 `wait()` 获取错误与结果。
4. 为什么本章静态库需要设备链接？答：`array_kernel.cu` 的 kernel 调用另一个 `.cu` 中定义的 `__device__` 函数，设备符号跨翻译单元。
5. 若要增加“多请求并发”，先需要改什么契约？答：定义每个请求独立的缓冲/Stream或明确复用时机、结果所有权与完成句柄，并规定线程安全和错误传播。

[下一章：测试、部署与性能回归](../ch20-testing-deployment/README.md)把正确性、Sanitizer、环境与性能回归记录组合成交付检查。
