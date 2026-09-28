# 第 4 章：管理数据、内存和资源

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch03-array-indexing/README.md)

本章把第 3 章的索引计算放进完整的数据生命周期：输入在 CPU 上产生，复制到 GPU，核函数处理，结果回到 CPU，最后释放资源。用数组平方建立最小流程，再复用缓冲区做多轮更新，最后把两次小输入合并成一次传输。完整源码见 [memory_resources.cu](examples/memory_resources.cu)。

## 1. 最小示例：数组平方

输入 `[-3,-2,-1,0,1,2,3]`，CPU 手算平方结果为 `[9,4,1,0,1,4,9]`。GPU 核函数只做一件事：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) output[i] = input[i] * input[i];
```

为了让这个表达式真正处理数据，Host 端依次完成：

1. 为输入与输出各申请一块设备缓冲区。
2. 用 Host-to-Device 传输把输入复制到 GPU。
3. 启动 kernel，检查启动错误并等待执行完成。
4. 用 Device-to-Host 传输把结果取回，与 CPU 参考逐项比较。
5. 释放两块设备缓冲区。

[公共头文件](../../common/cuda_support.cuh) 中的 `DeviceBuffer<int>` 将 `cudaMalloc` 与 `cudaFree` 绑在对象生命周期上，`upload` 和 `download` 明确指定 `cudaMemcpy` 方向。它禁止复制，以免两个对象都认为自己拥有同一指针。这是一个教学封装，析构函数不会抛异常，也没有记录分配所属设备；运行错误应在显式的 `cudaGetLastError` 与 `cudaDeviceSynchronize` 处报告。不能把 Host 指针直接当成本例设备指针解引用。

## 2. 递进示例：一块缓冲区做五轮更新

假设数组的每个值开始为 `i%11`，连续加 1 五次后应为 `i%11+5`。源码对长度 1003 只申请一次 `DeviceBuffer<int>`，上传一次，在同一地址上启动五次 `increment`，同步后下载一次。CPU 参考计算的是最终值，GPU 输出逐项完全一致。

```cpp
DeviceBuffer<int> reused(n);
reused.upload(values);
for (int step = 0; step < 5; ++step) {
    increment<<<(n + 127) / 128, 128>>>(reused.data, n);
    CUDA_CHECK(cudaGetLastError());
}
CUDA_CHECK(cudaDeviceSynchronize());
verify("reused_buffer", reused.download(), expected);
```

这里同一默认流里的提交保持顺序，最后一次同步等待前面工作完成。若中途要由 CPU 读取某轮结果，必须在正确的同步和复制之后读取。缓冲区复用减少了重复申请和传输的次数，但本章没有测量复用带来多少速度变化，不能给出加速比。

## 3. 将两份小输入打包

再设两个长度 7 的数组：`a=[0,1,2,3,4,5,6]`，`b=[10,11,12,13,14,15,16]`。逐项相加应得 `[10,12,14,16,18,20,22]`。Host 将它们排列为 `[a 的 7 项 | b 的 7 项]`，用一次 `upload` 复制到设备；kernel 读取 `packed[i]` 和 `packed[n+i]`，把结果写到独立输出缓冲区。

这说明接口层面可以把两份数据合并为一次 H2D 调用。打包本身需要 Host 工作，也可能增加复制量；当前示例只验证正确性和调用结构，没有对“两次传输”和“一次传输”进行公平计时，不能据此断言打包更快。

## 4. 同步后的完整任务计时

源码对数组平方先预热 5 次，再重复 20 次。Host 的 `steady_clock` 包围每次调用 `square_task` 的完整范围：设备分配、H2D、kernel、显式同步、D2H、输出数组创建与设备释放；输入生成与 CPU 参考验证在计时范围外。打印总时长除以 20 的均值，名称为 `square_end_to_end_mean_ms`。

这不是纯 kernel 耗时。kernel 部分很短，测量范围包含许多其他操作；本章没有独立测 kernel，也没有 CPU 性能基线，不提供 GPU 加速结论。第 12 章将使用 CUDA Event 与更完整的统计来分别测量 kernel 和端到端过程。本章这次运行观察到均值 0.047444 ms，仅适用于该次机器与条件。

## 5. 构建、运行与实测输出

在服务器项目根目录运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target memory_resources -j2
./build/outline/book_ch04/memory_resources
ctest --test-dir build/outline -R '^ch04_memory_resources$' --output-on-failure
```

也可用 `-S chapters/ch04-memory-resources/examples -B build/ch04-standalone` 独立配置本章。实测程序输出：

```text
square: n=7 max_abs_error=0 mismatches=0 PASS
square_end_to_end_mean_ms=0.047444 (warmup=5 repeats=20; allocation through free)
reused_buffer: n=1003 max_abs_error=0 mismatches=0 PASS
packed_inputs: n=7 max_abs_error=0 mismatches=0 PASS
```

计时值会随运行波动；其余整数结果应与 CPU 参考完全一致。实际环境与测试状态见 [验证记录](results/validation.md)。

## 6. 常见错误与排查

- 把元素数量直接作为 `cudaMemcpy` 字节数；对 `n` 个 `int` 应使用 `n*sizeof(int)`，并与分配容量一致。
- 把 `cudaMemcpyHostToDevice` 与 `cudaMemcpyDeviceToHost` 用反，或把输入和输出指针互换；先画出数据流，再逐个核对参数。
- 在多轮计算中每轮重新申请并上传同一数组，却把重复成本误称为 kernel 耗时。
- 取回结果前只检查启动错误，不等待可能尚未完成的设备工作。
- 让两个可复制的对象共同管理同一裸设备指针，导致重复释放。公共 `DeviceBuffer` 明确禁止复制。

本机没有找到 Compute Sanitizer，因此未执行 memcheck。CPU 对照通过不等于内存工具证明通过。故障练习中若故意缩小申请字节数，不要把“程序没崩溃”当作正确，修复后须重新运行正确性测试。

## 7. 练习与参考答案

1. 输入 `[-2,0,3]` 平方后是什么？答：`[4,0,9]`；输入、输出分配都要能容纳 3 个整数。
2. 值 7 连续递增 5 次，最终是多少？答：12。若只同步前四次，第五次结果可能尚未完成，不能直接用 Host 读取。
3. 两个长度为 `n` 的 `int` 数组打包后，设备输入至少需要多少字节？答：`2*n*sizeof(int)`；输出还需另有 `n*sizeof(int)`。
4. 本章打印的 0.047444 ms 是 kernel 耗时吗？答：不是，它是该次运行的端到端均值，包含分配、传输、同步与释放等步骤。
5. 将五轮更新的输入改成全零，CPU 参考应是什么？答：长度 1003 的全 5 数组；还应逐项核对 GPU 结果。

下一章将从执行模型与 Warp 解释线程组如何在硬件上执行，进一步理解分支和工作划分。
