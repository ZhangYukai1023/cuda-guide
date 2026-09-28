# 第 4 章：工程组织、错误处理与性能测量

[全书导航](../../README.md) · [上一章](../ch03-memory-and-synchronization/README.md) · [下一章](../ch05-image-filtering/README.md)

## 1. 一个 PASS 应该意味着什么

程序打印 PASS 很容易。可信的 PASS 至少需要：CUDA 调用成功、GPU 工作完成、结果与参考值一致。把验证、计时、资源释放写成混乱的一团，会让一个遗漏掩盖另一个遗漏。

最小例子是仿射变换 y=2x+1。输入 [0,1,2,3] 的手算结果为 [1,3,5,7]。本章扩大到 1048576 个 float，仍保留逐项 CPU 对照，然后增加有清楚边界的计时。

## 2. 错误处理和资源所有权

[公共头文件](../../common/cuda_support.cuh) 定义 CUDA_CHECK。它把失败调用的文件、行号、表达式与 CUDA 错误字符串组合成异常；main 通过 guarded 捕获异常、打印原因并返回非零。

启动 kernel 后检查 cudaGetLastError，再在需要结果时同步。这两处检查解决不同问题：提交配置错误可能立即出现，执行中的错误可能要到同步时才报告。

DeviceBuffer 是一个小型 RAII 对象：构造时申请，析构时释放。禁止复制，避免两个对象误以为自己拥有同一块设备内存。它提供按元素数量检查的 upload 与 download，空数组不申请内存。

析构函数不能随意抛异常，因此公共封装的析构采用尽力清理，不将 cudaFree 的错误转成第二个异常；真正的执行错误应在前面的显式同步处报告。这是教学用封装，不是完整生产级库：它不记录分配所属设备，不提供移动操作，不处理跨设备所有权。第 10 章会在正确设备上显式销毁对象。

CMake 把每章变成可单独构建的项目，根 CMakeLists 再统一组织。不要将 .o 或 GPU 可执行文件当作可移植源码提交；当前 build/ 已被 Git 忽略。若更换编译器路径或版本，使用新的构建目录，避免旧缓存干扰。

## 3. 最小示例：先验证再谈时间

核函数仍然是一条表达式：

```cpp
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i<n) b[i]=2.0f*a[i]+1.0f;
```

输入使用 (i%101)/8，数值范围有限，结果与 CPU 参考在本机完全一致。一般浮点验证应使用 abs_error <= atol+rtol*abs(reference)，并额外拒绝 NaN 和无穷。本例选择可以精确核对的数据，所以使用零容差。

测试不是性能基准。CTest 报告的总秒数会包含启动进程、初始化和 CPU 验证。接下来只对明确划定的工作区间计时。

## 4. 递进示例：分别测量 GPU 序列和端到端过程

本章先执行 5 次预热并同步，再重复 100 次。第一次 CUDA 调用可能带有初始化成本；预热也不保证 GPU 时钟完全稳定，因此结果仍应视为本机该次运行的观察。

第一种计时用同一默认流上的 CUDA events 包围 100 次 kernel，等待结束 event，再除以 100。它表示设备时间线上这段 kernel 序列的平均耗时，可能包括相邻提交之间的空隙，不等于每个单独 kernel 的精确硬件执行延迟。

第二种计时用 steady_clock 包围 100 次“H2D 输入复制 → kernel → D2H 输出复制 → 同步”。开始前同样预热 5 次并同步，结束时所有输出已可供 CPU 使用。范围包含传输和主机提交，不包括内存分配、数据生成或正确性比较。

| 测量 | 输入起点 | 输出终点 | 本次是否包含分配 |
| --- | --- | --- | --- |
| kernel_sequence_mean_ms | 数据已在设备 | 结束 event 完成 | 否 |
| end_to_end_mean_ms | 数据在主机 | 结果回到主机且同步完成 | 否 |

本次日志测得序列平均 0.007299 ms，端到端平均 0.537455 ms。这不是所有 RTX 5060 Ti 的保证，也不说明 CPU 与 GPU 的加速比，因为没有 CPU 性能基线。程序使用普通 std::vector 主机内存；第 7 章再解释页锁定内存。

计时方法的补充参考：[NVIDIA 性能计时说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#timing)。

## 5. 调试工具与检查命令

环境记录中的 GDB 只能证明主机调试器存在。本机未找到 cuda-gdb、Compute Sanitizer、Nsight Systems 或 Nsight Compute，本章没有假定它们可运行。若今后已安装 Compute Sanitizer，可执行：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/all/ch04/benchmark
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/all/ch03/transpose
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/all/ch03/transpose
```

以上命令尚未在本机验证。memcheck 关注内存错误，racecheck 检查共享内存访问风险，synccheck 检查同步使用；它们不是所有逻辑错误的证明器。工具用途见 [Compute Sanitizer 官方说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。在工具下运行获得的时间也不能直接当普通性能数据。

## 6. 常见错误

- 只用 CPU 时钟包围 kernel 启动，不等待执行结束，测到的主要是提交过程。
- 第一次调用只跑一次就给出性能结论。
- GPU 计时里混入 H2D，然后将结果称为“纯 kernel 时间”。
- 在计时循环中打印、分配、核对结果，改变被测工作。
- 启用 fast-math 后仍沿用旧的误差假设。
- 编译通过但测试程序无条件 return 0，使 CTest 无法发现错算。

## 7. 练习与参考答案

1. 如果把 cudaMalloc 放入端到端循环，结果还能叫端到端吗？
   答：可以，但必须重新标明范围包含分配，不能与本章范围不变地比较。
2. 本次 0.537455/0.007299 的比值是 GPU 相比 CPU 的加速比吗？
   答：不是，这是两种测量范围的比值，缺少 CPU 算法计时。
3. 为什么容差比较之外还要检查 isfinite？
   答：NaN 的比较可能不会按正常实数逻辑返回真假；必须显式拒绝非有限值。
4. 如何比较两种 block 大小？
   答：保持输入、构建选项和范围一致，各自预热、重复、验证，记录多次独立实验及波动；不能只挑最快一次。

## 构建、运行与实测输出

完整源码：[benchmark.cu](examples/benchmark.cu)；构建定义：[CMakeLists.txt](examples/CMakeLists.txt)。公共依赖也在本仓库，不需要复制未提供的代码。

在项目根目录执行（首次配置后可重复构建）：

```bash
cd /data2/cuda-guide
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all --target benchmark -j2
./build/all/ch04/benchmark
```

也可单独配置本章：把上面 -S 改为 chapters/ch04-engineering-and-timing/examples，-B 改为 build/ch04-standalone，其他选项不变；程序位于该独立构建目录下。无需第三方图像或数学库。

本次实际输出如下。除计时数据可能随运行波动外，同一固定输入应得到这些结果或在注明容差内一致；最终错误数量应为 0：

```text
affine: n=1048576 max_abs_error=0 mismatches=0 PASS
end_to_end: n=1048576 max_abs_error=0 mismatches=0 PASS
n=1048576 warmup=5 repeats=100
kernel_sequence_mean_ms=0.007299
end_to_end_mean_ms=0.537455 (H2D+kernel+D2H, allocation excluded)
```

更多环境与限制见 [validation.md](results/validation.md)。本机没有 Compute Sanitizer，不能把 CPU 对照通过等同于内存或同步工具检查通过。
