# 第 17 章：Stream、Event 与处理流水线

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch16-matmul-case/README.md)

前几章的程序多是“一次输入、一次计算”。本章把 10 批数组处理串成流水线：每批各有 65536 个 float，结果仍要回到主机。示例 [stream_pipeline.cu](examples/stream_pipeline.cu) 以相同输入依次运行 1、2、4 个设备缓冲槽；每个槽有自己的非默认 Stream、输入输出设备缓冲区和完成 Event。结果逐项与 CPU 比较。三种槽数已在 zyk 构建并通过 GPU/CPU 对照；Nsight 不可用，实际传输与计算是否重叠尚未得到时间线证据。

## 1. 先理解一个槽的顺序

一个批次在同一条 Stream 中依次排队：

```text
H2D(A) → H2D(B) → add kernel → D2H(output) → record(done Event)
```

同一 Stream 保证这些操作按提交顺序执行，`cudaEventRecord(done)` 到达时说明该槽此前排队的工作已完成。1 槽模式在提交下一批前等 `done`，所以是十批串行处理。2 槽模式在第 0、1 批分别使用两个槽，第 2 批想重用第 0 槽时，先等它上一次的 Event。4 槽模式同理。这种等待保障设备缓冲不会在旧批次尚未完成时被新批次覆盖。

Host 为十批分别保留不同的页锁定输入输出区域，避免 D2H 尚未完成就重用 host 输出。使用 `cudaHostAlloc` 和 `cudaMemcpyAsync` 提供异步传输的必要条件，但 API 名字含 “Async” **不保证**传输与 kernel 实际重叠；设备拷贝引擎、依赖、资源竞争及传输大小都会影响时间线。NVIDIA 的 [CUDA 异步执行文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)与[最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)解释了 Stream、Event、Pinned Memory 与重叠的条件。

## 2. 用 Event 表达跨 Stream 依赖

十批提交完后，合并 Stream 对每个槽最后一次 `done` Event 调用 `cudaStreamWaitEvent`，随后一个小 kernel 读取各槽**最后一批**输出的首项并相加。合并操作只会在所有相应 Event 完成后执行；主机最后同步合并 Stream，才读取合并值并验证十批完整 host 输出。由于每个槽的事件排在同一 Stream 所有旧批次之后，等待最终事件也意味着该槽之前的批次已结束。

源码输入使第 `batch` 批首项结果为 `batch+1`。1 槽最终合并值预期 10；2 槽的最后批次是 8、9，预期 `9+10=19`；4 槽最后批次为 8、9、6、7，预期 `9+10+7+8=34`。这些是代码推导出的预期值，zyk 实测也分别得到 10、19、34。合并 kernel 仅演示跨 Stream 依赖，没有对十批全部元素求总和；十批全部元素由 host CPU 对照检查。

Event 在复用槽之前先等待完成，且合并 Stream 在下一轮重新记录这些 Event 之前完成同步。这是事件生命周期约定的一部分。若拿尚未记录的 Event 建立依赖，等待可能立刻成功，不能保护数据；若在旧任务仍用缓冲区时就重新排队写入，将产生数据竞争。

## 3. 测量结果应该怎样读

示例先做一次预热，再对每种槽数重复 5 次，打印主机墙钟中位数。计时包括十批传输、kernel、需要的 Event 等待、合并 kernel 和最后的同步；不包括页锁定与设备缓冲分配、输入填充和最终 CPU 正确性检查。三个模式的任务边界一致，但槽数改变了同时持有的设备缓冲数量。若实际应用每次请求都重新分配，必须把这部分成本加回完整任务口径。

2 或 4 个 Stream 不意味着必然比 1 个快：可能没有可用的拷贝/计算重叠，数据量太小，主机提交成为瓶颈，或共享设备带宽已满。应先通过本章 CPU 对照，再用第 13 章 Nsight Systems 检查 H2D、kernel、D2H 是否重叠，最后用多次未插桩墙钟样本判断收益。默认 Stream 的语义可能给其它流带来额外排序；本例为槽和合并操作显式使用非默认 Stream 与 Event 依赖。

## 4. 构建与验证

在仓库根目录运行；最后的 `nsys` 命令需工具可用：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target stream_pipeline -j2
./build/outline/book_ch17/stream_pipeline
ctest --test-dir build/outline -R '^ch17_stream_pipeline$' --output-on-failure
nsys profile --trace=cuda -o build/nsys-ch17 \
  ./build/outline/book_ch17/stream_pipeline
```

单章独立配置源目录为 `chapters/ch17-stream-pipeline/examples`。若页锁定分配失败，应记录环境限制；不要未经标注就改为普通 host 内存并声称具有相同重叠性质。实际输出与限制见[验证记录](results/validation.md)。当前 GPU 有其他任务，墙钟数值仅作原始记录，不能据此断言发生了传输与计算重叠。

## 5. 常见错误

- 第 2 批直接覆盖第 0 批仍在使用的设备槽，输出随机错误。
- 过早改写页锁定 host 输入或读取 D2H 目标，未等待对应 Event 完成。
- `cudaStreamWaitEvent` 对一个从未记录过的 Event 建立依赖，误以为保护了缓冲区。
- 忘记指定 Stream，让某一步落入默认 Stream，打乱预期并发关系。
- 只看到 `cudaMemcpyAsync` 返回很快，就断言 GPU 与拷贝已重叠。
- 多槽计时只算 kernel，串行模式计入传输，导致口径不一致。

## 6. 练习与参考答案

1. 10 批、2 槽时，第 8 批会用哪个槽？答：槽 `8%2=0`，必须等该槽上一次第 6 批完成再重用。
2. 为什么 Event 记录在 D2H 后面？答：这样 Event 完成时，该批次结果已复制到对应 host 区域，槽和结果都可安全使用。
3. 4 槽模式下最后一次合并首项的预期值？答：`9+10+7+8=34`。
4. 若 Nsight 时间线显示两个 Stream 的 kernel 从不同时执行，是否代表代码错？答：不一定；可能资源、依赖或设备能力限制。先看正确性，再看时间线和设备属性。
5. 若要把这个接口做成异步返回，调用者需要拿到什么？答：至少要有结果缓冲所有权和完成事件/句柄的清楚约定，并在读结果或复用输入输出前等待完成。

[下一章：CUDA Graphs 与内存池](../ch18-graphs-memory-pool/README.md)讨论重复提交同一批小任务时的准备成本和资源生命周期。
