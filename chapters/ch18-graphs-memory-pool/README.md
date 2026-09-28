# 第 18 章：重复任务、CUDA Graphs 与内存池

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch17-stream-pipeline/README.md)

第 17 章把分批任务安排进 Stream。本章研究**相同两段小 kernel 重复执行**的提交成本。示例 [repeated_work.cu](examples/repeated_work.cu) 对长度 1003 的整数数组先乘 2、再加 1，以 CPU 结果逐项精确验证；比较普通提交 100 次与同一 CUDA Graph 启动 100 次，并单独记录 Graph 捕获/实例化准备时间。另用一个独立小实验演示流顺序分配与释放。已在 zyk 编译并通过两段 kernel、Graph 重放与流顺序分配的 GPU/CPU 对照。计时发生在其他 GPU 任务运行期间，性能结论待空闲复测。

## 1. 普通提交与 Graph 复用

直接版在同一条非默认 Stream 中重复提交：

```text
for 100 次: double_values → add_one
最后同步 Stream
```

Graph 版先在 Stream capture 模式提交一次相同的两段 kernel；捕获期间操作被加入图结构，不按普通方式运行。结束捕获后实例化为可执行图，首次启动并同步作为预热，然后每轮连续启动这个图 100 次，最后同步。输入、输出和临时设备缓冲都在实验前分配并保持有效；Graph 节点记录的指针不会因为下一轮启动而自动指向新数组。本例每次处理相同输入与地址，适合复用同一个图。若尺寸、地址或节点结构变化，应按 API 支持范围更新节点/图，或者重新捕获和实例化；不能修改已释放缓冲区后沿用旧图。[CUDA Graphs 官方文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)说明捕获、实例化、执行与更新限制。

两个版本每轮都含 100 次“两段 kernel”工作，输出应全等于 `2*input[i]+1`。打印三列：直接版 100 对 kernel 的墙钟中位数、Graph 100 次启动的墙钟中位数、一次捕获与实例化的准备时间。每个稳态口径取 5 轮样本，每轮末同步；输入传输、分配与最终验证在这两个稳态计时之外。首次 Graph 启动也在稳态计时之外，避免把首次准备成本混入重复执行的中位数。

若 Graph 稳态更短，也要看任务实际重复多少次才能抵消准备成本。不能把两个中位数直接当成任意业务场景的端到端收益；真实工作流还包含数据更新、传输、可能的图更新和资源维护。若两者接近，先看原始样本波动与时间线，不强行宣称加速。

## 2. 流顺序分配的生命周期

源码先查询设备是否支持 `cudaDevAttrMemoryPoolsSupported`。支持时，在同一条 Stream 中依次提交 `cudaMallocAsync(temporary)`、两段 kernel、`cudaFreeAsync(temporary)`，最后同步并与 CPU 对照。分配的指针在同一 Stream 的顺序语义下，先于 kernel 使用生效；释放排在最后一次使用之后。它默认使用该设备的当前内存池，没有在本章另建或调整池属性。不支持时打印明确 SKIP，普通提交与 Graph 部分仍可验证。

这个小实验只验证基本使用与生命周期，不证明内存池一定更快。工作区跨调用复用可减少反复申请的开销，但也要保证容量、并发流依赖与所有权；跨 Stream 访问异步分配的指针时，须用 Event/等待显式建立“已分配、已完成使用、才可释放”的顺序。[流顺序分配官方文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)解释了默认池、复用策略与跨 Stream 依赖。

## 3. 构建与验证

在仓库根目录：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target repeated_work -j2
./build/outline/book_ch18/repeated_work
ctest --test-dir build/outline -R '^ch18_repeated_work$' --output-on-failure
```

单章独立源目录为 `chapters/ch18-graphs-memory-pool/examples`。本章使用与 zyk 既有环境对应的 CUDA 12.8 风格 `cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)`；若实际工具包 API 不同，先核对安装版本并做兼容修复，不更换驱动或 CUDA 安装。[验证记录](results/validation.md)保留实际输出与负载限制；本机已通过上述 CUDA 12.8 API 构建。

## 4. 常见错误

- 把捕获期间的 kernel 当作已经执行，直接读取输出。
- 图节点引用的设备缓冲被释放或重新分配，却继续启动旧图。
- 只报告 Graph 稳态时间，隐去捕获和实例化准备成本。
- 两个版本的重复次数或同步位置不同，却用时间做一对一比较。
- `cudaFreeAsync` 排在最后一次使用之前，或另一个 Stream 未建立依赖便访问异步分配指针。
- 把默认内存池复用能力与“每次分配零成本”等同；仍需实际测量。
- 把 Graph 适合固定任务拓扑外推为“所有动态任务都容易捕获”。

## 5. 练习与参考答案

1. 输入 `[0,2,-3]` 经过两段 kernel 的结果？答：`[1,5,-5]`。
2. 为什么 Graph 准备时间与稳态时间要分开？答：捕获和实例化只在准备时发生，复用多少次决定这笔成本如何摊销。
3. 若图中的输入指针指向已释放缓冲会怎样？答：图仍保存原指针参数，继续启动会产生无效访问风险；必须确保缓冲在所有执行结束前有效，或更新图节点参数。
4. 一个异步分配指针要交给另一条 Stream 使用，应建立什么？答：让使用 Stream 等待分配 Stream 上表示“分配已生效”的 Event；释放前再等待所有使用结束。
5. 设备不支持内存池时怎样报告本章？答：Graph 与直接提交可分别验证；流顺序分配标记 SKIP 并写明设备能力，不伪称该部分通过。

[下一章：组织 CUDA C++ 工程](../ch19-cuda-project/README.md)把示例整理成可复用模块，明确缓冲所有权和错误传播。
