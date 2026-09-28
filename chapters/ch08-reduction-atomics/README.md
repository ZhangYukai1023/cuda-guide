# 第 8 章：归约、原子操作和基础并行模式

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch07-shared-memory/README.md)

前一章得到了每个 Block 的部分和。本章把部分和汇总成完整结果，再求“最大值及其位置”，最后用原子操作统计 16 个整数桶。完整源码见 [reduction_atomic.cu](examples/reduction_atomic.cu)。本章已在 zyk 用 nvcc 编译，并在 RTX 5060 Ti 上运行；两阶段归约、最大值与位置、16 桶直方图均通过 CPU 对照，详情见[验证记录](results/validation.md)。

## 1. 手算求和树

数组 `[3,1,4,2]` 的总和为 10。可以先两两合并：`3+1=4`、`4+2=6`，再算 `4+6=10`。Block 内 128 个线程采用同样的折半思路：每线程先写一个输入值到共享内存，不足 128 项的尾块写 0；每轮只让前一半线程把后一半的值加进来，之后**所有线程**到达 `__syncthreads()`。

```cpp
values[t] = i < n ? input[i] : 0;
__syncthreads();
for (int offset = 64; offset > 0; offset /= 2) {
    if (t < offset) values[t] += values[t + offset];
    __syncthreads();
}
if (t == 0) partial[blockIdx.x] = values[0];
```

一个 Block 只能为自己的最多 128 项得到部分和。长度 1003 需要 8 个 Block，第一轮产生 8 个部分和；第二次启动同一个 kernel，用一个 Block 把这 8 项合并为最终总和。两次 kernel 在同一默认流中按提交顺序执行，不依赖第一轮中各 Block 的先后完成次序。本章输入规模保证部分和不超过 128 项；处理更大的数据须重复分层或选择成熟归约库。

## 2. 最大值与位置：先定义并列规则

若输入为 `[7,3,7]`，最大值都是 7。只说“找最大值”不能决定应返回哪个位置；本章规定并列时取最小下标，因此答案是 `(value=7,index=0)`。每个线程把值和原始下标组成一个 `Candidate`，归约时用同一比较规则选择较优候选。越界线程使用 `(INT_MIN,INT_MAX)` 作为空候选，避免尾块读到未初始化数据。

第一轮每个 Block 产生一个候选，第二轮合并所有 Block 的候选。CPU 用同一并列规则计算参考答案，并对 GPU 的值与下标都做精确比较。示例输入 `i%11-5` 在长度 7 时最大值是 1、位置 6；长度较大时重复出现最大值 5，答案应选择最早出现的下标 10。

## 3. 直方图：竞争写入为什么需要原子操作

输入 `[0,1,0,2,1,0]`、桶编号 0—2 的计数应为 `[3,2,1]`。若多个线程同时做普通的 `bins[value]++`，读取、加一、写回可能互相覆盖，造成丢计数。`atomicAdd` 将同一地址的更新作为原子操作处理，但更新顺序本身没有规定。

本章先让一个 Block 在共享内存中建 16 个局部桶：前 16 个线程清零，整块同步；有效线程对 `local[input[i]]` 执行 `atomicAdd`；整块再同步；前 16 个线程将本 Block 的局部计数原子合并到全局桶。Host 在启动前将全局桶清零。所有线程都参加两次屏障，尾块越界线程只跳过计数。

循环输入 0—15 共 64 项时，每桶预期 4；1003 项全为 0 时，桶 0 预期 1003、其他桶为 0。后者让竞争更集中，但本章没有计时，不能由此给出性能结论。NVIDIA 的 [CUDA 编程指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html) 说明了 `atomicAdd` 及共享内存直方图的基本用法。

## 4. 认识 Scan、Gather 与 Scatter

这些模式在后续算法中反复出现，这里先用小数组识别语义：

| 模式 | 输入例子 | 输出或动作 |
| --- | --- | --- |
| Reduce | `[2,1,3]` | 求和得到单个值 6 |
| Inclusive Scan | `[2,1,3]` | 前缀和 `[2,3,6]` |
| Gather | `in=[10,20,30]`、`index=[2,0]` | 按下标读取，得到 `[30,10]` |
| Scatter | `in=[10,20]`、`index=[2,0]` | 写入目标 2 和 0；若目标重复，需定义冲突处理 |
| Histogram | `[0,1,0,2]` | 桶 0、1、2 的计数为 `[2,1,1]` |

本章源码实现 Reduce、最大值位置及 Histogram。Scan 与 Gather/Scatter 保留为练习和后续章节主题，不声称已实测代码。

## 5. 构建与验证

服务器恢复响应后，在项目根目录运行：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target reduction_atomic -j2
./build/outline/book_ch08/reduction_atomic
ctest --test-dir build/outline -R '^ch08_reduction_atomic$' --output-on-failure
```

也可用 `-S chapters/ch08-reduction-atomics/examples -B build/ch08-reduction-standalone` 独立配置。单章与根工程 CTest 已运行，实际输出和 CPU 对照见[验证记录](results/validation.md)。Compute Sanitizer 当前不可用，故障模式未做工具实测。

## 6. 常见故障

- 只运行一次块内归约便把第 0 个 Block 的部分和当作全局总和；长度 1003 会遗漏其他 Block。
- 把尾块中越界线程提前 `return`，导致部分线程跳过 `__syncthreads()`；应写入中性值 0 或空候选并共同参加屏障。
- 未规定并列最大值的下标选择规则，使 CPU/GPU 对照随归约次序而变。
- 全局直方图未清零，或把 `atomicAdd` 改为普通 `++`，导致累积旧值或线程间更新丢失。
- 因为整数结果通过就认定浮点归约位级相同；浮点加法顺序不同可能产生舍入差异，第 9 章专门讨论容差。

## 7. 练习与参考答案

1. `[1,2,3,4,5]` 求和是多少？若每块最多 4 项，第一轮部分和是多少？答：总和 15，第一轮为 `[10,5]`，第二轮再合并。
2. `[4,7,7,2]` 按本章规则的最大值与下标？答：`(7,1)`。
3. 16 桶直方图输入 `[0,0,2,15,2]`，非零桶的计数？答：桶 0 为 2、桶 2 为 2、桶 15 为 1。
4. `[2,1,3]` 的 exclusive scan 是什么？答：`[0,2,3]`，每项不包含当前位置的值。
5. Gather 的索引若越界应怎么办？答：先定义接口约定并检查或拒绝非法索引，不能直接访问输入数组之外的位置。

[下一章](../ch09-correctness-validation/README.md)从 CPU 参考、边界输入、浮点误差和可复现随机测试建立更完整的正确性验证方法。
