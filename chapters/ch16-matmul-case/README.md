# 第 16 章：矩阵乘优化贯穿案例

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch15-execution-resources/README.md)

前几章分别学习正确性、分块、资源与计时；本章把它们用于同一个矩阵乘。示例 [matmul_compare.cu](examples/matmul_compare.cu) 对比 CPU double 参考、朴素 GPU、`16×16` 共享内存分块、每线程计算相邻两列的寄存器分块，以及 cuBLAS SGEMM。测试小型、非整齐非方阵和较大整齐矩阵。三个手写版本已在 zyk 编译并通过 GPU/CPU 对照；cuBLAS 开发库缺失，SGEMM 分支未编译或运行。当前 GPU 有并发任务，性能仍待空闲复测。

## 1. 从手算到索引

令 A 为 `M×K`、B 为 `K×N`，结果 C 为 `M×N`。每个输出元素是：

```text
C[row,col] = sum(A[row,t] * B[t,col]), t = 0..K-1
```

例如 `A=[[1,2,3],[4,5,6]]`、`B=[[7,8],[9,10],[11,12]]`，则 `C=[[58,64],[139,154]]`。这个手算例子解释公式；源码使用另一个由下标生成的可重现数据集，不应把这四个数当成源码实测输出。源码中 A、B、C 都是**行主序**，索引分别为 `A[row*K+t]`、`B[t*N+col]`、`C[row*N+col]`。

CPU 用 double 积累 float 输入的乘积，作为更高精度参考。每个 GPU 版本允许 `abs_error <= 1e-4 + 1e-4*abs(reference)`；该容差只适用于本章输入范围和长度，不能替代业务精度要求。输出打印最大绝对误差与不匹配项数。若非整齐形状不通过，先查边界和布局，再查浮点顺序差异。

## 2. 四个设备版本的递进

| 版本 | 一个线程负责什么 | 数据重用与边界 |
| --- | --- | --- |
| `naive` | 一个 C 元素 | 每个线程自行循环 K；只对输出坐标做一次守卫 |
| `shared_tile` | 一个 C 元素 | A、B 的 `16×16` tile 载入共享内存，每轮 K 维不足处填 0 |
| `two_register_outputs` | 同一行的相邻两个 C 元素 | 一份 A tile 供两个累加器使用；B tile 覆盖 32 列，尾列分别守卫 |
| `cublas_sgemm` | 由库安排 | 依赖可用时用成熟 GEMM 作正确性与性能对照，注意布局映射；本机缺开发库，当前 SKIP |

两个共享内存版本都让所有线程到达两次 `__syncthreads()`。非整齐 `M=65,N=37,K=19` 会同时考验行、列与 K 维尾块，越界加载以 0 填充，输出只写有效坐标。`M=N=K=256` 适合观察数据重用是否有收益，但性能要来自实际设备测量。每线程双输出只是最小寄存器分块示例，不代表最优 GEMM；更复杂的向量化、异步拷贝、Tensor Core 与 CUTLASS 放在后续章节。

## 3. 行主序数据怎样交给 cuBLAS

经典 `cublasSgemm` 默认按列主序解释矩阵。若把行主序 A (`M×K`) 的同一连续内存视为列主序 Aᵀ (`K×M`)，B 类似视为 Bᵀ (`N×K`)，那么行主序 C 的内存等于列主序 Cᵀ：

```text
Cᵀ = Bᵀ × Aᵀ
```

因此源码交换传入的 B 与 A，调用 `cublasSgemm(..., m=N, n=M, k=K, B, ldb=N, A, lda=K, C, ldc=N)`，得到与其它版本同一块行主序 C。这里的参数名称是 cuBLAS 接口中的位置，不等同源码变量的数学角色。若改成显式转置或者改用支持行主序布局的其它接口，必须重新写清数据流和时间边界。[cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)说明 GEMM 尺寸与 leading dimension 规则。

## 4. 先查正确性，再记录性能

形状依次为 `(M,N,K)=(2,2,3)`、`(65,37,19)`、`(256,256,256)`。每个版本先预热两次，CUDA Event 采 10 次 kernel 或 SGEMM 调用的设备时间，取中位数；输入传输、分配、CPU 参考和输出回传在计时外，但输出验证必须通过。对一个真实调用者，若矩阵在 CPU 而结果需回 CPU，应另按第 12 章方法测端到端时间。小矩阵可能由调用开销主导；单次 kernel Event 的微小差别不够作性能结论。

优化分析还要看实际访存、共享内存、寄存器和占用。分块带来复用，也带来同步与资源成本；双输出可能增加每线程寄存器。不能只凭源码结构给出“必然更快”的排名。NVIDIA 的 [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)可用于解释矩阵乘数据重用，本机的库分支因缺开发库尚未测量；手写版本的设备计时受并发负载影响。

## 5. 构建与验证

在仓库根目录：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/outline --target matmul_compare -j2
./build/outline/book_ch16/matmul_compare
ctest --test-dir build/outline -R '^ch16_matmul_compare$' --output-on-failure
```

独立构建路径为 `chapters/ch16-matmul-case/examples`。cuBLAS 开发库可用时 CMake 自动启用 SGEMM 对照；缺失时仍可构建三个手写版本，并将 SGEMM 明确输出为 `SKIP`。结果见[验证记录](results/validation.md)与[完整运行输出](results/run-2026-09-28.txt)。

## 6. 常见错误

- A、B 的 K 维不一致，或者把行主序与列主序混用，结果数值错误。
- 分块版本只检查 C 的边界，不检查 A/B 尾块加载，读到数组之外。
- 尾块线程在屏障前退出，导致块内同步参与不一致。
- `beta` 不为 0 却没有初始化 C；本例设 `beta=0`。
- cuBLAS 的 `lda/ldb/ldc` 按矩阵总元素数填写，而非按其列主序解释的领先维度填写。
- 对大矩阵只测 kernel，对小矩阵测端到端，然后把数字放在同一列比较。
- 为追求低误差把 CPU 参考也改成与某个 GPU kernel 相同的累加顺序，失去独立对照。

## 7. 练习与参考答案

1. 上述 2×3 与 3×2 的手算例子中，`C[1,0]` 是多少？答：`4*7+5*9+6*11=139`。
2. `M=65,N=37,K=19` 用 `16×16` 输出 tile，需要多少 Block？答：行方向 `ceil(65/16)=5`、列方向 `ceil(37/16)=3`，共 15 个；K 方向循环两轮。
3. 双输出版本的一个 Block 最多计算多少 C 元素？答：16 行 × 32 列 = 512 项，由 256 个线程各计算最多两项。
4. 为什么 cuBLAS 调用时传 B 再传 A？答：原始行主序内存被解释为各自转置的列主序矩阵，利用 `Cᵀ=BᵀAᵀ` 保持输出内存是原始行主序 C。
5. 若 cuBLAS kernel 时间最短，能否马上断言整个应用也最快？答：不能；需计入布局转换、传输、分配、调用频率及其它阶段。

[下一章：Stream、Event 与流水线](../ch17-stream-pipeline/README.md)把单次计算组织成重复处理流水线。
