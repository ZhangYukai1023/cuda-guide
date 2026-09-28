# 第 31 章：线性代数与稀疏求解

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch30-attention-inference/README.md)

本章把“矩阵乘向量”推进到“求 Ax=b”。示例 [linear_solvers.cu](examples/linear_solvers.cu) 依次运行非对称稠密矩阵的 cuBLAS GEMV、CSR 矩阵的 cuSPARSE SpMV、cuSOLVER LU 直接求解，以及在满足对称正定条件下的共轭梯度（CG）迭代。输入以已知解构造，GPU 输出同时检查**解误差**和代回 `Ax-b` 的**相对残差**。zyk 当前 Toolkit 缺少 cuBLAS、cuSPARSE、cuSOLVER 开发文件，库目标没有构建或实测。本章另提供 [reference_solvers.cu](examples/reference_solvers.cu)，以普通 CUDA kernel 完成稠密 GEMV、CSR SpMV，并用 GPU SpMV + CPU 标量递推完成教学用混合式 CG；这条路径已在 zyk 用 nvcc 12.8 构建并通过 CPU 对照，见[验证记录](results/validation.md)。混合式 CG 不是 cuSPARSE/cuBLAS 全 GPU 求解器。

## 1. 稠密与稀疏的布局

稠密矩阵按行优先 `A[row*n+col]` 保存；CSR 将每行非零元素依次放入 `values`，相应列号放入 `columns`，`offsets[i]..offsets[i+1]` 是第 `i` 行区间。COO 则给每个非零元素都存 `(row,col,value)`，构建方便，但同一行元素可能分散；不同稀疏格式的访存和工作量分配依赖矩阵行长度分布，不能只按 `nnz` 比较。代码先用三对角矩阵保证每行最多三个非零值，`n=32`、`nnz=94`，适合手工核对；这不代表任意不规则稀疏矩阵的最佳 SpMV 选择。

稠密入门例子用非对称 3×3：

```text
A = [1 2 3; 0 4 5; 6 0 7], x = [1,2,3]
Ax = [14,23,27]
```

cuBLAS 的 GEMV 接口按列优先解释矩阵；行优先内存相当于列优先 `Aᵀ`，因此调用 `CUBLAS_OP_T` 还原逻辑 `A*x`。非对称输入能发现错用 `CUBLAS_OP_N`，而对称矩阵可能掩盖这一错误。

## 2. 已知解、LU 与收敛判据

求解问题取 `A` 对角为 4、紧邻上下对角为 -1，其余为 0。它对称且严格对角占优，适合作为正定 CG 教学输入。先令 `x*_i=1+0.01i`，在 CPU 按稠密矩阵计算 `b=A*x*`，这样知道期望解。GPU CSR SpMV 先验证 `A*x*≈b`。cuSPARSE 的 Generic API 需要 CSR 矩阵、稠密向量描述符和 `cusparseSpMV_bufferSize` 返回的设备工作区；描述符引用设备缓冲，缓冲必须在调用完成前保持有效。代码复用工作区和描述符做后续 CG SpMV。

cuSOLVER `getrf`/`getrs` 用 LU 分解和带主元的回代解同一个方程；`getrf` 改写 A，故用单独设备拷贝。要检查 `devInfo`：0 表示本次分解/求解未报告失败，非零不能继续把输出当有效解。例子的 A 是对称矩阵，行优先与列优先转置在数值上相同；一般非对称问题要按 cuSOLVER 列优先布局显式转换，不可照搬这段数据准备代码。

CG 从 `x0=0` 开始，`r0=b`、`p0=r0`。每轮用 cuSPARSE 计算 `Ap`，cuBLAS 点积取得 `pᵀAp` 与 `rᵀr`，用 `axpy/scal` 更新 `x,r,p`。若 `pᵀAp<=0` 或非有限，立即报告不满足预期或数值故障；若 `sqrt(rᵀr)/sqrt(bᵀb)<1e-6`，停止；最多 `4n` 轮，超限算失败。这个递推残差是停止依据，程序还把最后解下载后用 CPU **重新计算真残差** `||Ax-b||₂/||b||₂`，并检查 `max_i|x_i-x*_i|`。两项阈值分别为 `1e-5`、`1e-4`。CG 只应对满足前提的对称正定 A 使用；非对称、不定或病态矩阵需要别的求解器、预条件、精度与失败处理策略。

## 3. 构建与运行

有三种库开发文件时，CMake 才构建 cuBLAS/cuSPARSE/cuSOLVER 库示例；普通 CUDA 教学路径始终构建。在仓库根目录运行：

```bash
cmake -S chapters/ch31-linear-sparse-solvers/examples -B build/ch31 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch31 -j
ctest --test-dir build/ch31 --output-on-failure
./build/ch31/reference_solvers
# 仅当 CMake 配置显示三种库目标均存在时，再运行：
./build/ch31/linear_solvers
```

普通 CUDA 路径应打印稠密 GEMV、CSR SpMV、混合式 CG 的 `PASS`，CG 还打印迭代轮数、真残差和解误差；末尾为 `chapter 31 reference solvers: PASS`。若三种库齐备，库目标还应依次打印 cuBLAS GEMV、cuSPARSE SpMV、cuSOLVER LU、库支持的 CG 的 `PASS`；当前 zyk 缺开发库，这部分不能标记通过。真实运行后须保存 GPU、Toolkit、库版本、退出码和原始数值。不能从这个 32 阶例子得出稀疏库比稠密库更快的结论：库句柄创建、描述符、工作区、主机点积结果传回和小矩阵启动开销都很显著。性能实验应扩大具有代表性的行长度分布，分开测预处理与复用阶段，并比较同等精度/停止阈值下的时间。

API 约定可核对 NVIDIA 的 [cuSPARSE 12.8 文档](https://docs.nvidia.com/cuda/archive/12.8.1/cusparse/index.html)、[cuSOLVER 12.8 文档](https://docs.nvidia.com/cuda/archive/12.8.2/cusolver/index.html)和 [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)。具体 Toolkit 内的头文件与库 ABI 仍以 zyk 实际版本为准。

## 4. 常见错误

- CSR `offsets` 长度不是 `n+1`，最后一项不等于 `nnz`，或列号未在 `[0,n)`。
- 用 32 位索引描述符却传 64 位索引数组，或在工作区/描述符仍被使用时释放设备内存。
- 行优先稠密矩阵不转换/不转置就喂给列优先 cuBLAS/cuSOLVER。
- LU `devInfo` 非零仍报告“求解成功”。
- 对非对称、不定矩阵使用 CG，并把不收敛归为 GPU 错误。
- 只比较 `x` 与已知解，未算 `Ax-b`；或只看递推残差，未计算真残差。
- 固定最大迭代轮数内没收敛却仍按最后一个向量算通过。
- 对非常小的矩阵拿单次 kernel 时间做稠密/稀疏算法排名。
- 把混合式 CG 的 CPU 标量递推误写成全 GPU 求解；每轮的 H2D/D2H 往返是教学简化。

## 5. 练习与参考答案

1. CSR 的 `offsets=[0,2,3,5]` 代表多少行、多少非零？答：3 行、5 个非零；各行分别有 2、1、2 个。
2. 非对称 3×3 手算例的第二个输出是多少？答：`0*1+4*2+5*3=23`。
3. 三对角 32 阶矩阵有多少非零？答：两端各 2 个、内部 30 行各 3 个，总数 `2*2+30*3=94`。
4. CG 的递推残差低于阈值就无需再算 `Ax-b` 吗？答：仍应重新计算真残差，有限精度递推可能偏离实际残差。
5. `pᵀAp<=0` 时应如何处理？答：停止并报告前提或数值问题；不能继续用该分母计算步长。

下一章将用 cuFFT 处理频域变换，继续明确归一化、填充和边界语义。
