# 第 29 章：Tensor Core、GEMM 与 CUTLASS

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch28-softmax-normalization/README.md)

第 16 章用普通 CUDA 线程实现 GEMM；本章把同一 `C=A×B` 问题送入 FP16 输入、FP32 累加的 WMMA 路径，并在 cuBLAS 开发库可用时用 GEMMEx 对照。主程序的朴素和 WMMA 路径都计算 Bias+ReLU：WMMA 在同一个 kernel 中完成，可选 cuBLAS GEMMEx 后接独立的 Bias+ReLU kernel。可选的 CUTLASS C++ 示例单独演示设备级 GEMM API 与行优先布局，但采用 FP32，不把它的单次时间与半精度路径直接比较。本章主程序在 zyk 使用 nvcc 12.8 独立编译并对朴素与 WMMA 路径进行 GPU/CPU 对照；cuBLAS 开发库与 CUTLASS 头文件未找到，两个可选库路径均未编译或 GPU 实测，见[验证记录](results/validation.md)。

## 1. 从矩阵语义到输入精度

全部矩阵按行优先存储：`A` 为 `M×K`、`B` 为 `K×N`、输出为 `M×N`。两个输入先从确定性 float 值量化为 FP16，CPU 参考**读回量化后的半精度数**，用双精度累加，再加列方向 Bias 和做 ReLU；如果 CPU 参考仍使用量化前 float 输入，误差来自输入表示而不是 GEMM 实现。程序运行 `16×16×16`、`19×23×37`、`64×64×64`，后者与前者需用同一语义逐元素比较。非 16 倍数尺寸补零到 `Mp,Np,Kp`，尾部输出不参与判定；补零与裁切成本应在完整任务计时中另算。

WMMA 用每个 Warp 处理一个 `16×16×16` 乘加 tile，A/B 的行跨度补齐到 16 的倍数，保证 fragment 载入要求。每个 `16×16` 输出 tile 的 FP32 累加结果先写入 32 字节对齐共享内存，Warp 同步后，由 lane 按逻辑行列加 Bias 并做 ReLU，直接写回最终输出。即使只是教学实现，也不能从 WMMA fragment 的内部寄存器排列猜每个 lane 拥有哪些逻辑元素；显式 `store_matrix_sync` 后按行列索引更清楚。普通朴素 half×half→float 路径也在同一程序中，提供第 16 章到 WMMA 的衔接。

cuBLAS 默认采用列优先 GEMM 语义；本章把行优先 `C=A×B` 视为列优先 `C^T=B^T×A^T`，交换 A/B 指针并使用补齐后的 leading dimension。若交换顺序或 `lda/ldb/ldc` 写错，小方阵可能恰好看不出问题，故加入非方形 `19×23×37`。本程序使用 FP16 A/B、FP32 C 和 `CUBLAS_COMPUTE_32F`，随后另启一个 kernel 做 Bias+ReLU。库是否实际选择 Tensor Core 路径须看该设备与 cuBLAS 的算法选择或 profiler，不能仅凭 API 名称断言。性能对比应使用相同输入、输出和包含的步骤，分别报告只计 kernel 与包含量化、padding、传输、后处理的完整任务。

## 2. CUTLASS 可选路径

[cutlass_gemm.cu](examples/cutlass_gemm.cu) 使用 `cutlass::gemm::device::Gemm` 做 32×32×32 的**FP32 行优先** GEMM，CPU 双精度参考逐元素比较。它展示 CUTLASS 的类型、布局、问题尺寸、TensorRef/leading dimension 和 epilogue 参数 `alpha=1,beta=0`。当前采用 CUTLASS 2.x 风格的兼容 API，CUTLASS 3.x 另有 `GemmUniversalAdapter` 与 CuTe 层次；必须在实际拿到的 CUTLASS 版本上构建并记录版本。可选示例不带 Bias/ReLU、不保证启用 Tensor Core，因此它是 API 入门，不参与半精度融合性能排名。后续若将 CUTLASS 改成 FP16 TensorOp 与融合 epilogue，需重新匹配精度、布局、输出和 CPU 参考。

NVIDIA [CUDA 12.8 编程指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)说明 WMMA fragment 的协作与对齐限制；[CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)与 [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)展示不同抽象层。CUTLASS 官方兼容性表列出 RTX 50 系列 SM120 的最低 Toolkit 要求为 CUDA 12.8；仍需核对具体 CUTLASS 版本、编译器和实际 GPU。[cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)列出 GEMMEx 和 cuBLASLt epilogue；本章主路径没有直接使用 cuBLASLt 的融合 Bias+ReLU。

## 3. 构建、运行与验证

在 zyk 仓库根目录执行基础路径：

```bash
cmake -S chapters/ch29-tensor-core-cutlass/examples -B build/ch29 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch29 -j
ctest --test-dir build/ch29 --output-on-failure
./build/ch29/tensor_core_gemm
```

若已有合适版本 CUTLASS **源码目录**，可给出其绝对路径，CMake 才会添加可选 target/test；本例不自动下载或安装：

```bash
cmake -S chapters/ch29-tensor-core-cutlass/examples -B build/ch29-cutlass \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_ROOT=/path/to/existing/cutlass
cmake --build build/ch29-cutlass -j
ctest --test-dir build/ch29-cutlass --output-on-failure
```

主程序在计算能力低于 7.0 的设备上以 77 退出，CTest 标记跳过；这不等于 WMMA 验证通过。实际输出须显示三种尺寸的朴素、WMMA 路径均 `bad=0 PASS`；若 cuBLAS 可用，它也须逐项通过，否则打印 `SKIP`；误差规则为绝对 `0.02 + 0.002*|CPU参考值|`，需记录最大误差。`kernel_ms` 为单次 Event 时间，cuBLAS 路径包含后处理 kernel，WMMA 路径在一个 kernel 中后处理；它**不包含** Host 半精度量化、padding、传输、分配与 CPU 验证。大尺寸性能结论要预热、重复、区分数学模式，并用 profiler 核查设备实际指令和资源。CUTLASS 未配置时，CTest 只有主测试，不可写“CUTLASS 测试通过”。本机 cuBLAS 开发库缺失时，GEMMEx 也只报告 `SKIP`。

## 4. 常见错误

- 用量化前 float CPU 结果评判 FP16 输入的 GEMM，误把输入精度差异归为 Tensor Core 错误。
- WMMA 的 leading dimension 不是对齐倍数，或 Warp 中部分 lane 没参与 fragment 操作。
- 从 fragment 的寄存器布局猜逻辑 `(row,col)`，把 Bias 加到错误列。
- 非整齐尺寸只补输出，不补 K/A/B，最后一个 MMA tile 读取越界。
- cuBLAS 行/列布局转换时没有交换 A/B 或弄错 leading dimension。
- cuBLAS 结果另起后处理 kernel，却只计 GEMM 时间与 WMMA 融合时间比较。
- CUTLASS 头文件缺失时跳过可选测试，却把整章标记为 CUTLASS 已实测。
- 在不同输入 dtype、累加精度或 epilogue 上比较加速比。

## 5. 练习与参考答案

1. `M=19,N=23,K=37` 补齐 16 后是多少？答：`Mp=32,Np=32,Kp=48`；输入补零，验证只看原始 19×23 区域。
2. 为什么 WMMA 的 FP32 累加仍不等于“完整 FP32 GEMM”？答：乘法输入先量化为 FP16，丢失部分有效位；FP32 只用于累加。
3. 行优先 C=A×B 如何映射给列优先 cuBLAS？答：解释为 `C^T=B^T×A^T`，交换 B/A 指针，并把列优先维度设为 `N,M,K`。
4. 本章 WMMA 与 cuBLAS 时间可否从单次输出推断稳定加速？答：不能；须预热、重复，核对后处理、padding/量化/传输是否同口径，并确认实际算法。
5. CUTLASS 可选示例通过是否证明 Tensor Core 路径通过？答：不能；它使用 FP32 默认 GEMM 类型，没有声明 TensorOp，需单独核查所选内核和指令。

[下一章](../ch30-attention-inference/README.md) 将在相同“数值语义先固定”的原则下实现显式 Attention 和分块前向。
