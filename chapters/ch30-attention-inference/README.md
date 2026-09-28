# 第 30 章：Attention 与推理专题

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch29-tensor-core-cutlass/README.md)

本章只处理单头、单 batch 的 **float32 前向 Attention**：`Q,K` 为 `S×16`，`V` 与输出为 `S×16`，不含 dropout。示例 [attention_forward.cu](examples/attention_forward.cu) 先显式生成 `S×S` 分数、逐行 Softmax，再乘 V；随后用 16-key 逻辑 tile 和在线 Softmax 在每个 query 内更新归一化状态，不保留整张分数/概率矩阵。CPU 双精度参考独立计算，含因果与非因果路径。本章已在 zyk 使用 nvcc 12.8 独立编译并通过 GPU/CPU 对照，见[验证记录](results/validation.md)。

## 1. 从小矩阵得到数学语义

对 query `i` 与 key `j`，分数为 `dot(Q_i,K_j)/sqrt(16)`。非因果版本允许所有 `j`；因果版本只允许 `j<=i`，其余位置分数记为负无穷，Softmax 概率应**精确为 0**。每行先减最大有效分数，再取指数并除以指数和，输出为 `sum_j P_ij V_j`。若 `S=2`、分数行是 `[0,ln 3]`，概率是 `[1/4,3/4]`；若查询为第 0 行且启用因果 mask，则只能看 key0，概率是 `[1,0]`，输出就是 `V_0`。这说明 mask 改变的是归一化范围，而不仅是最后把输出若干项清零。

显式路径的 `scores` 和 `probability` 分别需要 `S²` 个 float；在本例 `S=65` 时两者共 `2*65²*4=33800` 字节，长序列会按平方增长。程序用三个 kernel 表示 QKᵀ、稳定 Softmax 和 PV，再检查每个因果禁用位置概率为 0、每行和接近 1。CPU 参考用双精度点积/指数，避免与 GPU 实现共享同一索引循环。对照容差是 `2e-3 + 2e-3*|参考值|`；若超出，应先定位 mask、缩放和数值范围，不能直接放宽。

## 2. 在线 Softmax 与分块前向

若已处理若干 key，维护最大值 `m`、指数和 `l`、以及加权 V 分子向量 `o`。下一 tile 的最大值并入 `m' = max(m, tile_max)`；旧状态按 `exp(m-m')` 重标定，新 tile 权重按 `exp(score-m')` 加入：

```text
l' = l * exp(m-m') + sum_tile exp(score-m')
o' = o * exp(m-m') + sum_tile exp(score-m') * V
output = o_final / l_final
```

首次 tile 还没有有效旧分数，本例将旧项的重标定系数设为 0。对因果行，纯未来 key tile 被跳过。实现每个线程负责一个 query，按 16 个 key 为一组循环；局部保存 16 个分数和 16 个输出累计值。它说明**在线重标定与不保存 S² 中间矩阵**的正确性原理，但没有把 Q/K/V 切成共享内存 tile，也没有 Warp 协作、Tensor Core、重计算/反向、KV Cache、量化或 FlashAttention 的生产级性能优化，不能称作 FlashAttention 实现。显式临时缓冲在进入在线路径前释放，才使两条路径的设备内存需求有可观察差别。

## 3. 构建、运行与解释结果

在 zyk 仓库根目录执行：

```bash
cmake -S chapters/ch30-attention-inference/examples -B build/ch30 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch30 -j
ctest --test-dir build/ch30 --output-on-failure
./build/ch30/attention_forward
```

默认跑 `S=5` 的非因果/因果、`S=17` 因果、`S=65` 的两种 mask，并额外用 `S=17` 的大幅 Q/K 值检验减最大值的稳定性。每组显式和在线路径都需 `bad=0 PASS`；末尾是 `chapter 30 attention forward: PASS`。程序报告单次 `kernel_ms`（显式路径包含三个 kernel，在线路径一个 kernel），不含分配、H2D/D2H、CPU 参考和初始化；不可用这些数值宣称哪条实现普遍更快。记录序列长度、mask、dtype、数据范围、GPU、预热与重复样本，再用 profiler 检查中间矩阵写读、寄存器压力和占用率。

框架对照要固定同一 Q/K/V 形状与数据、缩放因子、因果 mask、dtype、是否允许 TF32/融合核以及容差。若 zyk 有兼容 PyTorch，可另用 `scaled_dot_product_attention` 进行同语义结果与时间对照；**本章当前代码尚未实现或实测这个对照**。推理中的 KV Cache 可复用已算的 K/V，但还需明确位置、分页/容量、并发读写；量化又引入 scale、零点、累加精度与校准问题。这些是本章扩展专题，不应由当前 float32 前向结果推断正确。

## 4. 常见错误

- 只把未来位置输出设零，却让它们参与 Softmax 分母。
- 先对原始极大 logit 取指数，再试图除法修复 `Inf/Inf`。
- 在线算法更新最大值后，忘记按 `exp(old_max-new_max)` 重标定旧分子和旧分母。
- 对纯未来 tile 继续把 `-Inf` 送入 `exp(score-tile_max)`，得到 `NaN`。
- 把 `S×S` 临时缓冲仍常驻设备，却声称在线路径实际节省了这些显存。
- 因单次在线 kernel 更慢，就认为在线 Softmax 思想无价值；本实现没有共享内存/矩阵指令优化。
- 对框架结果只看最大误差，不核对 mask、缩放、TF32 与 dtype 语义。

## 5. 练习与参考答案

1. `S=2`、一行分数 `[0,ln3]`，Softmax 是多少？答：`[1/4,3/4]`。
2. 因果查询第 0 行的概率是什么？答：只有 key0 有效，`[1,0,...]`，输出为 `V_0`。
3. `S=1024`、float32 分数与概率两张矩阵占多少显式设备内存？答：`2*1024²*4=8,388,608` 字节，约 8 MiB，不包括 Q/K/V/输出与其它开销。
4. 在线更新的最大值从 10 变 12，旧 `l,o` 乘什么？答：`exp(10-12)=exp(-2)`，随后加入新 tile 基于最大值 12 的权重。
5. 本例在线代码是否能直接称 FlashAttention？答：不能；它展示在线归一化和有界临时状态，缺少生产级分块数据搬运、Warp/矩阵协作、融合与反向等实现要件。

[下一章](../ch31-linear-sparse-solvers/README.md) 进入线性代数与稀疏求解，再转向 FFT、热扩散等科学计算。
