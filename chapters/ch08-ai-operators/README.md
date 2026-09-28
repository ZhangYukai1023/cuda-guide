# 第 8 章：AI 算子：矩阵乘法与 Softmax

[全书导航](../../README.md) · [上一章](../ch07-image-pipeline/README.md) · [下一章](../ch09-scientific-computing/README.md)

## 1. 矩阵乘法先算清行和列

输入 A 有 2 行 3 列，B 有 3 行 2 列：

```text
A = [1 2 3]    B = [1 2]
    [4 5 6]        [3 4]
                   [5 6]
C = [22 28]
    [49 64]
```

例如 C[0,0]=1×1+2×3+3×5=22。输出每个位置对应 A 的一行与 B 的一列做点积。最小 kernel 让一个线程负责一个输出元素，再在该线程中遍历公共维度 k。

数据都按行优先连续存放：A[row*k+p]、B[p*n+col]、C[row*n+col]。m、n、k 分别控制输出行数、输出列数、点积长度，不能因为示例是方阵就把三者合并。

源码还测试 m=17、n=19、k=23，使用非整块形状暴露边界和步长错误。CPU 参考用 double 累加，比较容差为 atol=1e-5、rtol=1e-5。当前输入采用二进制可精确表示的小值，本次最大误差为 0；这不是一般浮点矩阵乘法的承诺。

这个直接实现没有共享内存 tiling 或 Tensor Core，也不是库性能基线。先把接口、布局和参考结果固定，后续才能有意义地引入优化或与 cuBLAS 等库比较。本机没有在此示例中链接或验证这些库。

## 2. Softmax 为什么需要“先减最大值”

对 [0,0]，exp 后是 [1,1]，归一化得到 [0.5,0.5]。对 [1000,1000]，数学上也应是 [0.5,0.5]，但直接算单精度 exp(1000) 会溢出。

稳定写法：

```text
m = max(x)
p[i] = exp(x[i]-m) / sum_j exp(x[j]-m)
```

分子分母同时乘 exp(-m) 不改变数学比值。减最大值后所有指数输入不大于 0，至少一项为 exp(0)=1，有限输入时分母不会全部下溢为 0。特别小的项仍可能下溢为 0，这与“所有值都能被精确表示”不同。

本章只定义有限 float 输入，不支持含 NaN、正无穷或全负无穷的扩展语义；这些情况在框架中需要单独约定。

## 3. 递进示例：一块处理一行，做两次归约

一行可能有 257 项，不能要求每一项都对应一个常驻线程。固定 128 个线程，线程 t 先处理 t、t+128、t+256……，得到自己的局部最大值；共享数组再做树形归约。

```text
128 个局部结果
 → 64 对合并
 → 32 对合并
 → 16 → 8 → 4 → 2 → 1
```

每一轮合并后都需要屏障，下一轮才能读取其他线程刚写好的值。第一轮求最大值，第二轮求指数和，最后每个线程写出自己的若干概率。

共享数组被复用前还有一个容易遗漏的屏障：所有线程必须先读走 work[0] 中的最大值，才能让某些线程用局部和覆盖它。源码在读取 maximum 后显式同步。不能以为第一个 warp 先执行就天然安全。

这个 kernel 要求启动 blockDim.x 恰好为 128，因为共享数组和归约步长写死了 128。修改启动参数必须同时修改实现，不能把 block 大小当成任意可调按钮。

## 4. 验证要覆盖什么

测试列数为 1、3、129、257。三行输入分别位于约 +1000、约 -1000 和常量 7。它们检查单元素、跨线程循环、大正数、大负数和相等输入。

CPU 使用 double 计算稳定 Softmax，GPU 与其比较 atol=2e-7、rtol=2e-5，并检查概率非负与每行和接近 1，行和容差为 2e-6。仅检查行和不够：所有元素都写成 1/cols 也满足行和，却可能完全忽略输入。

本次最大绝对误差约 7.19e-8。误差日志和矩阵小例的 22 28 49 64 均来自实际运行。没有使用 fast-math，也没有测量算子吞吐。

## 5. 从教学算子走向工程接口

接入 AI 框架时，还需定义 dtype、布局、stride、批次、设备与 stream。普通连续二维数组只是最简单的一种情况。广播、非连续张量、半精度累加和反向传播都不能从本章自动获得。

建议先让正确性契约清晰：输入形状是否合法，空张量怎么办，谁拥有输出，当前 stream 是否由调用者提供。本章固定正形状，作为完整可运行实验；生产接口需补这些边界，而不是把示例函数直接当通用库。

## 6. 常见错误

- 将行优先 B 当成列优先，方阵可能让尺寸错误不明显。
- 直接 exp(x)，大数溢出。
- 归约阶段只让活跃线程到达屏障。
- 归约结果尚未被所有线程读走就复用共享数组。
- 只比较 argmax 或行和，不逐项检查。
- 把教程 kernel 的性能当成 cuBLAS、cuDNN 或框架算子的性能。

## 7. 练习与参考答案

1. A=[2,3]、B=[4,5]ᵀ，输出是多少？
   答：2×4+3×5=23。
2. 对三项相同的有限值做 Softmax，结果是什么？
   答：每项 1/3；float 存储有舍入。
3. 一行有 257 项，线程 0 在局部归约中读哪些列？
   答：0、128、256。
4. 为什么不能直接让每个线程 atomicAdd 到分母后立刻除？
   答：它无法知道其他线程是否已经完成，跨块时更无普通块内屏障可用；需要明确的归约阶段与同步依赖。
5. 如果把 block 改为 64，当前 Softmax 是否仍正确？
   答：不正确，固定 128 项的共享归约会读取未定义数据；必须一起重写或模板化实现。

## 构建、运行与实测输出

完整源码：[operators.cu](examples/operators.cu)；构建定义：[CMakeLists.txt](examples/CMakeLists.txt)。公共依赖也在本仓库，不需要复制未提供的代码。

在项目根目录执行（首次配置后可重复构建）：

```bash
cd /data2/cuda-guide
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all --target operators -j2
./build/all/ch08/operators
```

也可单独配置本章：把上面 -S 改为 chapters/ch08-ai-operators/examples，-B 改为 build/ch08-standalone，其他选项不变；程序位于该独立构建目录下。无需第三方图像或数学库。

本次实际输出如下。除计时数据可能随运行波动外，同一固定输入应得到这些结果或在注明容差内一致；最终错误数量应为 0：

```text
matmul: n=4 max_abs_error=0 mismatches=0 PASS
22 28 49 64
matmul: n=323 max_abs_error=0 mismatches=0 PASS
softmax: n=3 max_abs_error=0 mismatches=0 PASS
softmax cols=1 row_sums PASS
softmax: n=9 max_abs_error=7.19476123e-08 mismatches=0 PASS
softmax cols=3 row_sums PASS
softmax: n=387 max_abs_error=6.24329424e-09 mismatches=0 PASS
softmax cols=129 row_sums PASS
softmax: n=771 max_abs_error=2.31778568e-09 mismatches=0 PASS
softmax cols=257 row_sums PASS
```

更多环境与限制见 [validation.md](results/validation.md)。本机没有 Compute Sanitizer，不能把 CPU 对照通过等同于内存或同步工具检查通过。
