# 第 9 章：科学计算：二维热扩散

[全书导航](../../README.md) · [上一章](../ch08-ai-operators/README.md) · [下一章](../ch10-multi-gpu/README.md)

## 1. 一个热点如何向四周扩散

想象一个 3×3 网格，中心温度 100，四周和边界为 0。采用更新式：

```text
new = center + alpha*(left+right+up+down-4*center)
```

alpha=0.2 时，中心下一步为 100+0.2×(-400)=20。由于这个最小网格的外围全部是固定边界，外围仍为 0。本机 float 实测中心为 19.999998，与手算的差别来自舍入。

更大的网格中，原热点的相邻内部点会升温。更新必须全部读取同一个旧时刻的数据，否则会把已经更新过的邻居混入本轮，变成不同的算法。

## 2. 数学模型与边界条件

它是二维热方程在均匀网格上的显式有限差分形式。若 x、y 网格间距均为 dx，热扩散率为 kappa，则 alpha=kappa*dt/(dx²)。

对本章的四邻域形式，新值也可以写成：

```text
new = (1-4*alpha)*center + alpha*(left+right+up+down)
```

当 0<=alpha<=1/4 时，各系数非负且和为 1，因此每次内部更新是旧值的加权平均。这解释了本例不会凭空产生超过初始最大值的温度，也给出常见的稳定步长限制。不同维数、网格间距和离散方法需要重新推导，不能套用 1/4。

本章采用固定的零边界，即 Dirichlet 边界；边缘每轮复制旧值。边界相当于维持低温的外部环境，系统热量可能流失，所以不能把总温度严格守恒作为这里的验证条件。

## 3. 最小示例：两份状态，交替读写

```text
第 0 步：A 初值 → kernel → B
第 1 步：B      → kernel → A
第 2 步：A      → kernel → B
```

源码用 src、dst 指针交换，而不搬动整个数组。两个 DeviceBuffer 对象始终拥有原来的内存；交换的是工作指针，避免资源所有权混乱。

最后从 src 取结果，不能总从 B 复制。程序实际测试 0、1、2 步，专门覆盖不执行、奇数步和偶数步。零步时 src 仍指向初值。

一个时间步用一个 kernel，相同流的提交顺序保证下一轮看到上一轮结果。因此不必每轮让 CPU 同步，但最后取结果前仍要同步并检查错误。普通块内 __syncthreads 无法让所有网格点一起跨越时间步，不能直接把多轮循环搬进一个普通 kernel 后假定全局一致。

## 4. 递进示例：更大的网格与独立参考

程序测试 7×5 的 0、1、2 步，以及 33×25 的 20 步。CPU 使用 double 状态数组，按同样的时间层推进。系数从与 GPU 相同的 float alpha 转成 double，避免“常数本身不同”混入比较。

容差为 atol=2e-5、rtol=2e-5，并检查 GPU 值都在 [0,100]。本次 20 步最大绝对误差约 1.42e-7，中心温度为 1.952703。范围检查是补充性质，不能替代逐项参考：全写成 0 也会满足范围。

| 初始状态 | 20 步后 |
| --- | --- |
| ![initial](results/images/initial.svg) | ![diffused](results/images/diffused.svg) |

两图统一把温度 0..100 映射到灰度 0..255。扩散后亮度较低是同一标尺下的实际变化，没有对每张图分别拉伸对比度。保存前的误差仍在温度单位中计算，而不是对量化图像做比较。

## 5. 验证正确不等于验证物理模型

CPU 对照证明 GPU 实现接近同一离散算法。要用于真实科学问题，还需要分析网格收敛、时间步收敛、边界条件、单位和材料参数；本章没有声称已完成物理实验校准。

更复杂的科学计算会遇到归约顺序、长时间累计误差、双精度需求和通信。可以先用较小网格、较短时间、已知解析或制造解建立验证，再扩大规模。本章的热点例属于可追踪的入门离散实验。

## 6. 常见错误

- 原地更新，读到本轮的新邻居值。
- 忘记更新边界输出，下一轮读到未初始化内存。
- 交换指针后从固定对象而非当前 src 下载，偶数步出错。
- alpha 过大出现不稳定，却误判为 CUDA 浮点错误。
- 在 kernel 中用块内屏障冒充全网格同步。
- 只对比图片，不验证数值、边界和迭代次数。

## 7. 练习与参考答案

1. alpha=0.25，3×3 中心初始为 100，边界为 0，一步后中心是多少？
   答：0；本章源码使用 0.2，这里只是数学练习。
2. alpha=0.3 时热点一步后中心是什么？
   答：100-120=-20，非负温度的加权平均性质被破坏，说明不能随意增大时间步。
3. 运行两步后结果在最初的 A 还是 B？
   答：A，但应始终使用 src，不要手写奇偶分支。
4. 为什么固定零边界下总和不必守恒？
   答：热量可以通过边界流出；边界不是绝热条件。
5. 把网格间距减半，为保持相同 alpha，时间步如何变化？
   答：dt 应缩小为原来的 1/4。

## 构建、运行与实测输出

完整源码：[heat.cu](examples/heat.cu)；构建定义：[CMakeLists.txt](examples/CMakeLists.txt)。公共依赖也在本仓库，不需要复制未提供的代码。

在项目根目录执行（首次配置后可重复构建）：

```bash
cd /data2/cuda-guide
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all --target heat -j2
./build/all/ch09/heat build/ch09-images
```

也可单独配置本章：把上面 -S 改为 chapters/ch09-scientific-computing/examples，-B 改为 build/ch09-standalone，其他选项不变；程序位于该独立构建目录下。无需第三方图像或数学库。

本次实际输出如下。除计时数据可能随运行波动外，同一固定输入应得到这些结果或在注明容差内一致；最终错误数量应为 0：

```text
heat: n=9 max_abs_error=7.15255737e-07 mismatches=0 PASS
shape=3x3 steps=1 center=19.999998 range=[0.000000,19.999998]
heat: n=35 max_abs_error=0 mismatches=0 PASS
shape=7x5 steps=0 center=100.000000 range=[0.000000,100.000000]
heat: n=35 max_abs_error=7.15255737e-07 mismatches=0 PASS
shape=7x5 steps=1 center=19.999998 range=[0.000000,20.000000]
heat: n=35 max_abs_error=3.57627876e-07 mismatches=0 PASS
shape=7x5 steps=2 center=20.000000 range=[0.000000,20.000000]
heat: n=825 max_abs_error=1.41793282e-07 mismatches=0 PASS
shape=33x25 steps=20 center=1.952703 range=[0.000000,1.952703]
```

运行命令将效果图写入 build/ 下，避免覆盖归档结果；正文引用的实测图已保存在 results/images/。SVG 和 PGM 均为程序生成。

更多环境与限制见 [validation.md](results/validation.md)。本机没有 Compute Sanitizer，不能把 CPU 对照通过等同于内存或同步工具检查通过。
