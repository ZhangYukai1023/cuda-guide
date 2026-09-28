# 第 7 章：图像处理流水线

[全书导航](../../README.md) · [上一章](../ch06-image-resampling/README.md) · [下一章](../ch08-ai-operators/README.md)

## 1. 两步像素处理能否少走一次内存

设像素为 [0,100,250]。先反色 255-p，得到 [255,155,5]；再加亮 20 并截断到 255，得到 [255,175,25]。

最小实现是两个 kernel：第一个写临时数组，第二个读取它。递进实现把两步合并成 min(275-p,255)，省去一次中间图的全局内存写入和读取。这里输入在 [0,255]，否则还需要明确负数或超范围数据的语义。

融合需要证明等价。如果前一步先转成 8 位并舍入，或后一步读取邻域，那么合并公式和执行顺序可能不同。不能看到两步就机械地拼接表达式。本例的单精度合成图通过 CPU 对照，分离版与融合版均为零误差。

## 2. 最小示例：设备上保留中间图

```text
CPU 输入 → H2D → GPU 反色 → GPU 加亮 → D2H → CPU 结果
```

中间图不必回 CPU。相同流上的两个 kernel 按提交顺序执行，因此第二步可以消费第一步结果；无需在每一步之间都让 CPU 等待。只有主机要读结果或管理资源复用时，才等待相应工作完成。

本章前半段保留分离版作为容易理解的参考，再运行融合版，与同一 CPU 结果比较。两种版本都已经实际运行，未进行性能对比，不宣称融合后一定快多少。

## 3. 递进示例：两条流处理多帧

一帧的处理流程固定为 H2D → fused → D2H。为了让 CPU 可以准备下一帧，代码准备两个槽，每槽拥有独立的：

- CUDA stream；
- 输入与输出设备数组；
- 页锁定主机输入与输出数组；
- 当前尚未收取的帧编号。

```text
slot 0: frame 0 → 收取 → frame 2 → 收取 → frame 4 → 收取
slot 1: frame 1 → 收取 → frame 3 → 收取
```

这张图表达资源复用顺序，不是实测时间轴。流之间能否真的重叠取决于硬件、工作量、传输条件与调度，本章没有 profiler 证据，不能根据两个 stream 对象就宣称并发加速。

源码调用 cudaMallocHost 申请页锁定内存，再配合 cudaMemcpyAsync。普通 std::vector 不能替代这个承诺来保证传输重叠。页锁定内存也不应无限申请；本例仅保存两个小帧。

## 4. 最重要的正确性条件：谁还在使用缓冲区

CPU 不能在异步 H2D 尚未结束时改写输入，也不能在 D2H 尚未结束时读取输出。槽复用前的 collect 先 cudaStreamSynchronize，再核对对应帧的结果，随后才允许写入下一帧。

如果不等就写 frame 2 的内容，frame 0 的 DMA 可能仍在读取同一主机内存。这样的错误可能偶尔通过，不能用“多运行几次没错”证明安全。

本例测试 5 帧而非 4 帧，故意让双槽收尾不整齐。循环结束后 collect(0)、collect(1) 收取剩余工作，避免漏掉最后一帧。每一帧输入是 (i+frame)%256，CPU 参考根据帧编号生成，因此拿错帧也能被发现。

同一流表达内部依赖；跨流共享结果时应考虑 event 及等待关系，而不是依赖提交的墙钟先后。流语义可查阅 [CUDA Streams 文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#streams)。

## 5. 实际图像

| 输入 | 反色并加亮 |
| --- | --- |
| ![input](results/images/input.svg) | ![output](results/images/output.svg) |

这些效果图来自前半段融合结果；五帧双槽验证使用另一组确定性数据并记录逐帧 PASS。不要把静态效果图理解为已保存的全部视频帧。

将本章连接第 5、6 章时，可以让滤波输出直接留在设备上供几何变换读取；但滤波读取邻域，切片或分块时要考虑 halo。本章双槽示例处理完整帧，没有实现带 halo 的分块滤波。

## 6. 常见错误

- 将两个流共用同一中间缓冲区，产生跨帧覆盖。
- 异步复制提交后立即读取主机输出。
- 槽正在处理上一帧，CPU 就写入下一帧。
- 只等待默认流，实际工作在 non-blocking streams 上。
- 忘记收取末尾未满一轮的帧。
- 把算法融合、传输减少、并发重叠混为同一项性能提升。

## 7. 练习与参考答案

1. [0,100,250] 的最终结果是多少？
   答：[255,175,25]。
2. 两个槽处理 7 帧，循环结束时还要处理哪些待收取项？
   答：两个槽都要检查 pending 状态；不能按“偶数帧数量”假定只剩一个。按本例调度最后对应 frame 6 和 frame 5。
3. 如果某一步滤波需要上一帧的信息，还能独立并发处理帧吗？
   答：不能沿用独立帧假设。必须显式表示帧间依赖并保护历史缓冲区。
4. 融合版为什么不需要 tmp？
   答：一个线程在局部表达式内完成两步，只把最终值写到 out；输入与输出仍分开。

## 构建、运行与实测输出

完整源码：[pipeline.cu](examples/pipeline.cu)；构建定义：[CMakeLists.txt](examples/CMakeLists.txt)。公共依赖也在本仓库，不需要复制未提供的代码。

在项目根目录执行（首次配置后可重复构建）：

```bash
cd /data2/cuda-guide
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all --target pipeline -j2
./build/all/ch07/pipeline build/ch07-images
```

也可单独配置本章：把上面 -S 改为 pilot/ch07-image-pipeline/examples，-B 改为 build/ch07-standalone，其他选项不变；程序位于该独立构建目录下。无需第三方图像或数学库。

本次实际输出如下。除计时数据可能随运行波动外，同一固定输入应得到这些结果或在注明容差内一致；最终错误数量应为 0：

```text
two_stage: n=3185 max_abs_error=0 mismatches=0 PASS
fused: n=3185 max_abs_error=0 mismatches=0 PASS
stream_frame: n=3185 max_abs_error=0 mismatches=0 PASS
stream_frame: n=3185 max_abs_error=0 mismatches=0 PASS
stream_frame: n=3185 max_abs_error=0 mismatches=0 PASS
stream_frame: n=3185 max_abs_error=0 mismatches=0 PASS
stream_frame: n=3185 max_abs_error=0 mismatches=0 PASS
```

运行命令将效果图写入 build/ 下，避免覆盖归档结果；正文引用的实测图已保存在 results/images/。SVG 和 PGM 均为程序生成。

更多环境与限制见 [validation.md](results/validation.md)。本机没有 Compute Sanitizer，不能把 CPU 对照通过等同于内存或同步工具检查通过。
