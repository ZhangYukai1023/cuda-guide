# 第 21 章：像素、通道、布局与逐像素操作

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch20-testing-deployment/README.md)

本章从可观察的字节布局开始做图像处理：灰度图的 ROI 反色、亮度与阈值，RGB/BGR 转灰度，以及 RGBA 亮度变换。示例用 5×3 的小图和故意加宽的行跨度检查索引，逐字节与 CPU 参考实现比较，再输出 PGM/PPM 图像供查看。源码和公共 PNM 头文件已在 zyk 编译；六项默认图像操作和可选 PGM 输入路径均通过 GPU/CPU 逐字节对照。

## 1. 像素、通道与行跨度

8 位灰度图每像素 1 字节，交错 RGB/BGR 每像素 3 字节，交错 RGBA 每像素 4 字节。设行首为 `base + y * stride`，像素首字节为 `base + y * stride + x * channels`。`stride` 的单位是字节，可能大于 `width * channels`；行末多出的字节是填充，不属于图像像素。ROI 用左上角 `(roi_x, roi_y)` 和宽高表示，内核线程的局部 `(x, y)` 先判断是否在 ROI 中，访问时再加 ROI 偏移。

RGB 与 BGR 的字节顺序必须由输入契约决定；仅凭三通道无法猜测。RGBA 示例只处理前三个颜色通道，原样复制 alpha。若业务数据采用预乘 alpha，颜色变换与合成规则要另行定义，不能直接套用这里的直通 alpha 语义。

本章的 RGB/RGBA 缓冲是**交错布局**：一个像素的通道相邻。平面布局则先存整张 R 平面，再存 G/B 平面，每个平面可有自己的行跨度；它需要不同的地址计算，不能把平面指针强转为交错指针。8 位示例的数值范围为 0—255；浮点图像常用 0—1 或其它物理量范围，必须由接口明确，不可在转换时默认除以 255。本章操作直接作用于编码后的 8 位值，不做 sRGB 解码；需要线性光计算的算法应先明确传递函数和转换步骤。

示例的灰度公式是整数近似 `Y = (77R + 150G + 29B + 128) >> 8`，其 CPU 与 GPU 结果应逐字节相同。它用于教学，不声称实现色彩管理或感知上精确的亮度。亮度操作把 `v + delta` 饱和到 `[0,255]`，阈值规则是 `v >= threshold` 输出 255，否则输出 0；先写清这些边界规则，再谈优化。

## 2. 构建与运行

本章已使用仓库 `chapters/common/image_io.hpp`。在仓库根目录执行：

```bash
cmake -S chapters/ch21-image-layout/examples -B build/ch21 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch21 -j
ctest --test-dir build/ch21 --output-on-failure
./build/ch21/image_layout build/ch21/manual-images
```

`ch21_image_layout` 的通过条件是 ROI 内像素正确、ROI 外与所有填充字节未变化，亮度/阈值逐字节一致，RGB 与 BGR 两条路径输出相同灰度，RGBA 的 alpha 不变。程序每项打印 `mismatches=0 PASS`；末尾打印 `chapter 21 image layout: PASS`。图像写到命令给出的目录，包括输入、ROI、亮度、阈值和通道转换结果；`-zoom` 文件用最近邻放大 16 倍供检查像素边界，`-diff` 是 CPU/GPU 绝对差异图。可以用支持 PGM/PPM 的图像查看器打开，也可用 `python3` 读取原始字节核对。小图仅用于发现索引错误，性能结论要换真实尺寸并重复采样。

可选地传入一张 **8 位二进制 P5 灰度**图：

```bash
./build/ch21/image_layout build/ch21/user-images /path/to/input.pgm
```

程序输出 `user-invert.pgm`，用 CPU 全像素反色作为参考。输出的 `kernel_ms` 是 CUDA Event 在默认流上测的单次内核时间；`task_ms` 包含主机到设备复制、内核、设备到主机复制及同步；`file_read_ms` 和 `file_write_ms` 单独列出。设备分配、目录创建及文件系统缓存状态并未统一计入 `task_ms`，所以这些数字不能直接当完整业务吞吐量。PFM、JPEG、PNG、16 位 PNM 和色彩空间元数据不在这个最小读取器的支持范围内；实际项目须用合适的解码器明确输出格式与行跨度。

## 3. 对照与观察

默认灰度输入为 5×3、`stride=8`，后三字节每行填 `0xee`。ROI 为 `(1,1,3,2)`；内核只修改其中六个像素。RGB/BGR 输入为 5×3、`stride=20`，灰度输出 `stride=8`，填充字节预填 `0xcc`。RGBA 输入 `stride=24`，填充字节预填 `0xaa`。CPU 参考按同一索引与舍入规则生成整个目标缓冲，比较时**包含填充字节**，从而能发现越界写入或 ROI 外误写。RGB 目标缓冲只预填哨兵值，防止内核没有写入也误报通过。

若扩展到 `cudaMallocPitch`，需把返回的 `pitch`（字节）传给内核；不能假设它等于 `width * channels`。若读取含下采样色度的 YUV，平面尺寸、采样位置、范围和转换矩阵均需显式约定；不能仅改通道数就复用 RGB 内核。

## 4. 常见错误

- 把 `stride` 当像素数，三通道图从第二行开始错位。
- 以 `width * channels` 推断外部图像或 `cudaMallocPitch` 的行距。
- ROI 的局部线程坐标用于边界判断后，访问地址忘记加左上角偏移。
- BGR 缓冲按 RGB 读取，颜色相近的测试图不易发现；应用红蓝不同的输入。
- RGB 转灰度时 CPU 与 GPU 使用不同的系数或舍入，误把算法差异当浮点误差。
- 亮度运算先转回 `uint8_t` 再截断，导致溢出回绕。
- 修改 RGBA 时覆盖 alpha 或行填充，破坏后续合成。
- 只看输出图像“差不多”，没有保存逐字节差异和输入格式约定。

## 5. 练习与参考答案

1. 图宽 5、RGB 交错、`stride=20`，第 2 行第 3 个像素的 R 在第几个字节？答：若行列从 0 起算，偏移为 `2*20 + 3*3 = 49`；这一像素的三个通道位于 49—51。
2. ROI `(1,1,3,2)` 在 5×3 图中覆盖哪些像素？答：`x=1,2,3` 且 `y=1,2`，共六个；其他像素和填充必须保持原值。
3. 输入 `R=255,G=0,B=0`，本章公式的灰度值是多少？答：`(77*255+128)>>8 = 77`。
4. 若 `v=240,delta=40`，亮度结果是什么？答：先用足够宽的整数计算 280，再饱和为 255；不是 `uint8_t` 回绕后的 24。
5. 怎样发现“内核未写输出，却通过了比较”？答：用与预期像素不同的哨兵值初始化输出缓冲；检查所有有效像素与填充，并加入至少一个预期不等于哨兵的输入。

[下一章：均值与高斯滤波](../ch22-mean-gaussian/README.md)在明确像素布局的基础上进行邻域处理，并处理边界与共享内存重用。
