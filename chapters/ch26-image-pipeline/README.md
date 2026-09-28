# 第 26 章：完整图像处理流水线

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch25-edges-morphology-contrast/README.md)

本章把读图、上传、3×3 中值去噪、半像素双线性缩小、浮点归一化、下载和保存组成一个命令行任务。示例 [image_pipeline.cu](examples/image_pipeline.cu) 对同一批图像分别运行串行单槽与双槽多 Stream 路径，并对每帧的 8 位预览和浮点结果与 CPU 参考比较。合成六帧与 PGM 目录三帧的串行、双槽路径已在 zyk 编译并通过 GPU/CPU 对照；批量性能待空闲设备复测。

## 1. 输入、输出和步骤约定

输入是 8 位二进制 P5 灰度图。所有帧尺寸相同，宽高各不超过 1024，每批 1—128 张；无输入目录时生成 6 张 65×49 的确定性测试帧。去噪用第 23 章的复制边界 3×3 中值，双线性缩放沿用第 24 章的半像素中心和复制边界，输出大小为 `ceil(width/2) × ceil(height/2)`。缩小后的每个 8 位像素转换为单精度浮点 `2*v/255-1`，目标数值范围为 `[-1,1]`。中值和缩放之间保留 8 位中间结果与舍入，CPU 参考按同序执行。

每帧输出一张缩小后的 PGM 预览和一份行优先 `float32` 原始张量 `.f32`；`manifest.tsv` 记录输入文件名、输出尺寸和文件名。示例在写张量前检查主机为小端 IEEE 32 位浮点。可用下面的 Python 代码读取：

```python
import numpy as np
tensor = np.fromfile('frame-000.f32', dtype='<f4').reshape(height, width)
```

这里 `height,width` 从 `manifest.tsv` 读取；`.f32` 没有自描述头。输出文件名使用帧序号以免输入同名或奇怪字符覆盖文件。预览 PGM 用于人工观察，模型输入应使用 `.f32` 与本章的形状、范围约定。

## 2. 从串行到双槽

单槽版本为每帧顺序执行 H2D、三个 kernel、D2H 并等待完成。双槽版本为每槽持有独立的 pinned Host 输入/输出、设备输入/中间/输出、非阻塞 Stream 与计时 Event。第 `i` 帧使用 `i%2` 的槽；**复用前先对该槽的 Stream 同步**，把上一帧结果复制到普通 Host 向量，才改写 pinned 输入。每个槽内依赖靠同一 Stream 的提交顺序维持：H2D → 中值 → 缩放 → 归一化 → D2H。不同槽互不共享中间缓冲，因而不会把一帧的预览拿去归一化另一帧。即便双槽在某 GPU 上没带来速度收益，这一资源归属和同步规则仍是正确性条件。

CPU 参考先算好全部帧，但不计入 GPU 批量时间；两条 GPU 路径都需要 `preview_mismatches=0` 且 `norm_mismatches=0`，浮点容差 `1e-6`。不能只拿双槽结果和单槽结果比：若两者共享同一个错误，仍会一起通过。示例保留三个 kernel；如果融合中值、缩放或归一化，必须维持此处**缩放先输出 8 位再归一化**的舍入语义，或明确新接口允许的差异后重做 CPU 参考。

## 3. 构建与运行

仓库已包含公共 `chapters/common/image_io.hpp`。在仓库根目录执行：

```bash
cmake -S chapters/ch26-image-pipeline/examples -B build/ch26 \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ch26 -j
ctest --test-dir build/ch26 --output-on-failure
./build/ch26/image_pipeline build/ch26/manual-output
```

将同尺寸 P5 文件放在一个目录，再运行真实批量：

```bash
./build/ch26/image_pipeline build/ch26/user-output /path/to/pgm-directory
```

程序按文件名排序，先完整读入一批，再做 GPU 处理与 CPU 对照，最后保存双槽结果。它不是无限流式读取器；对大目录或长视频应加受控队列、错误恢复和增量写盘。每帧输出 `kernel_ms`（本槽 Event 记录的三个 kernel 总时间）与 `frame_task_ms`（Host 从入队到观察完成的延迟，包含该槽传输/排队/等待与拷回普通向量）。批量输出 `serial_batch_ms` 和 `double_slot_batch_ms`，范围覆盖任务提交、等待及拷回，不含文件读写、CPU 参考和设备缓冲区分配；`file_read_ms`、`file_write_ms` 单列。`device_buffer_bytes_per_slot` 是本例显式分配的字节数，双槽约为两倍，**不代表驱动或 CUDA 上下文的峰值显存**。数值先预热、重复并观察设备/拷贝引擎能力后才可比较，不能预设双槽一定更快。

NPP 若提供同语义的中值与缩放路径，可以另做库对照；先匹配边界、半像素、舍入及 Stream 上下文，避免把不同算法结果直接比较。当前示例不调用 NPP，也不宣称已与其逐像素一致。参考 [NPP 图像滤波](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html)和 [NPP 几何变换](https://docs.nvidia.com/cuda/npp/image_geometry_transforms.html)。

## 4. 常见错误

- `cudaMemcpyAsync` 使用可分页 Host 内存却期待稳定的传输重叠；本例为每槽明确申请 pinned 内存。
- 前一帧的异步 D2H 尚未完成，就复用 Host 输出或设备中间缓冲。
- 为两个 Stream 共用同一预览缓冲，偶尔得到另一帧的归一化张量。
- 缩放前后归一化顺序交换，却仍使用原 CPU 参考和逐字节预览判定。
- 只计算三次 kernel 的 Event 和，拿来冒充文件读写及整个批量时间。
- 把“显式分配字节数”称为实际峰值显存，遗漏上下文、库与分配器开销。
- 输入目录混入不同尺寸或非 P5 文件；本例拒绝不同尺寸，忽略非 `.pgm` 扩展名文件。
- 未确认 `.f32` 的字节序、形状和范围就直接交给模型。

## 5. 练习与参考答案

1. 65×49 输入的输出尺寸是多少？答：`ceil(65/2)=33`、`ceil(49/2)=25`。
2. 第 4 帧（从 0 起）在双槽里用哪个槽？答：`4%2=0`，必须等槽 0 的上一帧完成再覆盖。
3. 8 位像素 `v=0,127,255` 的归一化大约是多少？答：`-1`、`-0.00392`、`1`。
4. 为何不能仅用串行输出验证双槽？答：相同错误可能存在于两条 GPU 路径；两者都需与独立 CPU 参考比较。
5. 合并缩放和归一化内核时，怎样保持本章结果？答：先按原规则把插值量舍入/饱和成 8 位值，再从该值计算 `2*v/255-1`；若直接从未舍入浮点插值归一化，接口结果会变化。

[下一章：PyTorch 自定义算子](../ch27-pytorch-custom-op/README.md)进入张量与 AI 算子，继续把 shape、stride、dtype 和当前 Stream 写成明确的接口契约。
