# 第 27 章：张量与 PyTorch 自定义算子

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch26-image-pipeline/README.md)

前六篇的 kernel 直接处理指针，本章把它们接入有 Shape、Stride、Dtype、Device 和当前 Stream 约束的张量框架。完整示例在 [examples](examples/)：`scale(x,factor)` 将 float32 CUDA 张量逐元素乘常数；`scale_relu(x,factor)` 再做 `max(0,v)`，并注册反向公式。Python 检查覆盖空张量、非整齐长度、二维、非连续转置视图、当前 Stream、错误输入与梯度。示例已使用 zyk 上现有的 ComfyUI PyTorch 虚拟环境与 nvcc 编译并通过 GPU 检查；当前 PyTorch 2.13.0+cu130 与 nvcc 12.8 有版本差异警告，具体环境和命令见验证记录。

## 1. 算子接口先于 kernel

两算子的输入为任意形状的 **CUDA float32** Tensor，`factor` 为有限 Python 浮点数；输出形状、设备和 dtype 与输入一致，输出是新分配的连续 Tensor。非连续输入会在 C++/CUDA 边界调用 `contiguous()` 复制，语义上支持任意正常 strided 视图，但复制成本计入任务时间；本例没有实现逐 stride 读取或广播。空输入直接返回空输出，不启动零 Block kernel。元素数超过 32 位索引范围会报错。CPU Tensor 与 float16 被拒绝，避免错误地从 CUDA 实现静默回退。NaN/Inf 输入和 `factor` 转 float32 后溢出的细节不在本章支持范围内；性能或训练项目必须先扩展并测试数值契约。

`ops.cpp` 用 `TORCH_LIBRARY` 定义两个 schema；`ops_cuda.cu` 用 `TORCH_LIBRARY_IMPL(..., CUDA, ...)` 关联设备实现。C++ 包装先检查输入与因子，使用 `CUDAGuard` 切到输入设备，再从 PyTorch 查询**当前 CUDA Stream**提交 kernel，避免绕过用户在 Python 中设置的 Stream。`cudaGetLastError` 检查启动错误；实际异步执行错误会在后续同步或 Tensor 消费时报出。输出由框架分配和拥有，本章不返回裸设备指针。`setup.py` 采用 PyTorch 的 `CUDAExtension` 构建共享库 `_C`；`ops.py` 先用 `torch.ops.load_library` 加载静态注册，再注册 fake/meta 和 autograd。共享库不需要 Python 扩展初始化函数，因此本机缺少 Python 开发头文件时仍可构建。

这套例子使用 PyTorch 的非稳定 ATen C++ API，**须针对实际安装的 PyTorch 版本构建**，不能把生成的二进制拿到任意版本复用。官方教程说明 PyTorch 2.4 及以后提供本章所用自定义算子路径，较新的版本另有稳定 ABI；本次已针对 zyk 虚拟环境的 PyTorch 2.13.0+cu130 实测；换环境或升级框架后仍须重新构建验证。参考 [PyTorch 自定义 C++/CUDA 算子教程](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)、[扩展构建接口](https://docs.pytorch.org/docs/stable/cpp_extension.html)和 [CUDA Stream 工具](https://docs.pytorch.org/cppdocs/api/cuda/utilities.html)。

## 2. 前向、反向和注册顺序

`scale` 的参考是 `x*factor`；其反向是 `grad_output*factor`。`scale_relu` 的前向是 `relu(x*factor)`，本章有限输入的反向是 `grad_output * (x*factor>0) * factor`，在零点按 0 的次梯度约定。反向函数用已有 PyTorch 算子构成，框架可以追踪它；它不通过裸 CUDA 指针直接改梯度。`ops.py` 的 fake 注册告诉形状推断输出与输入同形同 dtype、连续；这有助于 `opcheck` 检查注册契约。若先注册 fake/autograd 再加载 `_C`，算子 schema 尚不存在，会报注册错误。

Python 检查对 `shape=(0,), (7,), (3,5), (1003,)` 比较前向输出；对转置视图明确确认 `is_contiguous()==False` 后比较；在新建 CUDA Stream 的上下文中运行并同步；反向对比手算掩码和中心差分。差分输入远离 ReLU 零点，步长 `1e-3`，容差 `2e-3`，这不是对所有非光滑点的数学证明。最后运行 `torch.library.opcheck` 检查代表性注册行为。CPU/float16 拒绝也是测试的一部分。

## 3. 在已有环境构建与运行

先检查已有 Python、PyTorch 和 CUDA Toolkit。系统 `/usr/bin/python3` 没有 PyTorch；本次使用服务器现有 `/data/ComfyUI/.venv/bin/python`，不安装新软件。以下从仓库根目录运行，编译产物放在忽略的 `build/` 中：

```bash
/data/ComfyUI/.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
cd chapters/ch27-pytorch-custom-op/examples
CUDA_HOME=/home/zhangyukai/.local/cuda \
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=2 \
/data/ComfyUI/.venv/bin/python setup.py build_ext \
  --build-lib /data2/cuda-guide/build/ch27-ext/lib \
  --build-temp /data2/cuda-guide/build/ch27-ext/temp
LD_LIBRARY_PATH=/home/zhangyukai/.local/cuda/lib64:/data/ComfyUI/.venv/lib/python3.12/site-packages/torch/lib \
CUDA_GUIDE_OPS_LIBRARY=/data2/cuda-guide/build/ch27-ext/lib/_C.cpython-312-x86_64-linux-gnu.so \
/data/ComfyUI/.venv/bin/python check_ops.py
```

`CUDA_GUIDE_OPS_LIBRARY` 指向实际生成的共享库；若 Python ABI 后缀不同，要替换文件名。`LD_LIBRARY_PATH` 使本机动态加载器找到 Toolkit 的 `libcudart.so.12`。PyTorch 对其 CUDA 13.0 构建与 nvcc 12.8 给出版本差异警告，但本次编译、加载和 GPU 测试均成功；此事实不保证其它 PyTorch/CUDA 组合可用。最初直接构建因 `Python.h`、`cusparse.h` 缺失失败，已通过无 Python 初始化函数的库加载方式和更轻量的 Stream 头文件解决，不改动驱动或安装依赖。

测试输出包含 `opcheck: PASS` 与 `chapter 27 custom operators: PASS`，详见[验证记录](results/validation.md)。本章扩展按 PyTorch 自身机制单独构建测试，不在根 CMake/CTest 中增加目标；根工程回归结果也另行记录。

## 4. 常见错误

- 直接用 `data_ptr<float>()` 遍历转置输入，却按连续地址解释；本例先 `contiguous()`。
- CUDA 包装没切到输入设备，当前进程默认设备与 Tensor 实际设备不同。
- 强制提交到默认 Stream，导致用户 Stream 上的依赖与 Tensor 生命周期出错。
- 元素数为 0 时仍以 0 Block 启动 kernel。
- Python 注册顺序颠倒，或只写 CUDA 前向却期待自动得到正确反向。
- 把 ReLU 在零点的次梯度约定与正值处导数混淆。
- 对 float32 用过严的数值差分容差，或在非光滑点做中心差分。
- 将针对某 PyTorch/CUDA 版本编译的扩展视为可跨版本直接复用。

## 5. 练习与参考答案

1. 形状 `(3,5)` 的转置视图是什么形状？通常是否连续？答：`(5,3)`；普通转置视图 stride 改变，通常非连续，本例内部会复制成连续存储。
2. `x=[-2,1]`、`factor=1.5`，`scale_relu` 输出和对输入的梯度是什么？答：输出 `[0,1.5]`；若上游梯度都为 1，输入梯度 `[0,1.5]`。
3. 为什么不能把当前 Stream 换成默认 Stream？答：用户在另一个 Stream 上提交输入写入与后续消费，擅自换流会打乱依赖顺序。
4. 空 Tensor 的 CUDA kernel 应怎样处理？答：返回正确形状、dtype、设备的空输出，不做零 Block 启动。
5. 非连续输入被支持，是否代表零拷贝？答：否；本例调用 `contiguous()`，可正确处理但会发生设备端复制。若要避免复制，需显式实现 stride 索引并重新验证。

[下一章：Softmax 与归一化](../ch28-softmax-normalization/README.md)用批量归约实现数值稳定的 Softmax、LayerNorm 与 RMSNorm。
