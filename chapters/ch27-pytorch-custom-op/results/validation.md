# 第 27 章验证记录

日期：2026-09-28（Asia/Shanghai）。主机：ubuntu2404，RTX 5060 Ti（计算能力 12.0），驱动 595.84，nvcc 12.8.93。分支：`codex/align-outline-38`。

系统 Python 无 PyTorch；现有 `/data/ComfyUI/.venv/bin/python` 为 PyTorch `2.13.0+cu130`、`torch.version.cuda=13.0`，`torch.cuda.is_available()=True`。Python 源文件语法检查通过。最初扩展构建失败：本机无 `Python.h` 和 `cusparse.h`；将注册库改为 `torch.ops.load_library` 加载、使用 `c10/cuda/CUDAStream.h` 后，以 `CUDA_HOME=/home/zhangyukai/.local/cuda`、`TORCH_CUDA_ARCH_LIST=12.0`、`MAX_JOBS=2` 构建到 `build/ch27-ext/lib`，退出码 0。PyTorch 提示其 CUDA 13.0 与 nvcc 12.8 版本差异，构建日志在忽略目录 `build/ch27-ext/build2.log`。

第一次加载因动态链接器找不到 `libcudart.so.12` 失败；运行时给 `LD_LIBRARY_PATH` 加入现有 Toolkit `lib64` 后，`check_ops.py` 退出码 0。空张量、长度 7/1003、二维、转置非连续输入的前向对照，`scale`/`scale_relu` 反向、有限差分、当前 Stream、CPU/float16 拒绝和 `torch.library.opcheck` 均通过。完整 17 行 stdout 见 [run-2026-09-28.txt](run-2026-09-28.txt)，末行 `chapter 27 custom operators: PASS`。

根 `build/outline` 在本章同步后重新配置、全目标构建和完整 CTest：39 项中 38 项通过、0 项失败，1 项既有双 GPU 测试因单卡跳过，退出码 0，总时间 3.77 秒。第 27 章不在根 CTest 中，以上 Python 检查是独立实测。未测试其它 PyTorch 版本、其它 GPU 架构或打包分发。
