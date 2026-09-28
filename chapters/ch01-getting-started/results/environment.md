# 第 1 章环境预检查记录

这是首次检查的历史记录。后续已完成编译运行，最新结果见同目录 [validation.md](validation.md)。原始大纲仍不可用，用户已授权直接编写。

检查时间：2026-09-28 06:25–06:28（Asia/Shanghai）。仅执行只读环境检查，未安装、升级或修改系统配置。

## 工作目录与编写依据

- 当前目录：`/data2`。
- `hostname` 输出：`ubuntu2404`；尚未确认它是否就是用户所称的 zyk 服务器。
- `/data2` 中未找到《CUDA开发指导书-大纲.md》、`AGENTS.md` 或已有 `cuda-guide/`。
- 用户提供的 `/Users/zhangyukai/Documents/开发/CUDA开发指导书-大纲.md` 不存在。
- 对 `/data2`、`/home/zhangyukai`、`/tmp` 的文件名搜索未找到大纲；`/AGENTS.md` 也不存在。
- 因缺少用户指定的编写依据，尚未编写章节正文和示例，也未执行示例编译或运行。

## 实际检查结果

| 项目 | 实际结果 |
| --- | --- |
| 操作系统内核 | Linux 7.0.0-31-generic x86_64 GNU/Linux |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| 显存（nvidia-smi） | 16311 MiB |
| 驱动 | 595.84 |
| nvidia-smi 的 CUDA Version 字段 | 13.2；此字段不作为已安装 Toolkit 版本的证据 |
| GPU compute_cap 查询 | 12.0 |
| CUDA 编译器 | `/home/zhangyukai/.local/cuda/bin/nvcc` |
| nvcc 版本 | release 12.8, V12.8.93 |
| nvcc 构建 | cuda_12.8.r12.8/compiler.35583870_0 |
| 默认 PATH 中的 nvcc | 未找到；直接执行 `nvcc --version` 返回 command not found |
| C++ 编译器 | `/usr/bin/c++`、`/usr/bin/g++`；Ubuntu 13.3.0-6ubuntu2~24.04.1，GCC 13.3.0 |
| CMake | `/usr/bin/cmake`，3.28.3 |
| GDB | `/usr/bin/gdb`，Ubuntu 15.0.50.20240403-0ubuntu1 |
| Compute Sanitizer、cuda-gdb、nsys、ncu | PATH 和上述 CUDA 的 bin 目录中均未找到；额外搜索 `~/.local` 六层内也未找到 compute-sanitizer |

`/home/zhangyukai/.local/cuda` 下存在 `bin`、`include`、`lib64`、`nvvm`。仅凭目录存在尚不能认定编译、链接和运行均正常。

## 关键命令及输出摘录

```text
$ nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
name, driver_version, compute_cap
NVIDIA GeForce RTX 5060 Ti, 595.84, 12.0

$ /home/zhangyukai/.local/cuda/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

`nvcc --list-gpu-code` 实际列出了 `sm_120`。据 GPU 的实际 compute_cap 和编译器支持列表，本机后续示例可选择 `-arch=sm_120`，使用上述绝对路径调用 nvcc。这只是参数选择依据，尚未完成构建验证。

## 待完成

1. 取得大纲，以及用户要求遵守的 AGENTS.md（如有）。
2. 按大纲编写第 1 章及完整示例，实际编译、运行并与 CPU 参考结果比较。
3. 如能定位 Compute Sanitizer，再执行适用的检查；当前没有 sanitizer 验证结果。

本记录没有 GPU 计算结果、性能数据或示例测试通过的结论。
