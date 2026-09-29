# 附录 E：架构功能与环境版本记录

**计算能力（compute capability）是设备硬件特征编号，CUDA Toolkit 版本是软件版本；二者不能互换。** 同一程序能否用某功能，还取决于 nvcc/库支持的编译目标、生成的二进制或 JIT 代码、实际设备和运行配置。本表是阅读本书源码的门控入口，正式实验应以当前 [NVIDIA 计算能力与特性表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)及本机查询为准。

| 功能 | 教学示例中的门槛/检查 | 对应章节 | 不满足时 |
| --- | --- | --- | --- |
| 通用 kernel、全局/共享内存与 Stream | 以当前 Toolkit 支持的 GPU 和编译目标为前提，仍要查具体 API | 1—26、33—34、38 | 保留 CPU 参考，CUDA 测试记未运行 |
| cuBLAS/cuSPARSE/cuSOLVER/cuFFT | 库头文件/链接库与设备支持；格式和精度另查库文档 | 11、29、31—32 | 缺库时标记未构建，不以其他库替代同一结果 |
| FP16 WMMA/Tensor Core | 数据类型、Tile 形状、对齐与编译/设备能力一起核对 | 29 | 退回 FP32/普通路径，精度模式单独记录 |
| `cp.async` 小块异步全局→共享拷贝 | `sm_80+`、相应 PTX/二进制目标、地址对齐；运行时双重检查 | 37 | 运行通用 stencil，异步路径 `SKIP` |
| TMA 大块/多维异步拷贝 | 计算能力 9.0 起的特性，需匹配 Toolkit API、张量布局与对齐 | 37 选读 | 本书不提供该路径的可运行验收，标为设计讨论 |
| 线程块集群 | 计算能力 9.0 起，实际可用集群大小还受设备/MIG/资源限制 | 37 选读 | 不启动集群 kernel；保持普通 block 实现 |
| P2P 与双卡执行 | 至少两设备、`cudaDeviceCanAccessPeer`、拓扑和驱动能力 | 35、38D | 使用 Host 中转或对双卡项目明确跳过 |
| MPI/NCCL 多节点 | MPI/NCCL 版本、每 rank GPU、网络和作业调度环境 | 36 | 保留纯 CPU 分区自检，多节点项未验证 |

官方 [异步拷贝说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)区分 `sm_80+` 的 LDGSTS/`cp.async` 与 `sm_90+` 的 TMA；[集群说明](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)说明集群资源约束。高级 ISA 还可能有架构特定目标后缀；“数字更高”不保证所有特殊指令在所有后续设备和编译目标上无条件可用。第 37 章只实际实现 `cp.async` 教学路径，不能把表中其余选读功能写成实测。

## E.1 实验环境模板

```text
日期/时区：
Git 分支、SHA、工作区状态：
GPU 型号、UUID、计算能力、数量、MIG 状态：
驱动版本：
CUDA Toolkit 与 nvcc 版本：
CMake、C++ 编译器版本：
目标架构列表（CMAKE_CUDA_ARCHITECTURES）：
库版本（cuBLAS/cuFFT/NCCL/MPI/PyTorch 等，仅填写用到的）：
CUDA_VISIBLE_DEVICES、容器/作业调度器信息：
编译类型/选项、可选路径是否构建：
命令、退出码、原始日志位置：
```

可用 `nvidia-smi -L`、`nvidia-smi --query-gpu=name,uuid,driver_version --format=csv`、`nvcc --version`、程序中的 `cudaGetDeviceProperties`、`cmake --version` 取得大部分字段。`nvidia-smi` 报告的“CUDA Version”常是驱动支持的最高 CUDA API 版本显示，不等于当前 `nvcc`/Toolkit 版本；两者分别记录。不同节点应分别填表，尤其在多机实验中不能假设 GPU/驱动完全相同。

## E.2 门控判定顺序

1. 确认用户可见的真实设备和拓扑，不从产品名猜架构。
2. 查询 Toolkit/nvcc 是否支持需要的 `sm_XX` 或架构专属目标，并保留编译报告。
3. 检查运行时设备特性与 API/库可用性；需要 P2P、MPI、NCCL 时逐项查询，不以 GPU 数量替代。
4. 先在通用路径上建立 CPU/库参考，架构路径通过相同语义验证后才比较性能。
5. 对缺条件的路径写 `未构建`、`SKIP` 或 `未测`，明确原因；不要用“全章通过”覆盖部分选读路径。
