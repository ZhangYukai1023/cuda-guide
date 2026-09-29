# 附录 D：常见错误与排查入口

先保留最初失败的命令、完整 stderr、退出码和环境，再按“构建→启动→内存/同步→数值→性能”的顺序缩小范围。改动一项就复测最小样本；不要一边修改边丢弃原始记录。若是远程 SSH 失败，先确认是否连到了主机，不能把本机静态检查写成远程 GPU 通过。

| 症状 | 首要检查 | 可执行的下一步 |
| --- | --- | --- |
| `nvcc` 不存在 | PATH、Toolkit 安装位置 | 用绝对 `CMAKE_CUDA_COMPILER`；记录版本，不擅自安装/升级驱动 |
| 不支持 `sm_XX` | nvcc 版本、GPU compute capability、CMake 架构参数 | 在设备/Toolkit 都支持的目标间选择并重新配置 build 目录 |
| `undefined reference` 到 CUDA 库 | CMake `target_link_libraries`、Toolkit 查找结果 | 用对应 `CUDA::...` imported target，保留链接完整命令 |
| 可选头文件不存在（MPI/NCCL/CUTLASS） | 该章是否定义了可选/跳过条件 | 只构建可用目标并记录缺项；不把“未构建”算通过 |
| `invalid device ordinal` | `cudaGetDeviceCount`、`CUDA_VISIBLE_DEVICES`、本地 rank 映射 | 打印逻辑/物理设备，按进程可见列表选设备 |
| `invalid configuration argument` | grid/block 维度、动态 shared 字节、设备限制 | 先用最小尺寸和合法 block，检查 launch 后错误 |
| `illegal memory access` | 尾块、pitch/stride、Halo、指针设备归属 | 以小样本和 `compute-sanitizer --tool memcheck` 复现，定位首个出错 kernel |
| 偶现错值 | Stream 依赖、缓冲复用、未初始化、跨块同步 | 固定输入/种子，增加同步定位；用 synccheck/racecheck/时间线辅助 |
| 全部数值差一个固定倍数 | FFT IFFT 归一化、平均分母、量化尺度 | 手算最小样本，检查 `N`、`width*height`、255/256 等常数 |
| 仅边缘/尾部不同 | 边界策略、半像素、裁剪、尾块 mask | 分别报告四边/角/内部与最后一个 block 的差异 |
| 结果形状/顺序错 | 行列主序、leading dimension、manifest、MPI Gatherv 偏移 | 用非对称矩阵/不整除尺寸/唯一标记逐位置追踪 |
| NaN/Inf | 除零、exp 溢出、无效输入、未初始化 | 检查首个非有限值位置，Softmax 减最大值，N-body 软化 |
| 程序挂起 | `__syncthreads` 分支、MPI 标签/顺序、NCCL rank 不一致 | 全 rank 保存日志并设置作业超时，检查谁先未到 barrier/collective |
| GPU “更慢” | 计时边界、数据量、Host 传输、启动、预热 | 先对齐问题/精度，分开 kernel 与端到端，多轮重复 |

## D.1 复现顺序

1. **固定输入。** 保留原文件/哈希或固定 seed 和生成代码；把失败缩到几行、几列或一个 tile。
2. **检查约定。** 列出 shape、stride、dtype、布局、边界、舍入、输出裁剪、精度和 Stream。CPU 对照必须做同一算法语义。
3. **确认最早错误。** API 返回值立即检查；kernel launch 后检查即时错误，再同步检查执行错误。首次错误后若设备上下文处于错误状态，后续调用信息可能是连锁反应。
4. **隔离阶段。** 对流水线逐段保存中间结果；Softmax/LayerNorm/GEMM、去噪/缩放/归一化、FFT/频域乘法/IFFT 等都应能单独检查。
5. **跑合适工具。** 内存越界用 memcheck；同步问题用 synccheck/racecheck；时间线用 nsys；内核指标用 ncu。工具不可用就记录未测并保持数值/人工定位结果。
6. **修复后回归。** 先复现样本，再章节独立 CTest，最后根工程全量 CTest；更新验证记录和对应 Git 提交。只要依赖或硬件条件缺失，就保留未验证状态。

## D.2 差异如何读

把错误分为：值域（NaN/Inf/越界）、形状/位置、数值小误差、统计偏差、性能差异。逐字节输出不一致若仅在允许的浮点舍入范围内，可能是优化或指令融合造成；仍要报最大差异和超阈值个数。Monte Carlo `π̂` 与 π 的偏差先看标准误差，同一样本 CPU/GPU 命中数不同才是确定性实现故障。CG 等迭代算法要分别看递推停止残差与重新计算的真残差；“迭代结束”不等于“收敛”。

## D.3 远程与协作记录

远程仓库的 Git 分支/工作区、`.codegraph/` 是否存在、无关文件与既有改动要先核对；存在 `.codegraph/` 时先用它定位代码。SSH DNS 失败、TCP 超时、认证失败、命令退出非零是不同故障层次。DNS/SSH 未建立时不能读取远程当前 Git 状态，也不能声称已在 zyk 编译。离线草稿应保存在独立目录，写明未验证；连接恢复后按章同步、独立编译运行、根工程测试、更新覆盖表与验证表，再提交。
