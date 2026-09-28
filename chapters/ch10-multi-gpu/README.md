# 第 10 章：多 GPU：任务划分与结果汇总

[全书导航](../../README.md) · [上一章](../ch09-scientific-computing/README.md)

## 1. 先把任务拆对，再考虑通信

有 10 个整数，要计算 y=3x+1。两块 GPU 可各处理一段：

```text
GPU 0：下标 [0,5)
GPU 1：下标 [5,10)
CPU：按原来的下标收集结果
```

区间用左闭右开形式，长度为 end-begin。一般写成 begin=n*d/devices、end=n*(d+1)/devices。n=17、devices=2 时得到 [0,8) 和 [8,17)，没有遗漏也没有重复。

最小示例先在一块 GPU 上使用同样的分片框架。递进路径在有至少两块可见设备时使用前两块并分别提交异步工作。本机只有一块 RTX 5060 Ti，所以单卡回退已验证，真正双卡执行尚未验证。

## 2. 一块 GPU 对应一组资源

设备指针、stream 和相关资源都有设备归属。本例在创建分片对象之前 cudaSetDevice(d)，随后申请该分片的设备内存、流和页锁定输入输出。

每片执行：

```text
选择设备 → H2D → kernel → D2H
```

CPU 先把各片工作都提交，再逐片等待与收集，给设备并发留下可能性。调用 cudaSetDevice 只是切换主机线程后续操作面向的设备，不是把已有内存迁移过去。

本例不需要 P2P：输入由主机分别送入，输出由主机汇总。这样减少了第一份多设备代码的拓扑假设，但传输与汇总成本也必须在后续性能评估中考虑。

析构前同样切回资源所属设备。第 4 章的轻量 DeviceBuffer 不保存设备编号，所以不能任意在另一当前设备下销毁它。本章 Shard 记录 device，清理逻辑负责切换和等待。异常路径采用尽力清理；原始失败由主检查返回。

## 3. 完整性检查

CPU 参考直接按全局下标算出 3*i+1。每个分片输入也使用全局下标，避免把每片从 0 开始的局部编号当成原始输入。收集时把结果放回 got[begin:begin+count]。

测试长度为 1、17、1003。长度 1 在两卡分片时会出现空片，源码跳过它，不申请零长度页锁定缓冲区，也不启动空任务。当前只有单卡，不能声称已经实测这个双卡空片分支。

命令行支持：

```bash
./build/all/ch10/multi_gpu
./build/all/ch10/multi_gpu --require-two
```

第一条允许单卡回退；第二条要求至少两块设备，否则返回 77 并打印 SKIP。CMake 将 77 显式配置为“跳过”，不是“通过”。本机输出为 Single-device fallback only; multi-device path NOT verified。

## 4. 从独立任务走向有依赖的多 GPU

本章的逐元素变换不需要分片间通信。前面章节的其他算法不能一概照搬：

| 算法 | 可以怎样划分 | 额外依赖 |
| --- | --- | --- |
| 独立图像帧 | 每卡一组帧 | 汇总输出与负载平衡 |
| 图像 3×3 滤波 | 按行条带 | 上下各一行邻域 halo |
| 热扩散 | 按子网格 | 每个时间步更新 halo |
| 矩阵乘法 | 按输出行 | 每卡需相应 A 行及完整或分块 B |
| 多卡归约 | 每卡先算局部值 | 合并局部结果，处理浮点顺序 |

如果把第 9 章的网格切成两段，各卡把内部交界误当成固定零边界，得到的就是另一个物理问题。多 GPU 的难点往往不在启动两次 kernel，而在正确传递这些依赖。

P2P 允许某些设备间直接访问或复制，但必须查询设备对能力、拓扑与运行条件，不能因为都有 CUDA 就假定可用。NCCL 或 MPI 是进一步处理通信的工具；本项目没有安装、调用或验证它们。本章也没有声称 P2P、NVLink 或集合通信可用。设备选择与多设备语义可参阅 [CUDA 多设备系统说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#multi-device-system)。

## 5. 如何诚实地扩展验证

有第二块 GPU 后，先运行 --require-two 并保存新日志，再检查：

1. 两片均得到正确结果，包括奇数长度和 n 小于设备数。
2. 设备编号与可见设备映射一致。
3. 资源在正确设备上创建、同步和销毁。
4. 如果加入通信，CPU 或单卡参考仍逐项通过。
5. 性能测量等待所有参与设备，包含明确的输入、传输和汇总范围。

跨设备 CUDA event 不能直接拿来拼成一段统一时间。端到端可使用主机时钟，开始前和结束前等待所有参与设备；每卡内部可分别用该卡事件测量。必须预热和重复，且不要把单卡回退结果当成多卡扩展效率。

## 6. 常见错误

- 当前设备选错，资源创建或操作归属混乱。
- 把 device 0 的普通设备指针直接传给 device 1 而不确认可访问性。
- 所有分片从输入 0 开始，汇总看起来完整却重复数据。
- 主机只等最后一块 GPU，另一个结果还未复制完成。
- 把设备不足的跳过测试统计成双卡通过。
- 假定两卡必然快两倍，忽略串行部分、复制与负载差异。

## 7. 练习与参考答案

1. n=10 分给 3 个设备，区间是什么？
   答：[0,3)、[3,6)、[6,10)。本章程序最多使用两卡，此题仅验证通用分片公式。
2. 3×3 图像滤波按行分片，交界至少需要哪些额外像素？
   答：各自输出范围之外的一行输入邻域；更大半径需要更多 halo。
3. 本机单卡通过能证明双卡流与清理正确吗？
   答：不能，只验证了公共计算和单卡路径。双卡分支必须在真实多设备环境运行。
4. 如果只想比较单卡与双卡计算时间，应怎样避免范围不一致？
   答：保持总问题规模、数据类型、结果验证、预热和重复一致，分别标明纯设备计算与含传输汇总的端到端范围。

到这里，本书第一版完成从首次 CUDA 计算到图像、AI、科学计算和多 GPU 的学习路径。它提供可运行的教学基线；尚未验证的工具和多卡能力继续保留明确状态。

## 构建、运行与实测输出

完整源码：[multi_gpu.cu](examples/multi_gpu.cu)；构建定义：[CMakeLists.txt](examples/CMakeLists.txt)。公共依赖也在本仓库，不需要复制未提供的代码。

在项目根目录执行（首次配置后可重复构建）：

```bash
cd /data2/cuda-guide
cmake -S . -B build/all \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/all --target multi_gpu -j2
./build/all/ch10/multi_gpu
```

也可单独配置本章：把上面 -S 改为 chapters/ch10-multi-gpu/examples，-B 改为 build/ch10-standalone，其他选项不变；程序位于该独立构建目录下。无需第三方图像或数学库。

本次实际输出如下。除计时数据可能随运行波动外，同一固定输入应得到这些结果或在注明容差内一致；最终错误数量应为 0：

```text
sharded_transform: n=1 max_abs_error=0 mismatches=0 PASS
devices_used=1 n=1
sharded_transform: n=17 max_abs_error=0 mismatches=0 PASS
devices_used=1 n=17
sharded_transform: n=1003 max_abs_error=0 mismatches=0 PASS
devices_used=1 n=1003
Single-device fallback only; multi-device path NOT verified.
```

更多环境与限制见 [validation.md](results/validation.md)。本机没有 Compute Sanitizer，不能把 CPU 对照通过等同于内存或同步工具检查通过。
