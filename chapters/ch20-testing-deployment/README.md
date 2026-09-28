# 第 20 章：测试、部署与性能回归

[全书导航](../../README.md) · [原始大纲](../../outline.md) · [上一章](../ch19-cuda-project/README.md)

第 19 章提供可复用静态库，本章把“在本机能跑”变成可核查的交付记录。示例 [release_check.py](examples/release_check.py) 不改驱动或 CUDA 安装；它对已有构建目录执行构建、全量 CTest、可选的第 10 章安全模式 memcheck 和第 12 章基准程序，保存环境、命令、退出码与原始输出到 JSON。脚本已在 zyk 对现有 CUDA 工程运行：构建、全量 CTest 与基准采集路径通过；必需 memcheck 因工具缺失按设计报告失败。

## 1. 正确性、工具检查与性能各自留证据

交付时先明确输入和结果约定，然后检查边界长度、非整齐尺寸、空输入、特殊浮点和库对照。第 10 章的故障模式是故意失败的教学材料，不进入普通成功门禁；本脚本可把 `debug_cases safe` 放进 memcheck，要求安全模式无访存错误。全量 CTest 是正确性回归，可能因设备条件而有明确 SKIP，必须查看测试名和原因，不能只看总退出码。

性能回归以第 12 章 `benchmark_vector` 的 `wall_ms` 中位数为例。若提供上一份 JSON，脚本只在 GPU/驱动查询文本和 CMake 编译设置一致时比较共同输入尺寸；当前值超过基线的 `1.15` 倍便标记 `REVIEW`。这是待人工复核的信号，**不直接判定失败**：先看原始样本、设备共享负载、时钟、温度、重复运行和任务边界。两台机器或不同架构/驱动上的数字不可直接用这一阈值比较。若有业务吞吐、尾延迟或能耗目标，应另建对应基准。

脚本把构建、CTest、请求的基准与 memcheck、GPU 和 nvcc 探测视为必需步骤；缺工具、超时、没有找到 CTest、没有解析到所请求的基准指标都会使报告 `ok=false`。未请求的可选项不会凭空写成通过。报告中包含命令、stdout/stderr、超时和退出码，便于复现失败；不要把含敏感业务输入的日志直接公开。

## 2. 从空白环境到一份报告

在仓库根目录，要求完整 memcheck 的命令为：

```bash
cmake -S . -B build/outline \
  -DCMAKE_CUDA_COMPILER=/home/zhangyukai/.local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
python3 chapters/ch20-testing-deployment/examples/release_check.py \
  --repo "$PWD" --build-dir "$PWD/build/outline" \
  --benchmark "$PWD/build/outline/book_ch12/benchmark_vector" \
  --sanitizer-binary "$PWD/build/outline/book_ch10/debug_cases" \
  --out "$PWD/build/reports/release-zyk.json"
```

本机实测缺少 Compute Sanitizer，因此上述带 `--sanitizer-binary` 的命令会生成 `status=FAIL` 报告。仅检查已具备的构建、CTest 与基准采集时，省略该参数可生成 `status=PASS`，但 PASS 不覆盖 memcheck。首次报告在确认环境安静、正确性通过且原始样本可信后，才可由维护者保存为基线。下一次传 `--baseline-report /path/to/verified-baseline.json`。脚本不会自行覆盖基线，也不会因为某一次较慢就自动修改阈值。若只需检查构建与 CTest，可省略两个可选二进制参数。报告保留 `git rev-parse HEAD` 与 `git status --short`，方便区分提交版和含本地变更版。

## 3. 架构、驱动和依赖兼容性

构建参数 `CMAKE_CUDA_ARCHITECTURES=120` 是本仓库在 zyk 当前 GPU 上的示例目标，不等于“能部署到所有 NVIDIA GPU”。交付前应列出目标设备的计算能力、实际生成的 cubin/PTX、CUDA Toolkit、驱动、cuBLAS 等运行依赖，并在**第二台真实目标环境**上重建或安装后运行同一正确性检查。需要时可用工具包的 `cuobjdump --list` 检查可执行文件中的设备代码，结合目标架构与运行日志判定是否走 JIT。不要因为某机 `nvidia-smi` 显示一个 CUDA 版本，就推断另一机一定支持所有新 API。NVIDIA 的 [CUDA 兼容性文档](https://docs.nvidia.com/deploy/cuda-compatibility/latest/)解释驱动、二进制与 PTX JIT 的限制；具体目标以实际发布版本说明和测试为准。

容器可以固定用户态库和构建工具，但宿主驱动、GPU 设备及权限仍是部署条件。交付包至少写清启动命令、所需设备/驱动范围、环境变量、库依赖、数据目录和日志位置。图像和 AI 篇还会引入额外库，需在该模块自己的交付清单中列出。没有第二环境时，报告明确“未验证跨机器部署”，不能凭本机通过改写为已部署。

## 4. 可复现故障记录

一次失败记录至少回答：用了哪个 Git 提交、CMake 架构与构建类型、GPU/驱动/nvcc 版本、输入和测试名、首条关键错误、最小复现命令、修复内容以及修复后的同口径验证。脚本的 JSON 提供多数原始证据；若是数值或性能失败，还须补上输入种子、容差、样本分布与其它 GPU 负载。建议把报告保存在 `build/reports/` 这类构建输出目录，避免把大日志误当正文提交。

## 5. 常见错误

- 只执行某一个 happy path，就写“通过全量回归”。
- 把“CTest 返回 0”与“所有设备条件测试都实际运行”混同，不查看 SKIP。
- 基线来自不同 GPU、驱动或编译选项，却直接比较百分比。
- 一次慢样本就自动判为性能回归，忽略噪声与共享 GPU 负载。
- 构建只生成当前 GPU 的设备代码，却声称在未知架构上兼容。
- 容器里 CUDA 用户态库可用，就认为宿主驱动与 GPU 权限也必然满足。
- `compute-sanitizer` 不在 PATH 时把未执行的检查写成 PASS。

## 6. 练习与参考答案

1. 只有本机 zyk 的报告，能否写“多型号部署通过”？答：不能；还需要另一目标型号上的实际安装/运行证据。
2. 基线墙钟 0.50 ms、当前 0.58 ms，默认阈值 1.15 下结果如何？答：比值 1.16，标记 `REVIEW`，再复测与调查；这本身不是最终回归判定。
3. CTest 因只有一块 GPU 跳过多 GPU 用例，应怎样记录？答：保留用例名、跳过原因和其余测试结果；在有足够 GPU 的环境补测。
4. 一份性能记录为什么要包括编译架构与驱动？答：它们可能改变设备代码、JIT 和运行特征，是比较与部署兼容性的必要背景。
5. 如果 memcheck 工具不存在，能否在交付报告中省略该项而写全部通过？答：若它是交付要求，应报告工具不可用并使该门禁未通过；如确属可选，须明确未执行而不是伪造 PASS。

[下一章：像素、通道与布局](../ch21-image-layout/README.md)进入图像处理，沿用本章的环境记录、正确性和性能证据格式。
