# 第 11 章验证记录

2026-09-28（Asia/Shanghai）在 zyk 的 `codex/align-outline-38` 分支验证。RTX 5060 Ti，计算能力 12.0；nvcc 12.8.93，目标 `sm_120`。从本机草稿按章同步前远程目录不存在，`.vscode/`、`README.html` 未触及。

**依赖缺口**：初次独立 CMake 配置因不存在 `CUDA::cublas` 目标失败。已检查 Toolkit、常见本机库目录和 `ldconfig`，没有发现 `cublas_v2.h` 或 `libcublas.so`；未安装或升级任何驱动、Toolkit 或库。CMake 改为仅在 `CUDA::cublas` 可用时编译 SGEMM，缺失时显式打印 `SKIP`。初次使用动态 `CUDA::cudart` 构建后，程序因找不到 `libcudart.so.12` 无法启动；改为当前 Toolkit 自带的 `CUDA::cudart_static` 后重构建、CTest 和手动运行成功。前述失败都没有计入通过。

独立目录 `build/ch11-library-standalone` 配置、构建成功；CTest `ch11_library_baselines` 1/1 通过。手动运行输出：

```text
Thrust sort n=7: PASS
CUB sum n=7: PASS value=-35 temp_bytes=1
CUB sum n=1003: PASS value=0 temp_bytes=1
SKIP cuBLAS SGEMM: development library unavailable
chapter 11 Thrust/CUB baselines: PASS (cuBLAS SKIP)
```

根 `build/outline` 重新配置、全目标构建成功；`ctest --test-dir build/outline --output-on-failure --timeout 120` 退出码 0：23 项中 22 项通过、0 项失败、1 项旧双卡测试因单卡环境跳过；新增 `ch11_library_baselines` 通过，总时间 4.05 秒。该 CTest 只证明已编译的 Thrust/CUB 路径，不能把其中打印的 cuBLAS `SKIP` 算成 GEMM 验证。

NPP 仅在正文说明适用方向，没有本章代码或性能对照。cuBLAS 开发库可用后，应重新配置、确认 CMake 启用该路径、保存列主序 SGEMM 与 CPU 比较输出并重跑根 CTest。
