# 第 1 章验证记录

日期：2026-09-28。工作目录：`/data2/cuda-guide`。主机：`ubuntu2404`。

直接 nvcc 构建设备查询和数组加法均退出 0，无诊断输出。完整构建命令见本章正文。以下是实际运行两个直接编译的程序，随后执行 CMake 配置、构建和 CTest 的输出，整个命令链退出码为 0。

```text
CUDA devices: 1
Device 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0
n=1, mismatches=0, PASS
result: 11 22 33 44 55 66 77 88
n=8, mismatches=0, PASS
n=129, mismatches=0, PASS
n=1000, mismatches=0, PASS
-- The CXX compiler identification is GNU 13.3.0
-- The CUDA compiler identification is NVIDIA 12.8.93
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /home/zhangyukai/.local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Configuring done (0.9s)
-- Generating done (0.0s)
-- Build files have been written to: /data2/cuda-guide/build/ch01-cmake
[ 50%] Building CUDA object CMakeFiles/vector_add.dir/vector_add.cu.o
[ 50%] Building CUDA object CMakeFiles/device_info.dir/device_info.cu.o
[ 75%] Linking CUDA executable device_info
[ 75%] Built target device_info
[100%] Linking CUDA executable vector_add
[100%] Built target vector_add
Internal ctest changing into directory: /data2/cuda-guide/build/ch01-cmake
Test project /data2/cuda-guide/build/ch01-cmake
    Start 1: device_info
1/2 Test #1: device_info ......................   Passed    0.09 sec
    Start 2: vector_add
2/2 Test #2: vector_add .......................   Passed    0.16 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.25 sec
```

结论：两种构建方式成功；数组长度 1、8、129、1000 的 GPU 整数结果与 CPU 逐项完全一致。没有 Compute Sanitizer 结果，没有 kernel 或端到端性能结果。CTest 时间仅为测试进程耗时。初始环境记录中的“尚未编译”描述预检查阶段，当前状态以本记录为准。
