# 第 4 章验证记录

主机 `ubuntu2404`（本机 SSH 别名 `zyk`）；GPU RTX 5060 Ti，计算能力 12.0；nvcc 12.8.93；目标 `sm_120`。环境详情见 [第 1 章记录](../../ch01-getting-started/results/environment.md)。

项目根目录执行配置、`cmake --build build/outline --target memory_resources -j2` 和 `./build/outline/book_ch04/memory_resources`，退出码均为 0。程序实际输出：

```text
square: n=7 max_abs_error=0 mismatches=0 PASS
square_end_to_end_mean_ms=0.047444 (warmup=5 repeats=20; allocation through free)
reused_buffer: n=1003 max_abs_error=0 mismatches=0 PASS
packed_inputs: n=7 max_abs_error=0 mismatches=0 PASS
```

三个算例均与 CPU 参考逐项一致。端到端计时先预热 5 次，随后对 20 次完整调用取均值，使用 `steady_clock`，每次 kernel 后同步；包括设备分配、传输、输出数组创建与设备释放，不包括输入生成、CPU 参考验证。这个数字只是该次观察，不是 kernel 耗时或加速比。未找到 Compute Sanitizer，因此未运行内存工具检查。

全书 CTest 中 `ch04_memory_resources` 通过；本章单独配置、构建、CTest 也通过（1/1）。
