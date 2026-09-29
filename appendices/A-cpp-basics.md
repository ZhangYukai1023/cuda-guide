# 附录 A：读懂 CUDA 示例所需的 C++

本附录用于查阅第 1—38 章源码中的指针、数组、生命周期、RAII、模板和 CMake 约定，不替代一门完整的 C++ 课程。下面的代码片段是接口说明；全书的 CUDA 示例仍以各章独立 CMake/CTest 和实际设备输出验收。

## A.1 值、地址和连续内存

`std::vector<float> host(n)` 拥有 `n` 个连续 `float`；`host.data()` 是首元素指针，`host.size()*sizeof(float)` 才是传输字节数。`float* p` 本身只保存一个地址，不记录长度或所有权。二维行优先矩阵的 `(row,col)` 元素位于 `p[row*width+col]`；如果每行有字节 pitch，则不能用 `row*width`，要用 `reinterpret_cast<char*>(p)+row*pitch` 再取 `col`。传给 kernel 的设备指针必须指向**该设备上仍存活**的分配；普通 Host 指针不能因为类型同为 `float*` 就在 GPU 中解引用。

`const float* input` 表示函数不能通过此指针改写元素；`float* const p` 表示变量 `p` 不能指向别处，两者不同。参数 `(pointer,length)` 是一对：只传指针而没有尺寸无法验证尾块。`size_t` 适合表示内存字节数，但 kernel 索引若转为 `int`，要先证明元素数没有超出 `int` 范围；`rows*cols*sizeof(T)` 的乘法也应在足够宽的类型中完成并防溢出。

## A.2 所有权和 RAII

RAII（Resource Acquisition Is Initialization）把分配和释放绑定到对象生命周期。`std::vector`/`std::unique_ptr` 管理 Host 内存；CUDA 设备内存、Stream、Event 和库句柄也可分别封装。一个最小设备缓冲示意：

```cpp
template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
    ~DeviceBuffer() noexcept { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return count_; }
private:
    size_t count_;
    T* ptr_ = nullptr;
};
```

删除拷贝操作避免两个对象同时声称拥有同一设备指针并双重释放。若需要把缓冲放进可移动容器，要额外实现移动构造/移动赋值，并在移动后把原指针置空。构造函数在第二项分配失败时，已成功分配的第一项要在 `catch` 中释放，或将每项分别放进已有 RAII 成员。析构函数通常不抛异常；需要知道 `cudaFree` 失败时，应在明确的 `close()` 阶段检查并记录，而不是依赖析构异常。多 GPU 程序还要记录分配所属设备，销毁前设置正确的当前设备。

异步操作改变了“生命周期结束”的时机。`cudaMemcpyAsync`、kernel 和库调用可能在 Host 函数返回后才执行；源、目的、工作区、Stream 与 Event 必须保留到有关操作完成。使用复用双槽时，在覆盖某槽 Host/设备缓冲前先等待该槽的完成事件或 Stream。`cudaDeviceSynchronize()` 易懂但等待整个设备，真实流水线优先用适当的 Stream/Event 保持并行。

## A.3 错误、异常和同步

CUDA Runtime API 返回 `cudaError_t`，cuBLAS/cuFFT/cuSPARSE 等使用各自状态枚举，不能把所有状态交给 `cudaGetErrorString`。Host 包装函数可以把状态翻译成带 API 名称的异常；在 `main` 捕获并返回非零退出码，方便 CTest 判断。kernel launch 是异步的：launch 后的 `cudaGetLastError()` 可发现配置等即时错误；执行时的越界等要在同步或设备到主机拷贝之后才能可靠暴露。调试时保存**第一个失败 API**及输入形状、设备、Stream，不要让后续错误覆盖它。

`new/delete`、`new[]/delete[]`、`malloc/free`、`cudaMalloc/cudaFree` 必须成对匹配。把 Host `new[]` 得到的指针传给 `cudaFree`，或把 CUDA 指针交给 `delete[]`，都是错误。Pinned Host 缓冲由 `cudaMallocHost/cudaFreeHost` 配对；Unified Memory 由其对应 CUDA 分配/释放 API 管理，也不能混用。

## A.4 模板、编译单元与构建

`template<class T>` 使设备缓冲能管理 `float`、`int` 等不同元素类型；模板定义通常要放在调用方能看到的头文件中，除非在 `.cpp/.cu` 中显式实例化。`.cu` 经 nvcc 处理 Host 和 Device 代码；`__global__` 是从 Host 发起的 kernel，`__device__` 只在 GPU 调用，`__host__ __device__` 需要两侧都能编译的函数体。多个 `.cu` 文件若只通过普通 Host 函数相互调用，通常不需要设备代码跨编译单元链接；若跨单元调用 `__device__` 符号，则要配置可分离编译及设备链接。

章节根外的独立构建示例：

```bash
cmake -S chapters/ch33-stencil-heat/examples -B build/ch33 \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/ch33 -j
ctest --test-dir build/ch33 --output-on-failure
```

`-DCMAKE_CUDA_ARCHITECTURES` 应与目标设备和 Toolkit 支持匹配；它不是“跑得更快”的无条件开关。把 build 目录放在源码外，方便保留同一源码不同配置的编译报告。`CMakeLists.txt` 中的 `find_package(CUDAToolkit)` 与 `target_link_libraries(... CUDA::cublas)` 等应明确每个示例的库依赖，而不是靠全局链接器环境碰巧找到。

## A.5 自测与答案

1. `std::vector<float> x(100)` 上传多少字节？答：`100*sizeof(float)`，通常为 400 字节；不要只传 `100`。
2. 为什么复制一个只含裸 `cudaMalloc` 指针的默认类对象危险？答：两个对象会拥有同一指针，可能双重释放和悬空访问；应禁止复制或定义真正的深拷贝。
3. kernel launch 后立刻析构输入缓冲安全吗？答：除非相关 Stream 已完成且没有后续使用，否则不安全；异步 kernel 可能还在读它。
4. 二维图像有 pitch 时，`row*width+col` 还能定位像素吗？答：不能，必须按每行的实际字节 stride/pitch 计算。
5. `cudaGetLastError()` 没报错是否证明 kernel 数值正确？答：不能；仍需同步暴露执行错误，并与独立参考比较输出。
