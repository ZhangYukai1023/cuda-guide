#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 第 4 章解释此处的错误处理与不可复制资源对象。
inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 " " + expr + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t n) : count(n) {
        if (n) CUDA_CHECK(cudaMalloc(&data, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& v) {
        if (v.size() != count) throw std::runtime_error("upload size mismatch");
        if (count) CUDA_CHECK(cudaMemcpy(data, v.data(), count*sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> v(count);
        if (count) CUDA_CHECK(cudaMemcpy(v.data(), data, count*sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
struct Stream {
    cudaStream_t value{};
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
template<class T> struct PinnedBuffer {
    T* data = nullptr;
    explicit PinnedBuffer(size_t n) { CUDA_CHECK(cudaMallocHost(&data, n*sizeof(T))); }
    ~PinnedBuffer() { cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};
template<class T, class U>
void verify(const char* name, const std::vector<T>& got, const std::vector<U>& ref,
            double atol = 0, double rtol = 0) {
    if (got.size() != ref.size()) throw std::runtime_error("reference size mismatch");
    double max_error = 0;
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double a = static_cast<double>(got[i]), b = static_cast<double>(ref[i]);
        double e = std::abs(a-b);
        if (!std::isfinite(a) || !std::isfinite(b) || e > atol + rtol*std::abs(b)) ++bad;
        max_error = std::max(max_error, e);
    }
    std::printf("%s: n=%zu max_abs_error=%.9g mismatches=%zu %s\n",
                name, got.size(), max_error, bad, bad ? "FAIL" : "PASS");
    if (bad) throw std::runtime_error(std::string(name) + " verification failed");
}
template<class F> int guarded(F body) {
    try { body(); return EXIT_SUCCESS; }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return EXIT_FAILURE; }
}
