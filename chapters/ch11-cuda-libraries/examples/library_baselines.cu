#include <cuda_runtime.h>
#if GUIDE_HAS_CUBLAS
#include <cublas_v2.h>
#endif
#include <cub/device/device_reduce.cuh>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)
#if GUIDE_HAS_CUBLAS
#define CUBLAS_CHECK(call) do { \
    const cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d: %s: cuBLAS status %d\n", __FILE__, __LINE__, #call, static_cast<int>(s)); \
        throw std::runtime_error("cuBLAS failure"); \
    } \
} while (false)
#endif

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T))); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

class DeviceWorkspace {
public:
    explicit DeviceWorkspace(std::size_t bytes) {
        // Some CUB versions still expect a non-null pointer for a zero-byte query.
        CUDA_CHECK(cudaMalloc(&data_, std::max<std::size_t>(bytes, 1)));
    }
    ~DeviceWorkspace() { if (data_) cudaFree(data_); }
    void* get() const { return data_; }
private:
    void* data_ = nullptr;
};

#if GUIDE_HAS_CUBLAS
class BlasHandle {
public:
    BlasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~BlasHandle() { if (handle_) cublasDestroy(handle_); }
    cublasHandle_t get() const { return handle_; }
private:
    cublasHandle_t handle_ = nullptr;
};
#endif

void check_sort() {
    const std::vector<int> input = {7, 1, 7, -2, 5, 0, -2};
    std::vector<int> reference = input;
    std::sort(reference.begin(), reference.end());
    thrust::device_vector<int> data(input.begin(), input.end());
    thrust::sort(data.begin(), data.end());
    std::vector<int> actual(input.size());
    thrust::copy(data.begin(), data.end(), actual.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (actual != reference) throw std::runtime_error("Thrust sort mismatch");
    std::puts("Thrust sort n=7: PASS");
}

void check_reduce(int n) {
    std::vector<int> input(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) input[static_cast<std::size_t>(i)] = i % 17 - 8;
    const int reference = std::accumulate(input.begin(), input.end(), 0);
    DeviceBuffer<int> din(input.size()), dout(1);
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));
    std::size_t bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, din.get(), dout.get(), n));
    DeviceWorkspace workspace(bytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(workspace.get(), bytes, din.get(), dout.get(), n));
    CUDA_CHECK(cudaDeviceSynchronize());
    int actual = 0;
    CUDA_CHECK(cudaMemcpy(&actual, dout.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (actual != reference) throw std::runtime_error("CUB sum mismatch");
    std::printf("CUB sum n=%d: PASS value=%d temp_bytes=%zu\n", n, actual, bytes);
}

#if GUIDE_HAS_CUBLAS
void check_gemm() {
    // Column-major A=[[1,2],[3,4]], B=[[5,6],[7,8]], C=A*B.
    const std::vector<float> a = {1, 3, 2, 4};
    const std::vector<float> b = {5, 7, 6, 8};
    const std::vector<float> reference = {19, 43, 22, 50};
    DeviceBuffer<float> da(4), db(4), dc(4);
    CUDA_CHECK(cudaMemcpy(da.get(), a.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.get(), b.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
    BlasHandle handle;
    CUBLAS_CHECK(cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_HOST));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                            2, 2, 2, &alpha, da.get(), 2, db.get(), 2,
                            &beta, dc.get(), 2));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(4);
    CUDA_CHECK(cudaMemcpy(actual.data(), dc.get(), 4 * sizeof(float), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < 4; ++i)
        if (std::fabs(actual[i] - reference[i]) > 1e-5f)
            throw std::runtime_error("cuBLAS GEMM mismatch");
    std::puts("cuBLAS SGEMM 2x2 column-major: PASS");
}
#endif

int main() {
    try {
        check_sort();
        check_reduce(7);
        check_reduce(1003);
#if GUIDE_HAS_CUBLAS
        check_gemm();
        std::puts("chapter 11 library baselines: PASS");
#else
        std::puts("SKIP cuBLAS SGEMM: development library unavailable");
        std::puts("chapter 11 Thrust/CUB baselines: PASS (cuBLAS SKIP)");
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 11 library baselines: FAIL: %s\n", e.what());
        return 1;
    }
}
