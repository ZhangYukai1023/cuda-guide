#include <cuda_runtime.h>
#if GUIDE_HAS_CUBLAS
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
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
#define BLAS_CHECK(call) do { \
    const cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d: %s: cuBLAS status %d\n", __FILE__, __LINE__, #call, static_cast<int>(s)); \
        throw std::runtime_error("cuBLAS failure"); \
    } \
} while (false)
#endif

constexpr int kTile = 16;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T))); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

class Events {
public:
    Events() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~Events() { if (start_) cudaEventDestroy(start_); if (stop_) cudaEventDestroy(stop_); }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    double stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
};

#if GUIDE_HAS_CUBLAS
class BlasHandle {
public:
    BlasHandle() { BLAS_CHECK(cublasCreate(&handle_)); }
    ~BlasHandle() { if (handle_) cublasDestroy(handle_); }
    cublasHandle_t get() const { return handle_; }
private:
    cublasHandle_t handle_ = nullptr;
};
using BlasHandleValue = cublasHandle_t;
#else
using BlasHandleValue = void*;
#endif

__global__ void matmul_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0;
    for (int t = 0; t < k; ++t) sum += a[row * k + t] * b[t * n + col];
    c[row * n + col] = sum;
}

__global__ void matmul_tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[kTile][kTile], bs[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float sum = 0;
    for (int base = 0; base < k; base += kTile) {
        const int ai = base + threadIdx.x;
        const int bi = base + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = row < m && ai < k ? a[row * k + ai] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = bi < k && col < n ? b[bi * n + col] : 0.0f;
        __syncthreads();
        for (int t = 0; t < kTile; ++t) sum += as[threadIdx.y][t] * bs[t][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = sum;
}

__global__ void matmul_register2(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[kTile][kTile], bs[kTile][2 * kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col0 = blockIdx.x * (2 * kTile) + 2 * threadIdx.x;
    const int col1 = col0 + 1;
    float sum0 = 0, sum1 = 0;
    for (int base = 0; base < k; base += kTile) {
        const int ai = base + threadIdx.x;
        const int bi = base + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = row < m && ai < k ? a[row * k + ai] : 0.0f;
        bs[threadIdx.y][2 * threadIdx.x] = bi < k && col0 < n ? b[bi * n + col0] : 0.0f;
        bs[threadIdx.y][2 * threadIdx.x + 1] = bi < k && col1 < n ? b[bi * n + col1] : 0.0f;
        __syncthreads();
        for (int t = 0; t < kTile; ++t) {
            const float av = as[threadIdx.y][t];
            sum0 += av * bs[t][2 * threadIdx.x];
            sum1 += av * bs[t][2 * threadIdx.x + 1];
        }
        __syncthreads();
    }
    if (row < m && col0 < n) c[row * n + col0] = sum0;
    if (row < m && col1 < n) c[row * n + col1] = sum1;
}

struct Shape { int m, n, k; };

void launch(int variant, const float* a, const float* b, float* c, Shape s, BlasHandleValue handle) {
    const dim3 threads(kTile, kTile);
    const dim3 normal_grid((s.n + kTile - 1) / kTile, (s.m + kTile - 1) / kTile);
    if (variant == 0) matmul_naive<<<normal_grid, threads>>>(a, b, c, s.m, s.n, s.k);
    else if (variant == 1) matmul_tiled<<<normal_grid, threads>>>(a, b, c, s.m, s.n, s.k);
    else if (variant == 2) {
        const dim3 wide_grid((s.n + 2 * kTile - 1) / (2 * kTile), normal_grid.y);
        matmul_register2<<<wide_grid, threads>>>(a, b, c, s.m, s.n, s.k);
    } else {
#if GUIDE_HAS_CUBLAS
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major C=A*B is column-major C^T=B^T*A^T in the same storage.
        BLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              s.n, s.m, s.k, &alpha, b, s.n, a, s.k, &beta, c, s.n));
#else
        throw std::logic_error("cuBLAS unavailable");
#endif
    }
    if (variant < 3) CUDA_CHECK(cudaGetLastError());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
}

void run_shape(Shape s) {
    const std::size_t asize = static_cast<std::size_t>(s.m) * s.k;
    const std::size_t bsize = static_cast<std::size_t>(s.k) * s.n;
    const std::size_t csize = static_cast<std::size_t>(s.m) * s.n;
    std::vector<float> a(asize), b(bsize), actual(csize);
    std::vector<double> reference(csize, 0.0);
    for (std::size_t i = 0; i < asize; ++i) a[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    for (std::size_t i = 0; i < bsize; ++i) b[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 16.0f;
    for (int row = 0; row < s.m; ++row)
        for (int col = 0; col < s.n; ++col)
            for (int t = 0; t < s.k; ++t)
                reference[static_cast<std::size_t>(row) * s.n + col] +=
                    static_cast<double>(a[static_cast<std::size_t>(row) * s.k + t]) *
                    static_cast<double>(b[static_cast<std::size_t>(t) * s.n + col]);
    DeviceBuffer<float> da(asize), db(bsize), dc(csize);
    CUDA_CHECK(cudaMemcpy(da.get(), a.data(), asize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.get(), b.data(), bsize * sizeof(float), cudaMemcpyHostToDevice));
#if GUIDE_HAS_CUBLAS
    BlasHandle handle;
    BLAS_CHECK(cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_HOST));
    BlasHandleValue handle_value = handle.get();
#else
    BlasHandleValue handle_value = nullptr;
#endif
    Events events;
    const char* names[] = {"naive", "shared_tile", "two_register_outputs", "cublas_sgemm"};
    for (int variant = 0; variant < 4; ++variant) {
        if (variant == 3 && !GUIDE_HAS_CUBLAS) {
            std::printf("M=%d N=%d K=%d variant=cublas_sgemm SKIP: cuBLAS development library unavailable\n",
                        s.m, s.n, s.k);
            continue;
        }
        for (int i = 0; i < 2; ++i) launch(variant, da.get(), db.get(), dc.get(), s, handle_value);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> samples;
        for (int i = 0; i < 10; ++i) {
            events.start();
            launch(variant, da.get(), db.get(), dc.get(), s, handle_value);
            samples.push_back(events.stop());
        }
        CUDA_CHECK(cudaMemcpy(actual.data(), dc.get(), csize * sizeof(float), cudaMemcpyDeviceToHost));
        std::size_t mismatches = 0;
        double max_abs_error = 0;
        for (std::size_t i = 0; i < csize; ++i) {
            const double error = std::abs(static_cast<double>(actual[i]) - reference[i]);
            max_abs_error = std::max(max_abs_error, error);
            mismatches += !(error <= 1e-4 + 1e-4 * std::abs(reference[i]));
        }
        std::printf("M=%d N=%d K=%d variant=%s median_ms=%.4f max_abs_error=%.8g mismatches=%zu %s\n",
                    s.m, s.n, s.k, names[variant], median(samples), max_abs_error,
                    mismatches, mismatches ? "FAIL" : "PASS");
        if (mismatches) throw std::runtime_error("matmul CPU/GPU mismatch");
    }
}

int main() {
    try {
        for (Shape s : {Shape{2, 2, 3}, Shape{65, 37, 19}, Shape{256, 256, 256}}) run_shape(s);
        std::puts("chapter 16 matmul comparison: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 16 matmul comparison: FAIL: %s\n", e.what());
        return 1;
    }
}
