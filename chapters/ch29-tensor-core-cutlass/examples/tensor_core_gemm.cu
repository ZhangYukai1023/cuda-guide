#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#if GUIDE_HAS_CUBLAS
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)
#if GUIDE_HAS_CUBLAS
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d %s: cuBLAS status %d\n", __FILE__, __LINE__, #call, int(s)); \
        throw std::runtime_error("cuBLAS failure"); \
    } \
} while (false)
#endif

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

#if GUIDE_HAS_CUBLAS
class CublasHandle {
public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~CublasHandle() { if (handle_) cublasDestroy(handle_); }
    cublasHandle_t get() const { return handle_; }
private:
    cublasHandle_t handle_ = nullptr;
};
#endif

__global__ void naive_gemm(const half* a, const half* b, const float* bias, float* output,
                           int m, int n, int k, int kp, int np) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float sum = 0.f;
    for (int p = 0; p < k; ++p)
        sum += __half2float(a[row * kp + p]) * __half2float(b[p * np + col]);
    const float value = sum + bias[col];
    output[row * np + col] = value > 0.f ? value : 0.f;
}

__global__ void wmma_gemm_bias_relu(const half* a, const half* b, const float* bias,
                                    float* output, int m, int n, int kp, int np) {
    using namespace nvcuda;
    const int row_base = blockIdx.y * 16, col_base = blockIdx.x * 16;
    const int lane = threadIdx.x;
    __shared__ __align__(32) float tile[16 * 16];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> afrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bfrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
    wmma::fill_fragment(cfrag, 0.f);
    for (int p = 0; p < kp; p += 16) {
        wmma::load_matrix_sync(afrag, a + row_base * kp + p, kp);
        wmma::load_matrix_sync(bfrag, b + p * np + col_base, np);
        wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
    }
    wmma::store_matrix_sync(tile, cfrag, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        const int row = row_base + i / 16, col = col_base + i % 16;
        if (row < m && col < n) {
            const float value = tile[i] + bias[col];
            output[row * np + col] = value > 0.f ? value : 0.f;
        }
    }
}

__global__ void bias_relu(float* matrix, const float* bias, int m, int n, int np) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    const int row = i / n, col = i % n;
    const float value = matrix[row * np + col] + bias[col];
    matrix[row * np + col] = value > 0.f ? value : 0.f;
}

int padded16(int x) { return (x + 15) / 16 * 16; }

std::vector<float> cpu_reference(const std::vector<half>& a, const std::vector<half>& b,
                                 const std::vector<float>& bias, int m, int n, int k, int kp, int np) {
    std::vector<float> expected(static_cast<std::size_t>(padded16(m)) * np, 0.f);
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            double sum = 0;
            for (int p = 0; p < k; ++p)
                sum += static_cast<double>(__half2float(a[row * kp + p])) *
                       __half2float(b[p * np + col]);
            const double value = sum + bias[col];
            expected[row * np + col] = static_cast<float>(value > 0 ? value : 0);
        }
    return expected;
}

void check(const char* name, const std::vector<float>& actual, const std::vector<float>& expected,
           int m, int n, int np, float kernel_ms) {
    double max_error = 0;
    int bad = 0;
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            const int index = row * np + col;
            const double error = std::fabs(static_cast<double>(actual[index]) - expected[index]);
            max_error = std::max(max_error, error);
            bad += !std::isfinite(actual[index]) || error > 0.02 + 0.002 * std::fabs(expected[index]);
        }
    std::printf("%s M=%d N=%d max_abs_error=%.7g bad=%d kernel_ms=%.4f %s\n",
                name, m, n, max_error, bad, static_cast<double>(kernel_ms), bad ? "FAIL" : "PASS");
    if (bad) throw std::runtime_error(std::string(name) + " CPU reference mismatch");
}

template <typename Launch>
float timed_launch(Launch launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed;
}

void run_shape(int m, int n, int k) {
    const int mp = padded16(m), np = padded16(n), kp = padded16(k);
    std::vector<half> a(static_cast<std::size_t>(mp) * kp), b(static_cast<std::size_t>(kp) * np);
    std::fill(a.begin(), a.end(), __float2half_rn(0.f));
    std::fill(b.begin(), b.end(), __float2half_rn(0.f));
    std::vector<float> bias(np, 0.f);
    for (int row = 0; row < m; ++row)
        for (int p = 0; p < k; ++p)
            a[row * kp + p] = __float2half_rn(((row * 7 + p * 3) % 13 - 6) / 7.f);
    for (int p = 0; p < k; ++p)
        for (int col = 0; col < n; ++col)
            b[p * np + col] = __float2half_rn(((p * 5 + col * 11) % 17 - 8) / 9.f);
    for (int col = 0; col < n; ++col) bias[col] = (col % 7 - 3) * 0.1f;
    const auto expected = cpu_reference(a, b, bias, m, n, k, kp, np);
    DeviceBuffer<half> da(a.size()), db(b.size());
    DeviceBuffer<float> dbias(bias.size()), dout(expected.size());
    CUDA_CHECK(cudaMemcpy(da.get(), a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.get(), b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbias.get(), bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto download_check = [&](const char* name, float elapsed) {
        std::vector<float> actual(expected.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(name, actual, expected, m, n, np, elapsed);
    };
    float elapsed = timed_launch([&] {
        naive_gemm<<<dim3((n + 15) / 16, (m + 15) / 16), dim3(16,16)>>>(
            da.get(), db.get(), dbias.get(), dout.get(), m, n, k, kp, np);
    });
    download_check("naive_half_f32_bias_relu", elapsed);
    elapsed = timed_launch([&] {
        wmma_gemm_bias_relu<<<dim3(np / 16, mp / 16), 32>>>(
            da.get(), db.get(), dbias.get(), dout.get(), m, n, kp, np);
    });
    download_check("wmma_half_f32_bias_relu", elapsed);
#if GUIDE_HAS_CUBLAS
    CublasHandle handle;
    const float alpha = 1.f, beta = 0.f;
    elapsed = timed_launch([&] {
        // Row-major C = A*B maps to column-major C^T = B^T*A^T.
        CUBLAS_CHECK(cublasGemmEx(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                                 np, mp, kp, &alpha,
                                 db.get(), CUDA_R_16F, np,
                                 da.get(), CUDA_R_16F, kp,
                                 &beta, dout.get(), CUDA_R_32F, np,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        bias_relu<<<(m * n + 255) / 256, 256>>>(dout.get(), dbias.get(), m, n, np);
    });
    download_check("cublas_gemmex_plus_bias_relu", elapsed);
#else
    std::puts("cublas_gemmex_plus_bias_relu: SKIP (cuBLAS development library unavailable)");
#endif
    std::printf("shape M=%d N=%d K=%d padded=%dx%dx%d PASS\n", m, n, k, mp, np, kp);
}

int main() {
    try {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 7) {
            std::puts("WMMA requires compute capability >= 7.0: SKIP");
            return 77;
        }
        run_shape(16, 16, 16);
        run_shape(19, 23, 37);
        run_shape(64, 64, 64);
        std::puts("chapter 29 tensor core GEMM: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 29 tensor core GEMM: FAIL: %s\n", e.what());
        return 1;
    }
}
