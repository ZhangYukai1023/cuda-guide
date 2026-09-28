#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

int main() {
    try {
        constexpr int m = 32, n = 32, k = 32;
        std::vector<float> a(m * k), b(k * n), expected(m * n), actual(m * n);
        for (int i = 0; i < m * k; ++i) a[i] = static_cast<float>((i * 7) % 13 - 6) / 7.f;
        for (int i = 0; i < k * n; ++i) b[i] = static_cast<float>((i * 11) % 17 - 8) / 9.f;
        for (int row = 0; row < m; ++row)
            for (int col = 0; col < n; ++col) {
                double sum = 0;
                for (int p = 0; p < k; ++p) sum += static_cast<double>(a[row * k + p]) * b[p * n + col];
                expected[row * n + col] = static_cast<float>(sum);
            }
        DeviceBuffer<float> da(a.size()), db(b.size()), dc(actual.size());
        CUDA_CHECK(cudaMemcpy(da.get(), a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.get(), b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
        using Gemm = cutlass::gemm::device::Gemm<
            float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor>;
        Gemm gemm;
        Gemm::Arguments arguments(
            {m, n, k}, {da.get(), k}, {db.get(), n},
            {dc.get(), n}, {dc.get(), n}, {1.f, 0.f});
        const cutlass::Status status = gemm(arguments);
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error("CUTLASS GEMM returned non-success status");
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), dc.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double max_error = 0;
        int bad = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
            max_error = std::max(max_error, error);
            bad += !std::isfinite(actual[i]) || error > 1e-4;
        }
        std::printf("CUTLASS FP32 row-major GEMM 32x32x32 max_abs_error=%.8g bad=%d %s\n",
                    max_error, bad, bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CUTLASS GEMM: FAIL: %s\n", e.what());
        return 1;
    }
}
