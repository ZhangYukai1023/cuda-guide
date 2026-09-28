#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
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

using Clock = std::chrono::steady_clock;
constexpr int threads = 256;
constexpr double epsilon = 1e-5;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

__device__ double reduce_sum(double value, double* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }
    return scratch[0];
}

__device__ double reduce_max(double value, double* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) scratch[tid] = fmax(scratch[tid], scratch[tid + offset]);
        __syncthreads();
    }
    return scratch[0];
}

__global__ void softmax_rows(const float* input, float* output, int width) {
    __shared__ double scratch[threads];
    const int row = blockIdx.x, tid = threadIdx.x;
    const float* src = input + static_cast<std::size_t>(row) * width;
    float* dst = output + static_cast<std::size_t>(row) * width;
    double local_max = -INFINITY;
    for (int x = tid; x < width; x += blockDim.x)
        local_max = fmax(local_max, static_cast<double>(src[x]));
    const double maximum = reduce_max(local_max, scratch);
    double local_sum = 0;
    for (int x = tid; x < width; x += blockDim.x)
        local_sum += exp(static_cast<double>(src[x]) - maximum);
    const double denominator = reduce_sum(local_sum, scratch);
    for (int x = tid; x < width; x += blockDim.x)
        dst[x] = static_cast<float>(exp(static_cast<double>(src[x]) - maximum) / denominator);
}

__global__ void layernorm_rows(const float* input, float* output,
                               const float* gamma, const float* beta, int width) {
    __shared__ double scratch[threads];
    const int row = blockIdx.x, tid = threadIdx.x;
    const float* src = input + static_cast<std::size_t>(row) * width;
    float* dst = output + static_cast<std::size_t>(row) * width;
    double local_sum = 0;
    for (int x = tid; x < width; x += blockDim.x) local_sum += src[x];
    const double mean = reduce_sum(local_sum, scratch) / width;
    double local_sq = 0;
    for (int x = tid; x < width; x += blockDim.x) {
        const double centered = static_cast<double>(src[x]) - mean;
        local_sq += centered * centered;
    }
    const double variance = reduce_sum(local_sq, scratch) / width;
    const double inv = 1.0 / sqrt(variance + epsilon);
    for (int x = tid; x < width; x += blockDim.x)
        dst[x] = static_cast<float>((static_cast<double>(src[x]) - mean) * inv * gamma[x] + beta[x]);
}

__global__ void rmsnorm_rows(const float* input, float* output, const float* gamma, int width) {
    __shared__ double scratch[threads];
    const int row = blockIdx.x, tid = threadIdx.x;
    const float* src = input + static_cast<std::size_t>(row) * width;
    float* dst = output + static_cast<std::size_t>(row) * width;
    double local_sq = 0;
    for (int x = tid; x < width; x += blockDim.x) {
        const double value = src[x];
        local_sq += value * value;
    }
    const double mean_square = reduce_sum(local_sq, scratch) / width;
    const double inv = 1.0 / sqrt(mean_square + epsilon);
    for (int x = tid; x < width; x += blockDim.x)
        dst[x] = static_cast<float>(static_cast<double>(src[x]) * inv * gamma[x]);
}

enum Op { SOFTMAX, LAYERNORM, RMSNORM };

std::vector<float> cpu_reference(const std::vector<float>& input, int rows, int width,
                                 const std::vector<float>& gamma,
                                 const std::vector<float>& beta, Op op) {
    std::vector<float> output(input.size());
    for (int r = 0; r < rows; ++r) {
        const float* src = input.data() + static_cast<std::size_t>(r) * width;
        float* dst = output.data() + static_cast<std::size_t>(r) * width;
        if (op == SOFTMAX) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (int x = 0; x < width; ++x) maximum = std::max(maximum, static_cast<double>(src[x]));
            double sum = 0;
            for (int x = 0; x < width; ++x) sum += std::exp(static_cast<double>(src[x]) - maximum);
            for (int x = 0; x < width; ++x)
                dst[x] = static_cast<float>(std::exp(static_cast<double>(src[x]) - maximum) / sum);
        } else if (op == LAYERNORM) {
            double sum = 0;
            for (int x = 0; x < width; ++x) sum += src[x];
            const double mean = sum / width;
            double sq = 0;
            for (int x = 0; x < width; ++x) {
                const double centered = static_cast<double>(src[x]) - mean;
                sq += centered * centered;
            }
            const double inv = 1.0 / std::sqrt(sq / width + epsilon);
            for (int x = 0; x < width; ++x)
                dst[x] = static_cast<float>((static_cast<double>(src[x]) - mean) * inv * gamma[x] + beta[x]);
        } else {
            double sq = 0;
            for (int x = 0; x < width; ++x) {
                const double value = src[x];
                sq += value * value;
            }
            const double inv = 1.0 / std::sqrt(sq / width + epsilon);
            for (int x = 0; x < width; ++x)
                dst[x] = static_cast<float>(static_cast<double>(src[x]) * inv * gamma[x]);
        }
    }
    return output;
}

void run(int rows, int width, bool extreme) {
    const std::size_t count = static_cast<std::size_t>(rows) * width;
    std::vector<float> input(count), gamma(width), beta(width);
    for (int x = 0; x < width; ++x) {
        gamma[x] = 0.8f + (x % 7) * 0.05f;
        beta[x] = (x % 5 - 2) * 0.1f;
    }
    for (int r = 0; r < rows; ++r)
        for (int x = 0; x < width; ++x)
            input[static_cast<std::size_t>(r) * width + x] =
                extreme ? (r == 0 ? 1000.f + (x % 7 - 3) * 0.125f :
                           r == 1 ? -1000.f + (x % 5 - 2) * 0.25f : 0.f)
                        : static_cast<float>(((r * 17 + x * 13) % 37) - 18) / 7.f;
    DeviceBuffer<float> din(count), dout(count), dgamma(width), dbeta(width);
    CUDA_CHECK(cudaMemcpy(dgamma.get(), gamma.data(), width * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta.get(), beta.data(), width * sizeof(float), cudaMemcpyHostToDevice));
    for (Op op : {SOFTMAX, LAYERNORM, RMSNORM}) {
        const char* name = op == SOFTMAX ? "softmax" : op == LAYERNORM ? "layernorm" : "rmsnorm";
        const auto expected = cpu_reference(input, rows, width, gamma, beta, op);
        const auto task_start = Clock::now();
        CUDA_CHECK(cudaMemcpy(din.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dout.get(), 0xa5, count * sizeof(float)));
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        if (op == SOFTMAX) softmax_rows<<<rows, threads>>>(din.get(), dout.get(), width);
        else if (op == LAYERNORM)
            layernorm_rows<<<rows, threads>>>(din.get(), dout.get(), dgamma.get(), dbeta.get(), width);
        else rmsnorm_rows<<<rows, threads>>>(din.get(), dout.get(), dgamma.get(), width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        float kernel_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        std::vector<float> actual(count);
        CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
        const double task_ms = std::chrono::duration<double, std::milli>(Clock::now() - task_start).count();
        double max_error = 0;
        std::size_t bad = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
            max_error = std::max(max_error, error);
            bad += !std::isfinite(actual[i]) || error > 2e-5 + 2e-5 * std::fabs(expected[i]);
        }
        if (op == SOFTMAX) {
            for (int r = 0; r < rows; ++r) {
                double sum = 0;
                for (int x = 0; x < width; ++x) sum += actual[static_cast<std::size_t>(r) * width + x];
                bad += std::fabs(sum - 1.0) > 2e-5;
            }
        }
        std::printf("%s rows=%d width=%d extreme=%d max_abs_error=%.8g bad=%zu "
                    "kernel_ms=%.4f task_ms=%.4f %s\n",
                    name, rows, width, extreme, max_error, bad,
                    static_cast<double>(kernel_ms), task_ms, bad ? "FAIL" : "PASS");
        if (bad) throw std::runtime_error(std::string(name) + " CPU/GPU mismatch");
    }
}

int main() {
    try {
        run(1, 5, false);
        run(7, 37, false);
        run(4, 2049, false);
        run(3, 37, true);
        run(2, 1, true);
        std::puts("chapter 28 row normalization: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 28 row normalization: FAIL: %s\n", e.what());
        return 1;
    }
}
