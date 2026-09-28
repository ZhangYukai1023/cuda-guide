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
constexpr int d = 16, dv = 16, key_tile = 16;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

__global__ void score_kernel(const float* q, const float* k, float* score,
                             int sequence, bool causal) {
    const int key = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y * blockDim.y + threadIdx.y;
    if (query >= sequence || key >= sequence) return;
    if (causal && key > query) {
        score[query * sequence + key] = -INFINITY;
        return;
    }
    float sum = 0.f;
    for (int col = 0; col < d; ++col)
        sum += q[query * d + col] * k[key * d + col];
    score[query * sequence + key] = sum * (1.f / sqrtf(static_cast<float>(d)));
}

__global__ void softmax_kernel(const float* score, float* probability, int sequence) {
    const int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float scratch[128];
    float maximum = -INFINITY;
    for (int col = tid; col < sequence; col += blockDim.x)
        maximum = fmaxf(maximum, score[row * sequence + col]);
    scratch[tid] = maximum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
        if (tid < offset) scratch[tid] = fmaxf(scratch[tid], scratch[tid + offset]);
        __syncthreads();
    }
    maximum = scratch[0];
    float sum = 0.f;
    for (int col = tid; col < sequence; col += blockDim.x)
        sum += expf(score[row * sequence + col] - maximum);
    scratch[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
        if (tid < offset) scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (int col = tid; col < sequence; col += blockDim.x)
        probability[row * sequence + col] = expf(score[row * sequence + col] - maximum) / denominator;
}

__global__ void output_kernel(const float* probability, const float* v, float* output, int sequence) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= sequence || col >= dv) return;
    float sum = 0.f;
    for (int key = 0; key < sequence; ++key)
        sum += probability[row * sequence + key] * v[key * dv + col];
    output[row * dv + col] = sum;
}

__global__ void online_tiled_forward(const float* q, const float* k, const float* v,
                                     float* output, int sequence, bool causal) {
    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= sequence) return;
    float numerator[dv] = {};
    float running_max = -INFINITY, running_sum = 0.f;
    for (int begin = 0; begin < sequence; begin += key_tile) {
        float scores[key_tile];
        float tile_max = -INFINITY;
        for (int j = 0; j < key_tile; ++j) {
            const int key = begin + j;
            if (key >= sequence || (causal && key > query)) {
                scores[j] = -INFINITY;
                continue;
            }
            float dot = 0.f;
            for (int col = 0; col < d; ++col)
                dot += q[query * d + col] * k[key * d + col];
            scores[j] = dot * (1.f / sqrtf(static_cast<float>(d)));
            tile_max = fmaxf(tile_max, scores[j]);
        }
        if (tile_max == -INFINITY) continue;
        const float new_max = fmaxf(running_max, tile_max);
        const float rescale = running_max == -INFINITY ? 0.f : expf(running_max - new_max);
        running_sum *= rescale;
        for (int col = 0; col < dv; ++col) numerator[col] *= rescale;
        for (int j = 0; j < key_tile; ++j) {
            const int key = begin + j;
            if (key >= sequence || scores[j] == -INFINITY) continue;
            const float weight = expf(scores[j] - new_max);
            running_sum += weight;
            for (int col = 0; col < dv; ++col)
                numerator[col] += weight * v[key * dv + col];
        }
        running_max = new_max;
    }
    for (int col = 0; col < dv; ++col)
        output[query * dv + col] = numerator[col] / running_sum;
}

std::vector<float> cpu_reference(const std::vector<float>& q, const std::vector<float>& k,
                                 const std::vector<float>& v, int sequence, bool causal) {
    std::vector<float> output(static_cast<std::size_t>(sequence) * dv);
    for (int query = 0; query < sequence; ++query) {
        std::vector<double> scores(sequence, -std::numeric_limits<double>::infinity());
        double maximum = -std::numeric_limits<double>::infinity();
        for (int key = 0; key < sequence; ++key) {
            if (causal && key > query) continue;
            double dot = 0;
            for (int col = 0; col < d; ++col)
                dot += static_cast<double>(q[query * d + col]) * k[key * d + col];
            scores[key] = dot / std::sqrt(static_cast<double>(d));
            maximum = std::max(maximum, scores[key]);
        }
        double sum = 0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            sum += score;
        }
        for (int col = 0; col < dv; ++col) {
            double weighted = 0;
            for (int key = 0; key < sequence; ++key)
                weighted += scores[key] / sum * v[key * dv + col];
            output[query * dv + col] = static_cast<float>(weighted);
        }
    }
    return output;
}

template <typename Launch>
float timed(Launch launch) {
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

void verify(const char* name, const std::vector<float>& actual,
            const std::vector<float>& expected, int sequence, bool causal, float kernel_ms) {
    std::size_t bad = 0;
    double max_error = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        max_error = std::max(max_error, error);
        bad += !std::isfinite(actual[i]) || error > 2e-3 + 2e-3 * std::fabs(expected[i]);
    }
    std::printf("%s sequence=%d causal=%d max_abs_error=%.7g bad=%zu kernel_ms=%.4f %s\n",
                name, sequence, causal, max_error, bad, static_cast<double>(kernel_ms), bad ? "FAIL" : "PASS");
    if (bad) throw std::runtime_error(std::string(name) + " CPU/GPU mismatch");
}

void run(int sequence, bool causal, bool extreme) {
    std::vector<float> q(static_cast<std::size_t>(sequence) * d),
                       k(static_cast<std::size_t>(sequence) * d),
                       v(static_cast<std::size_t>(sequence) * dv);
    const float scale = extreme ? 50.f : 1.f;
    for (int i = 0; i < sequence; ++i) {
        for (int col = 0; col < d; ++col) {
            q[i * d + col] = scale * static_cast<float>((i * 3 + col * 7) % 11 - 5) / 7.f;
            k[i * d + col] = scale * static_cast<float>((i * 5 + col * 2) % 13 - 6) / 9.f;
        }
        for (int col = 0; col < dv; ++col)
            v[i * dv + col] = static_cast<float>((i * 7 + col * 3) % 17 - 8) / 8.f;
    }
    const auto expected = cpu_reference(q, k, v, sequence, causal);
    DeviceBuffer<float> dq(q.size()), dk(k.size()), dvbuf(v.size()), dout(expected.size());
    CUDA_CHECK(cudaMemcpy(dq.get(), q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk.get(), k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvbuf.get(), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    const dim3 threads2(16, 16), blocks_score((sequence + 15) / 16, (sequence + 15) / 16),
               blocks_output((dv + 15) / 16, (sequence + 15) / 16);
    std::vector<float> actual(expected.size());
    {
        DeviceBuffer<float> dscore(static_cast<std::size_t>(sequence) * sequence),
                            dprob(static_cast<std::size_t>(sequence) * sequence);
        const float explicit_ms = timed([&] {
            score_kernel<<<blocks_score, threads2>>>(dq.get(), dk.get(), dscore.get(), sequence, causal);
            softmax_kernel<<<sequence, 128>>>(dscore.get(), dprob.get(), sequence);
            output_kernel<<<blocks_output, threads2>>>(dprob.get(), dvbuf.get(), dout.get(), sequence);
        });
        CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        verify("explicit_scores_softmax_pv", actual, expected, sequence, causal, explicit_ms);
        std::vector<float> probabilities(static_cast<std::size_t>(sequence) * sequence);
        CUDA_CHECK(cudaMemcpy(probabilities.data(), dprob.get(), probabilities.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int query = 0; query < sequence; ++query) {
            double sum = 0;
            for (int key = 0; key < sequence; ++key) {
                sum += probabilities[query * sequence + key];
                if (causal && key > query && probabilities[query * sequence + key] != 0.f)
                    throw std::runtime_error("causal mask leaked probability");
            }
            if (std::fabs(sum - 1.0) > 1e-4) throw std::runtime_error("explicit Softmax row sum mismatch");
        }
    }
    const float online_ms = timed([&] {
        online_tiled_forward<<<(sequence + 127) / 128, 128>>>(
            dq.get(), dk.get(), dvbuf.get(), dout.get(), sequence, causal);
    });
    CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    verify("online_tiled_forward", actual, expected, sequence, causal, online_ms);
    const std::size_t explicit_intermediate_bytes = static_cast<std::size_t>(2) * sequence * sequence * sizeof(float);
    std::printf("sequence=%d explicit_score_probability_bytes=%zu online_key_tile=%d PASS\n",
                sequence, explicit_intermediate_bytes, key_tile);
}

int main() {
    try {
        run(5, false, false);
        run(5, true, false);
        run(17, true, false);
        run(65, false, false);
        run(65, true, false);
        run(17, true, true);
        std::puts("chapter 30 attention forward: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 30 attention forward: FAIL: %s\n", e.what());
        return 1;
    }
}
