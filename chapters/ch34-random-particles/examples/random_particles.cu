#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr int threads = 256;
constexpr int tile_size = 128;
constexpr double soften2 = 0.01;

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

template<class T> class Buffer {
public:
    explicit Buffer(size_t n) : n_(n) { check(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)), "cudaMalloc"); }
    ~Buffer() { if (p_) cudaFree(p_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() const { return p_; }
    void upload(const std::vector<T>& v) {
        if (v.size() != n_) throw std::runtime_error("upload size mismatch");
        check(cudaMemcpy(p_, v.data(), n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> v(n_);
        check(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
    void zero() { check(cudaMemset(p_, 0, n_ * sizeof(T)), "cudaMemset"); }
private:
    size_t n_;
    T* p_ = nullptr;
};

struct Stats {
    unsigned long long inside;
    double integral_sum;
    double integral_sum_sq;
};

// Counter-based mapping: sample index, not thread scheduling, determines the value.
// nvcc 12.8 -O3/sm_120 produced different host/device values when this
// 64-bit mixing sequence was inlined; keep the exact integer mapping out of line.
__host__ __device__ __noinline__ std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__host__ __device__ double uniform01(std::uint64_t seed, std::uint64_t counter) {
    std::uint64_t bits = mix64(seed + counter);
    return static_cast<double>(bits >> 40) * (1.0 / 16777216.0); // 24 exact bits, [0,1)
}

__global__ void monte_carlo(Stats* result, int samples, std::uint64_t seed) {
    __shared__ unsigned int hits[threads];
    __shared__ double sums[threads], squares[threads];
    int lane = threadIdx.x;
    int index = blockIdx.x * blockDim.x + lane;
    unsigned int hit = 0;
    double value = 0.0;
    if (index < samples) {
        double x = uniform01(seed, 2ULL * index);
        double y = uniform01(seed, 2ULL * index + 1);
        hit = x * x + y * y <= 1.0 ? 1U : 0U;
        double t = uniform01(seed ^ 0xd1b54a32d192ed03ULL, index);
        value = 4.0 / (1.0 + t * t);
    }
    hits[lane] = hit;
    sums[lane] = value;
    squares[lane] = value * value;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (lane < stride) {
            hits[lane] += hits[lane + stride];
            sums[lane] += sums[lane + stride];
            squares[lane] += squares[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        atomicAdd(&result->inside, static_cast<unsigned long long>(hits[0]));
        atomicAdd(&result->integral_sum, sums[0]);
        atomicAdd(&result->integral_sum_sq, squares[0]);
    }
}

void test_monte_carlo() {
    constexpr int samples = 1 << 18;
    constexpr std::uint64_t seed = 20260928ULL;
    Buffer<Stats> gpu(1);
    gpu.zero();
    monte_carlo<<<(samples + threads - 1) / threads, threads>>>(gpu.data(), samples, seed);
    check(cudaGetLastError(), "Monte Carlo launch");
    Stats out = gpu.download()[0];
    unsigned long long cpu_hits = 0;
    double cpu_sum = 0.0, cpu_sum_sq = 0.0;
    for (int i = 0; i < samples; ++i) {
        double x = uniform01(seed, 2ULL * i);
        double y = uniform01(seed, 2ULL * i + 1);
        cpu_hits += x * x + y * y <= 1.0 ? 1ULL : 0ULL;
        double t = uniform01(seed ^ 0xd1b54a32d192ed03ULL, i);
        double value = 4.0 / (1.0 + t * t);
        cpu_sum += value;
        cpu_sum_sq += value * value;
    }
    if (out.inside != cpu_hits) {
        std::cerr << "Monte Carlo hit mismatch gpu=" << out.inside << " cpu=" << cpu_hits << '\n';
        throw std::runtime_error("Monte Carlo hit count differs from CPU");
    }
    double cpu_integral = cpu_sum / samples;
    double gpu_integral = out.integral_sum / samples;
    double sum_error = std::abs(gpu_integral - cpu_integral);
    if (!std::isfinite(gpu_integral) || sum_error > 1e-6) throw std::runtime_error("Monte Carlo integral differs from CPU");
    double p = static_cast<double>(cpu_hits) / samples;
    double pi_estimate = 4.0 * p;
    double pi_se = 4.0 * std::sqrt(p * (1.0 - p) / samples);
    double variance = std::max(0.0, cpu_sum_sq / samples - cpu_integral * cpu_integral);
    double integral_se = std::sqrt(variance / samples);
    std::cout << std::setprecision(12) << "Monte Carlo N=" << samples << " seed=" << seed
              << " hits=" << out.inside << " pi_estimate=" << pi_estimate
              << " pi_abs_error=" << std::abs(pi_estimate - pi)
              << " pi_SE_estimate=" << pi_se << '\n';
    std::cout << "integral[0,1] 4/(1+x^2): gpu=" << gpu_integral
              << " cpu_same_samples=" << cpu_integral << " gpu_cpu_error=" << sum_error
              << " integral_SE_estimate=" << integral_se << '\n';
    // Statistical deviation from pi is reported, not used as a deterministic correctness gate.
}

__device__ float3 interact(float4 self, float4 other, float3 acceleration) {
    float dx = other.x - self.x, dy = other.y - self.y, dz = other.z - self.z;
    float distance2 = dx * dx + dy * dy + dz * dz + static_cast<float>(soften2);
    float inverse = rsqrtf(distance2);
    float scale = other.w * inverse * inverse * inverse;
    acceleration.x += dx * scale;
    acceleration.y += dy * scale;
    acceleration.z += dz * scale;
    return acceleration;
}

__global__ void nbody_naive(const float4* positions, float3* acceleration, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 self = positions[i];
    float3 total = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n; ++j) total = interact(self, positions[j], total);
    acceleration[i] = total;
}

__global__ void nbody_tiled(const float4* positions, float3* acceleration, int n) {
    __shared__ float4 tile[tile_size];
    int i = blockIdx.x * tile_size + threadIdx.x;
    float4 self = i < n ? positions[i] : make_float4(0, 0, 0, 0);
    float3 total = make_float3(0.0f, 0.0f, 0.0f);
    for (int start = 0; start < n; start += tile_size) {
        int source = start + threadIdx.x;
        tile[threadIdx.x] = source < n ? positions[source] : make_float4(0, 0, 0, 0);
        __syncthreads();
        int valid = n - start < tile_size ? n - start : tile_size;
        if (i < n) for (int j = 0; j < valid; ++j) total = interact(self, tile[j], total);
        __syncthreads();
    }
    if (i < n) acceleration[i] = total;
}

std::vector<float4> particles(int n) {
    std::vector<float4> out(n);
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i % 17) / 17.0f;
        float y = static_cast<float>((i / 17) % 17) / 17.0f;
        float z = static_cast<float>((i * 7) % 19) / 19.0f;
        float mass = 0.5f + static_cast<float>(i % 5) * 0.1f;
        out[i] = make_float4(x, y, z, mass);
    }
    return out;
}

std::vector<float3> cpu_nbody(const std::vector<float4>& p) {
    int n = static_cast<int>(p.size());
    std::vector<float3> out(n);
    for (int i = 0; i < n; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = 0; j < n; ++j) {
            double dx = static_cast<double>(p[j].x) - p[i].x;
            double dy = static_cast<double>(p[j].y) - p[i].y;
            double dz = static_cast<double>(p[j].z) - p[i].z;
            double d2 = dx * dx + dy * dy + dz * dz + soften2;
            double scale = p[j].w / (d2 * std::sqrt(d2));
            ax += dx * scale; ay += dy * scale; az += dz * scale;
        }
        out[i] = make_float3(static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(az));
    }
    return out;
}

double max_difference(const std::vector<float3>& a, const std::vector<float3>& b) {
    if (a.size() != b.size()) throw std::runtime_error("N-body output size mismatch");
    double error = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i].x) || !std::isfinite(a[i].y) || !std::isfinite(a[i].z))
            throw std::runtime_error("N-body nonfinite acceleration");
        error = std::max(error, std::abs(static_cast<double>(a[i].x) - b[i].x));
        error = std::max(error, std::abs(static_cast<double>(a[i].y) - b[i].y));
        error = std::max(error, std::abs(static_cast<double>(a[i].z) - b[i].z));
    }
    return error;
}

void test_nbody() {
    constexpr int n = 193; // deliberately not divisible by tile_size
    auto input = particles(n);
    auto reference = cpu_nbody(input);
    Buffer<float4> p(n);
    Buffer<float3> naive(n), tiled(n);
    p.upload(input);
    nbody_naive<<<(n + tile_size - 1) / tile_size, tile_size>>>(p.data(), naive.data(), n);
    check(cudaGetLastError(), "N-body naive launch");
    nbody_tiled<<<(n + tile_size - 1) / tile_size, tile_size>>>(p.data(), tiled.data(), n);
    check(cudaGetLastError(), "N-body tiled launch");
    auto a = naive.download();
    auto b = tiled.download();
    double naive_error = max_difference(a, reference);
    double tiled_error = max_difference(b, reference);
    double pair_error = max_difference(a, b);
    std::cout << "N-body n=" << n << " softened pair model: naive_cpu_max_error=" << naive_error
              << " tiled_cpu_max_error=" << tiled_error << " naive_tiled_max_error=" << pair_error << '\n';
    constexpr double tolerance = 3e-3;
    if (naive_error > tolerance || tiled_error > tolerance || pair_error > tolerance)
        throw std::runtime_error("N-body acceleration differs from CPU reference");
}

} // namespace

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        std::cout << "GPU=" << prop.name << '\n';
        test_monte_carlo();
        test_nbody();
        std::cout << "chapter 34 random and particles: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "chapter 34 FAIL: " << e.what() << '\n';
        return 1;
    }
}
