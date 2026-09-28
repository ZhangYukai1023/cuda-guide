#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
        throw std::runtime_error("CUDA call failed"); \
    } \
} while (false)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : ptr_(nullptr) {
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_;
};

__global__ void add_int(const int* a, const int* b, int* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void add_float(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void pairwise_four(const float* input, float* out) {
    __shared__ float values[4];
    const int t = threadIdx.x;
    values[t] = input[t];
    __syncthreads();
    if (t == 0 || t == 2) values[t] += values[t + 1];
    __syncthreads();
    if (t == 0) out[0] = values[0] + values[2];
}

__global__ void round_via_half(const float* input, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__float2half_rn(input[i]));
}

__global__ void large_logical_index(int* out, std::size_t base, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<int>((base + i) % 97u);
}

// NaN matches only an explicitly expected NaN. Infinities must have the same sign.
bool close_float(float actual, double expected, double atol, double rtol) {
    if (std::isnan(expected)) return std::isnan(actual);
    if (std::isnan(actual)) return false;
    if (std::isinf(expected) || std::isinf(actual)) return static_cast<double>(actual) == expected;
    const double diff = std::abs(static_cast<double>(actual) - expected);
    return diff <= atol + rtol * std::abs(expected);
}

void run_integer(std::size_t n) {
    if (n == 0) {
        std::puts("integer n=0: PASS (documented empty result; no kernel launch)");
        return;
    }
    std::vector<int> a(n), b(n), result(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<int>(i % 97) - 48;
        b[i] = static_cast<int>((i * 13) % 71) - 35;
    }
    DeviceBuffer<int> da(n), db(n), dout(n);
    CUDA_CHECK(cudaMemcpy(da.get(), a.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.get(), b.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    add_int<<<static_cast<unsigned>((n + 127) / 128), 128>>>(da.get(), db.get(), dout.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), dout.get(), n * sizeof(int), cudaMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (result[i] != a[i] + b[i]) {
            if (mismatches == 0)
                std::fprintf(stderr, "integer first mismatch i=%zu a=%d b=%d actual=%d expected=%d\n",
                             i, a[i], b[i], result[i], a[i] + b[i]);
            ++mismatches;
        }
    }
    std::printf("integer n=%zu: %s mismatches=%zu\n", n, mismatches ? "FAIL" : "PASS", mismatches);
    if (mismatches) throw std::runtime_error("integer comparison failed");
}

void run_float(std::size_t n) {
    if (n == 0) {
        std::puts("float n=0: PASS (documented empty result; no kernel launch)");
        return;
    }
    // A fixed generator and integer-to-binary-fraction mapping make the input reproducible.
    std::uint32_t state = 0x00c0ffeeu;
    auto next = [&state]() { state = state * 1664525u + 1013904223u; return state; };
    std::vector<float> a(n), b(n), result(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(static_cast<int>(next() % 2001u) - 1000) / 128.0f;
        b[i] = static_cast<float>(static_cast<int>(next() % 2001u) - 1000) / 128.0f;
    }
    DeviceBuffer<float> da(n), db(n), dout(n);
    CUDA_CHECK(cudaMemcpy(da.get(), a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.get(), b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    add_float<<<static_cast<unsigned>((n + 127) / 128), 128>>>(da.get(), db.get(), dout.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), dout.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double reference = static_cast<double>(a[i]) + static_cast<double>(b[i]);
        if (!close_float(result[i], reference, 1e-6, 1e-6)) {
            if (mismatches == 0)
                std::fprintf(stderr, "float first mismatch i=%zu a=%g b=%g actual=%g reference=%.17g\n",
                             i, static_cast<double>(a[i]), static_cast<double>(b[i]),
                             static_cast<double>(result[i]), reference);
            ++mismatches;
        }
    }
    std::printf("float n=%zu seed=0x00c0ffee: %s mismatches=%zu\n", n, mismatches ? "FAIL" : "PASS", mismatches);
    if (mismatches) throw std::runtime_error("float comparison failed");
}

void run_special_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> a = {0.0f, -0.0f, inf, -inf, inf, nan};
    const std::vector<float> b = {-0.0f, 0.0f, 1.0f, -1.0f, -inf, 2.0f};
    const std::vector<double> expected = {0.0, 0.0, INFINITY, -INFINITY, NAN, NAN};
    const std::size_t n = a.size();
    std::vector<float> result(n);
    DeviceBuffer<float> da(n), db(n), dout(n);
    CUDA_CHECK(cudaMemcpy(da.get(), a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.get(), b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    add_float<<<1, 128>>>(da.get(), db.get(), dout.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), dout.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n; ++i) mismatches += !close_float(result[i], expected[i], 0, 0);
    std::printf("special NaN/Inf: %s mismatches=%zu\n", mismatches ? "FAIL" : "PASS", mismatches);
    if (mismatches) throw std::runtime_error("special-value comparison failed");
    if (close_float(nan, 0.0, 1e-5, 1e-5) || close_float(inf, 1.0, 1e-5, 1e-5))
        throw std::runtime_error("comparator accepted invalid values");
}

void run_order_example() {
    const std::vector<float> values = {1.0e8f, 1.0f, -1.0e8f, 1.0f};
    DeviceBuffer<float> din(4), dout(1);
    CUDA_CHECK(cudaMemcpy(din.get(), values.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
    pairwise_four<<<1, 4>>>(din.get(), dout.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float pairwise = 0;
    CUDA_CHECK(cudaMemcpy(&pairwise, dout.get(), sizeof(float), cudaMemcpyDeviceToHost));
    volatile float sequential = 0.0f;
    for (float value : values) sequential = sequential + value;
    const double accurate = 2.0;
    std::printf("sum order: sequential=%g adjacent_pairwise=%g double_reference=%g\n",
                static_cast<double>(sequential), static_cast<double>(pairwise), accurate);
    if (sequential != 1.0f || pairwise != 0.0f)
        throw std::runtime_error("unexpected order demonstration result");
}

void run_half_roundtrip() {
    const std::vector<float> input = {0.0f, 1.0f, 1.0001f};
    std::vector<float> result(input.size());
    DeviceBuffer<float> din(input.size()), dout(input.size());
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    round_via_half<<<1, 32>>>(din.get(), dout.get(), input.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), dout.get(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const bool ok = result[0] == 0.0f && result[1] == 1.0f && result[2] == 1.0f;
    std::printf("half roundtrip 1.0001 -> %.7g: %s\n", static_cast<double>(result[2]), ok ? "PASS" : "FAIL");
    if (!ok) throw std::runtime_error("half roundtrip failed");
}

void run_large_index() {
    constexpr std::size_t n = 8;
    constexpr std::size_t base = (std::size_t{1} << 32) + 123;
    DeviceBuffer<int> dout(n);
    large_logical_index<<<1, 32>>>(dout.get(), base, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(), dout.get(), n * sizeof(int), cudaMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n; ++i)
        mismatches += result[i] != static_cast<int>((base + i) % 97u);
    std::printf("logical index base=%zu n=%zu: %s mismatches=%zu\n",
                base, n, mismatches ? "FAIL" : "PASS", mismatches);
    if (mismatches) throw std::runtime_error("large-index comparison failed");
}

int main() {
    try {
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{1003}, std::size_t{4097}}) {
            run_integer(n);
            run_float(n);
        }
        run_special_values();
        run_order_example();
        run_half_roundtrip();
        run_large_index();
        std::puts("chapter 9 validation: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 9 validation: FAIL: %s\n", e.what());
        return 1;
    }
}
