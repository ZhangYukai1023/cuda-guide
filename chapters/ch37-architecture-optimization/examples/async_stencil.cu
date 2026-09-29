#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int block_size = 256;
constexpr int tile_elements = block_size + 4; // 256 outputs, left/right halo, 16-byte copy padding

void check(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
}

class Buffer {
public:
    explicit Buffer(size_t n) : count_(n) { check(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(float)), "cudaMalloc"); }
    ~Buffer() { if (p_) cudaFree(p_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    float* data() const { return p_; }
    void upload(const std::vector<float>& v) {
        if (v.size() != count_) throw std::runtime_error("upload size mismatch");
        check(cudaMemcpy(p_, v.data(), count_ * sizeof(float), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<float> download() const {
        std::vector<float> v(count_);
        check(cudaMemcpy(v.data(), p_, count_ * sizeof(float), cudaMemcpyDeviceToHost), "download");
        return v;
    }
private:
    size_t count_;
    float* p_ = nullptr;
};

class Timer {
public:
    Timer() { check(cudaEventCreate(&start_), "event start"); check(cudaEventCreate(&stop_), "event stop"); }
    ~Timer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { check(cudaEventRecord(start_), "record start"); }
    float stop() {
        check(cudaEventRecord(stop_), "record stop");
        check(cudaEventSynchronize(stop_), "synchronize stop");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_, stop_), "event elapsed");
        return ms;
    }
private:
    cudaEvent_t start_{}, stop_{};
};

__global__ void generic_stencil(const float* padded, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 0.25f * padded[i] + 0.5f * padded[i + 1] + 0.25f * padded[i + 2];
}

__global__ void async_stencil(const float* padded, float* output, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) float tile[tile_elements];
    // Each of 65 threads copies 4 aligned floats. Extra two floats are allocated padding.
    if (threadIdx.x < tile_elements / 4) {
        unsigned int shared_address = static_cast<unsigned int>(
            __cvta_generic_to_shared(&tile[4 * threadIdx.x]));
        const float* global_address = padded + blockIdx.x * block_size + 4 * threadIdx.x;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::
                     "r"(shared_address), "l"(global_address));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads(); // every thread can now read copies issued by other threads
    int i = blockIdx.x * block_size + threadIdx.x;
    if (i < n) output[i] = 0.25f * tile[threadIdx.x] + 0.5f * tile[threadIdx.x + 1]
                         + 0.25f * tile[threadIdx.x + 2];
#else
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 0.25f * padded[i] + 0.5f * padded[i + 1] + 0.25f * padded[i + 2];
#endif
}

std::vector<float> make_padded(int n) {
    std::vector<float> input(n + 4, 0.0f);
    for (int i = 0; i < n; ++i)
        input[i + 1] = static_cast<float>(0.5 + 0.3 * std::sin(0.01 * i) + 0.1 * std::cos(0.031 * i));
    return input;
}

void verify(const char* label, const std::vector<float>& output, const std::vector<float>& input) {
    double max_error = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double expected = 0.25 * input[i] + 0.5 * input[i + 1] + 0.25 * input[i + 2];
        if (!std::isfinite(output[i])) throw std::runtime_error(std::string(label) + " nonfinite output");
        max_error = std::max(max_error, std::abs(static_cast<double>(output[i]) - expected));
    }
    std::cout << label << " max_abs_error=" << max_error << '\n';
    if (max_error > 2e-6) throw std::runtime_error(std::string(label) + " failed CPU comparison");
}

template<class Launcher> float average_ms(Launcher launch) {
    constexpr int warmup = 8, repetitions = 100;
    for (int i = 0; i < warmup; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup synchronize");
    Timer timer;
    timer.start();
    for (int i = 0; i < repetitions; ++i) launch();
    check(cudaGetLastError(), "timed launch");
    return timer.stop() / repetitions;
}

void run_case(int n, bool async_available) {
    if (n % block_size != 0) throw std::runtime_error("N must be divisible by 256");
    auto input = make_padded(n);
    Buffer d_input(input.size()), d_generic(n), d_async(n);
    d_input.upload(input);
    int blocks = n / block_size;
    auto launch_generic = [&] { generic_stencil<<<blocks, block_size>>>(d_input.data(), d_generic.data(), n); };
    auto launch_async = [&] { async_stencil<<<blocks, block_size>>>(d_input.data(), d_async.data(), n); };
    launch_generic();
    check(cudaGetLastError(), "generic launch");
    verify("generic", d_generic.download(), input);
    float generic_ms = average_ms(launch_generic);
    std::cout << "N=" << n << " generic_average_ms=" << generic_ms << '\n';
    if (!async_available) {
        std::cout << "N=" << n << " SKIP cp.async: requires compatible sm_80+ device and binary\n";
        return;
    }
    launch_async();
    check(cudaGetLastError(), "async launch");
    auto async_output = d_async.download();
    verify("cp.async", async_output, input);
    auto generic_output = d_generic.download();
    double pair_difference = 0.0;
    for (int i = 0; i < n; ++i)
        pair_difference = std::max(pair_difference, std::abs(static_cast<double>(async_output[i] - generic_output[i])));
    std::cout << "N=" << n << " generic_async_max_difference=" << pair_difference << '\n';
    if (pair_difference > 2e-6) throw std::runtime_error("generic/async outputs differ");
    float async_ms = average_ms(launch_async);
    std::cout << "N=" << n << " cp_async_average_ms=" << async_ms
              << " observed_generic_over_async=" << generic_ms / async_ms << '\n';
}

} // namespace

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        cudaFuncAttributes generic_attr{}, async_attr{};
        check(cudaFuncGetAttributes(&generic_attr, generic_stencil), "generic attributes");
        check(cudaFuncGetAttributes(&async_attr, async_stencil), "async attributes");
        bool async_available = prop.major >= 8 && async_attr.binaryVersion >= 80;
        std::cout << "GPU=" << prop.name << " compute_capability=" << prop.major << '.' << prop.minor
                  << " generic_regs=" << generic_attr.numRegs << " generic_shared=" << generic_attr.sharedSizeBytes
                  << " async_regs=" << async_attr.numRegs << " async_shared=" << async_attr.sharedSizeBytes
                  << " async_binary_version=" << async_attr.binaryVersion << '\n';
        run_case(1024, async_available);
        run_case(1 << 18, async_available);
        std::cout << "chapter 37 architecture optimization: PASS (cp.async may be SKIP)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "chapter 37 FAIL: " << e.what() << '\n';
        return 1;
    }
}
