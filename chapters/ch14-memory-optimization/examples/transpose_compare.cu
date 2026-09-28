#include <cuda_runtime.h>

#include <algorithm>
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

constexpr int kTile = 32;

class Buffer {
public:
    explicit Buffer(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr_, bytes)); }
    ~Buffer() { if (ptr_) cudaFree(ptr_); }
    float* get() const { return static_cast<float*>(ptr_); }
private:
    void* ptr_ = nullptr;
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

__global__ void transpose_naive(const float* input, float* output, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) output[x * height + y] = input[y * width + x];
}

template <int Padding>
__global__ void transpose_shared(const float* input, float* output, int width, int height) {
    __shared__ float tile[kTile][kTile + Padding];
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    __syncthreads(); // every thread in the block reaches this barrier
    const int out_col = blockIdx.y * kTile + threadIdx.x; // original row
    const int out_row = blockIdx.x * kTile + threadIdx.y; // original column
    if (out_col < height && out_row < width)
        output[out_row * height + out_col] = tile[threadIdx.x][threadIdx.y];
}

struct Shape { int width, height; };

void launch(int variant, const float* input, float* output, Shape s) {
    const dim3 threads(kTile, kTile);
    const dim3 blocks((s.width + kTile - 1) / kTile, (s.height + kTile - 1) / kTile);
    if (variant == 0) transpose_naive<<<blocks, threads>>>(input, output, s.width, s.height);
    else if (variant == 1) transpose_shared<0><<<blocks, threads>>>(input, output, s.width, s.height);
    else transpose_shared<1><<<blocks, threads>>>(input, output, s.width, s.height);
    CUDA_CHECK(cudaGetLastError());
}

double median(std::vector<double> data) {
    std::sort(data.begin(), data.end());
    return (data[data.size() / 2 - 1] + data[data.size() / 2]) / 2;
}

void run_shape(Shape s) {
    const std::size_t n = static_cast<std::size_t>(s.width) * s.height;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> input(n), reference(n), output(n);
    for (std::size_t i = 0; i < n; ++i) input[i] = static_cast<float>(i % 257);
    for (int y = 0; y < s.height; ++y)
        for (int x = 0; x < s.width; ++x)
            reference[static_cast<std::size_t>(x) * s.height + y] =
                input[static_cast<std::size_t>(y) * s.width + x];
    Buffer din(bytes), dout(bytes);
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), bytes, cudaMemcpyHostToDevice));
    Events events;
    const char* names[] = {"naive", "shared_32x32", "shared_32x33"};
    for (int variant = 0; variant < 3; ++variant) {
        for (int i = 0; i < 3; ++i) launch(variant, din.get(), dout.get(), s);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> samples;
        for (int i = 0; i < 20; ++i) {
            events.start();
            launch(variant, din.get(), dout.get(), s);
            samples.push_back(events.stop());
        }
        CUDA_CHECK(cudaMemcpy(output.data(), dout.get(), bytes, cudaMemcpyDeviceToHost));
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < n; ++i) mismatches += output[i] != reference[i];
        const double ms = median(samples);
        const double gbps = ms > 0 ? 2.0 * static_cast<double>(bytes) / (ms * 1e6) : 0.0;
        std::printf("width=%d height=%d variant=%s median_ms=%.4f effective_GBps=%.3f mismatches=%zu %s\n",
                    s.width, s.height, names[variant], ms, gbps, mismatches,
                    mismatches ? "FAIL" : "PASS");
        if (mismatches) throw std::runtime_error("transpose mismatch");
    }
}

int main() {
    try {
        for (Shape s : {Shape{16, 16}, Shape{35, 19}, Shape{1024, 1024}}) run_shape(s);
        std::puts("chapter 14 transpose comparison: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 14 transpose comparison: FAIL: %s\n", e.what());
        return 1;
    }
}
