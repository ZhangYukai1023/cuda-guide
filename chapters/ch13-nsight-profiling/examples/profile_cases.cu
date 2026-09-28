#include <cuda_runtime.h>
#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#define GUIDE_HAS_NVTX3 1
#else
#define GUIDE_HAS_NVTX3 0
namespace nvtx3 {
struct scoped_range {
    explicit scoped_range(const char*) {}
};
}
#endif

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#define CUDA_CHECK(call) do { \
    const cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)

constexpr std::size_t kCount = 16384;
constexpr std::size_t kChunk = 256;

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t n) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data_), n * sizeof(T), cudaHostAllocDefault));
    }
    ~PinnedBuffer() { if (data_) cudaFreeHost(data_); }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

__global__ void transform_contiguous(const float* input, float* output, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2.0f * input[i] + 1.0f;
}

__global__ void gather_stride(const float* input, float* output, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2.0f * input[(i * 17) % n] + 1.0f;
}

void verify(const float* input, const float* output, bool strided) {
    for (std::size_t i = 0; i < kCount; ++i) {
        const std::size_t source = strided ? (i * 17) % kCount : i;
        if (output[i] != 2.0f * input[source] + 1.0f)
            throw std::runtime_error("CPU/GPU mismatch");
    }
}

void run(const std::string& mode) {
    if (mode != "tiny" && mode != "batch" && mode != "stride")
        throw std::invalid_argument("mode must be tiny, batch, or stride");
    if (!GUIDE_HAS_NVTX3) std::puts("SKIP NVTX ranges: nvtx3/nvtx3.hpp unavailable");
    PinnedBuffer<float> input(kCount), output(kCount);
    DeviceBuffer<float> din(kCount), dout(kCount);
    Stream stream;
    for (std::size_t i = 0; i < kCount; ++i) input.get()[i] = static_cast<float>(i % 127) / 8.0f;
    const std::size_t all_bytes = kCount * sizeof(float);
    nvtx3::scoped_range overall{"profile_work_including_wait"};

    if (mode == "tiny") {
        nvtx3::scoped_range range{"tiny_copies_with_cpu_gaps"};
        for (std::size_t base = 0; base < kCount; base += kChunk) {
            nvtx3::scoped_range iteration{"one_tiny_chunk"};
            const std::size_t bytes = kChunk * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(din.get() + base, input.get() + base, bytes,
                                       cudaMemcpyHostToDevice, stream.get()));
            transform_contiguous<<<1, 256, 0, stream.get()>>>(din.get() + base, dout.get() + base, kChunk);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(output.get() + base, dout.get() + base, bytes,
                                       cudaMemcpyDeviceToHost, stream.get()));
            // Deliberate host-side submission gap, visible in the timeline.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    } else if (mode == "batch") {
        nvtx3::scoped_range range{"batched_copy_and_kernel"};
        CUDA_CHECK(cudaMemcpyAsync(din.get(), input.get(), all_bytes,
                                   cudaMemcpyHostToDevice, stream.get()));
        transform_contiguous<<<static_cast<unsigned>(kCount / 256), 256, 0, stream.get()>>>(
            din.get(), dout.get(), kCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(output.get(), dout.get(), all_bytes,
                                   cudaMemcpyDeviceToHost, stream.get()));
    } else {
        nvtx3::scoped_range range{"strided_gather_kernel"};
        CUDA_CHECK(cudaMemcpyAsync(din.get(), input.get(), all_bytes,
                                   cudaMemcpyHostToDevice, stream.get()));
        gather_stride<<<static_cast<unsigned>(kCount / 256), 256, 0, stream.get()>>>(
            din.get(), dout.get(), kCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(output.get(), dout.get(), all_bytes,
                                   cudaMemcpyDeviceToHost, stream.get()));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    verify(input.get(), output.get(), mode == "stride");
    std::printf("profile case=%s n=%zu: PASS\n", mode.c_str(), kCount);
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: profile_cases tiny|batch|stride");
        CUDA_CHECK(cudaSetDevice(0));
        run(argv[1]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "profile_cases: FAIL: %s\n", e.what());
        return 1;
    }
}
