#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t n) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data_), n * sizeof(T), cudaHostAllocDefault));
    }
    ~PinnedBuffer() { if (data_) cudaFreeHost(data_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
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
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
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

class EventPair {
public:
    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~EventPair() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    double stop(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms);
    }
private:
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
};

__global__ void add(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

struct Context {
    explicit Context(std::size_t count)
        : n(count), a(count), b(count), out(count), da(count), db(count), dout(count) {
        for (std::size_t i = 0; i < n; ++i) {
            a.get()[i] = static_cast<float>(i % 97) / 8.0f;
            b.get()[i] = static_cast<float>(i % 31) / 16.0f;
        }
    }
    std::size_t n;
    PinnedBuffer<float> a, b, out;
    DeviceBuffer<float> da, db, dout;
    Stream stream;
    EventPair events;
};

void launch(Context& c) {
    add<<<static_cast<unsigned>((c.n + 255) / 256), 256, 0, c.stream.get()>>>(
        c.da.get(), c.db.get(), c.dout.get(), c.n);
    CUDA_CHECK(cudaGetLastError());
}

void enqueue_pipeline(Context& c) {
    const std::size_t bytes = c.n * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(c.da.get(), c.a.get(), bytes, cudaMemcpyHostToDevice, c.stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(c.db.get(), c.b.get(), bytes, cudaMemcpyHostToDevice, c.stream.get()));
    launch(c);
    CUDA_CHECK(cudaMemcpyAsync(c.out.get(), c.dout.get(), bytes, cudaMemcpyDeviceToHost, c.stream.get()));
}

void validate(const Context& c) {
    for (std::size_t i = 0; i < c.n; ++i)
        if (c.out.get()[i] != c.a.get()[i] + c.b.get()[i])
            throw std::runtime_error("GPU/CPU result mismatch");
}

struct Stats { double median, p10, p90; };
Stats summarize(std::vector<double> samples) {
    if (samples.empty()) throw std::runtime_error("no benchmark samples");
    std::sort(samples.begin(), samples.end());
    const std::size_t count = samples.size();
    const double med = count % 2 ? samples[count / 2] : (samples[count / 2 - 1] + samples[count / 2]) / 2;
    return {med, samples[(count - 1) / 10], samples[9 * (count - 1) / 10]};
}

void benchmark(std::size_t n) {
    constexpr int warmups = 5;
    constexpr int repeats = 20;
    const auto setup_start = Clock::now();
    Context c(n);
    const double setup_ms = Milliseconds(Clock::now() - setup_start).count();
    const std::size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(c.da.get(), c.a.get(), bytes, cudaMemcpyHostToDevice, c.stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(c.db.get(), c.b.get(), bytes, cudaMemcpyHostToDevice, c.stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(c.stream.get()));

    for (int i = 0; i < warmups; ++i) launch(c);
    CUDA_CHECK(cudaStreamSynchronize(c.stream.get()));
    std::vector<double> kernel_ms;
    for (int i = 0; i < repeats; ++i) {
        c.events.start(c.stream.get());
        launch(c);
        kernel_ms.push_back(c.events.stop(c.stream.get()));
    }

    for (int i = 0; i < warmups; ++i) enqueue_pipeline(c);
    CUDA_CHECK(cudaStreamSynchronize(c.stream.get()));
    std::vector<double> pipeline_ms;
    for (int i = 0; i < repeats; ++i) {
        c.events.start(c.stream.get());
        enqueue_pipeline(c);
        pipeline_ms.push_back(c.events.stop(c.stream.get()));
    }

    std::vector<double> wall_ms;
    for (int i = 0; i < repeats; ++i) {
        const auto start = Clock::now();
        enqueue_pipeline(c);
        CUDA_CHECK(cudaStreamSynchronize(c.stream.get()));
        wall_ms.push_back(Milliseconds(Clock::now() - start).count());
    }
    validate(c);

    std::vector<float> cpu_out(n);
    std::vector<double> cpu_ms;
    for (int rep = 0; rep < repeats; ++rep) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < n; ++i) cpu_out[i] = c.a.get()[i] + c.b.get()[i];
        cpu_ms.push_back(Milliseconds(Clock::now() - start).count());
    }
    for (std::size_t i = 0; i < n; ++i)
        if (cpu_out[i] != c.out.get()[i]) throw std::runtime_error("CPU reference mismatch");

    const Stats k = summarize(kernel_ms), p = summarize(pipeline_ms);
    const Stats w = summarize(wall_ms), cpu = summarize(cpu_ms);
    const double gbps = k.median > 0 ? (3.0 * static_cast<double>(bytes)) / (k.median * 1e6) : 0.0;
    std::printf("n=%zu setup_ms=%.4f kernel_ms=%.4f [%.4f,%.4f] "
                "pipeline_ms=%.4f [%.4f,%.4f] wall_ms=%.4f [%.4f,%.4f] "
                "cpu_ms=%.4f [%.4f,%.4f] kernel_effective_GBps=%.3f PASS\n",
                n, setup_ms, k.median, k.p10, k.p90,
                p.median, p.p10, p.p90, w.median, w.p10, w.p90,
                cpu.median, cpu.p10, cpu.p90, gbps);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr)); // context initialization excluded from reported setup
        for (std::size_t n : {std::size_t{7}, std::size_t{4096}, std::size_t{1} << 20})
            benchmark(n);
        std::puts("chapter 12 benchmark: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 12 benchmark: FAIL: %s\n", e.what());
        return 1;
    }
}
