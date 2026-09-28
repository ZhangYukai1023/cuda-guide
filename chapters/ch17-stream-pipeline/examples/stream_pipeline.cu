#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)

constexpr int kBatches = 10;
constexpr std::size_t kItems = 65536;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t n) { CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data_), n * sizeof(T), cudaHostAllocDefault)); }
    ~PinnedBuffer() { if (data_) cudaFreeHost(data_); }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T))); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

class CompletionEvent {
public:
    CompletionEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
    ~CompletionEvent() { if (event_) cudaEventDestroy(event_); }
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

struct Slot {
    Slot() : a(kItems), b(kItems), out(kItems) {}
    DeviceBuffer<float> a, b, out;
    Stream stream;
    CompletionEvent done;
    bool busy = false;
    int latest_batch = -1;
};

__global__ void add(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void sum_last_first(const float* p0, const float* p1,
                               const float* p2, const float* p3,
                               int slot_count, float* sum) {
    if (threadIdx.x != 0) return;
    float value = p0[0];
    if (slot_count > 1) value += p1[0];
    if (slot_count > 2) value += p2[0];
    if (slot_count > 3) value += p3[0];
    sum[0] = value;
}

void enqueue_batch(Slot& slot, int batch, const float* a, const float* b, float* out) {
    if (slot.busy) CUDA_CHECK(cudaEventSynchronize(slot.done.get()));
    const std::size_t offset = static_cast<std::size_t>(batch) * kItems;
    const std::size_t bytes = kItems * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(slot.a.get(), a + offset, bytes,
                               cudaMemcpyHostToDevice, slot.stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(slot.b.get(), b + offset, bytes,
                               cudaMemcpyHostToDevice, slot.stream.get()));
    add<<<static_cast<unsigned>((kItems + 255) / 256), 256, 0, slot.stream.get()>>>(
        slot.a.get(), slot.b.get(), slot.out.get(), kItems);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(out + offset, slot.out.get(), bytes,
                               cudaMemcpyDeviceToHost, slot.stream.get()));
    CUDA_CHECK(cudaEventRecord(slot.done.get(), slot.stream.get()));
    slot.busy = true;
    slot.latest_batch = batch;
}

double median(std::vector<double> data) {
    std::sort(data.begin(), data.end());
    return data[data.size() / 2];
}

void run_mode(int slot_count) {
    const std::size_t total = static_cast<std::size_t>(kBatches) * kItems;
    PinnedBuffer<float> host_a(total), host_b(total), host_out(total), host_sum(1);
    for (int batch = 0; batch < kBatches; ++batch)
        for (std::size_t i = 0; i < kItems; ++i) {
            const std::size_t index = static_cast<std::size_t>(batch) * kItems + i;
            host_a.get()[index] = static_cast<float>(batch) + static_cast<float>(i % 17) / 8.0f;
            host_b.get()[index] = 1.0f + static_cast<float>(i % 11) / 16.0f;
        }
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < slot_count; ++i) slots.emplace_back(new Slot);
    Stream merge_stream;
    DeviceBuffer<float> device_sum(1);

    auto process_all = [&]() {
        for (int batch = 0; batch < kBatches; ++batch)
            enqueue_batch(*slots[batch % slot_count], batch,
                          host_a.get(), host_b.get(), host_out.get());
        for (const auto& slot : slots)
            CUDA_CHECK(cudaStreamWaitEvent(merge_stream.get(), slot->done.get(), 0));
        const float* p0 = slots[0]->out.get();
        const float* p1 = slot_count > 1 ? slots[1]->out.get() : nullptr;
        const float* p2 = slot_count > 2 ? slots[2]->out.get() : nullptr;
        const float* p3 = slot_count > 3 ? slots[3]->out.get() : nullptr;
        sum_last_first<<<1, 1, 0, merge_stream.get()>>>(p0, p1, p2, p3, slot_count, device_sum.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host_sum.get(), device_sum.get(), sizeof(float),
                                   cudaMemcpyDeviceToHost, merge_stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(merge_stream.get()));
    };

    process_all(); // warmup
    std::vector<double> samples;
    for (int i = 0; i < 5; ++i) {
        const auto start = Clock::now();
        process_all();
        samples.push_back(Milliseconds(Clock::now() - start).count());
    }
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < total; ++i)
        mismatches += host_out.get()[i] != host_a.get()[i] + host_b.get()[i];
    float expected_sum = 0;
    for (const auto& slot : slots) expected_sum += static_cast<float>(slot->latest_batch + 1);
    if (mismatches || host_sum.get()[0] != expected_sum)
        throw std::runtime_error("stream output or cross-stream dependency mismatch");
    std::printf("batches=%d items_per_batch=%zu slots=%d wall_median_ms=%.4f "
                "merge_sum=%.1f mismatches=%zu PASS\n",
                kBatches, kItems, slot_count, median(samples),
                static_cast<double>(host_sum.get()[0]), mismatches);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        run_mode(1);
        run_mode(2);
        run_mode(4);
        std::puts("chapter 17 stream pipeline: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 17 stream pipeline: FAIL: %s\n", e.what());
        return 1;
    }
}
