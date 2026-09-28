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

constexpr int kBins = 16;
constexpr std::size_t kCount = 65539; // deliberately not divisible by 64/128/256

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T))); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
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

__global__ void histogram_global(const int* input, int* bins, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[input[i]], 1);
}

__global__ void histogram_local(const int* input, int* bins, std::size_t n) {
    __shared__ int local[kBins];
    const int t = threadIdx.x;
    if (t < kBins) local[t] = 0;
    __syncthreads();
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + t;
    if (i < n) atomicAdd(&local[input[i]], 1);
    __syncthreads();
    if (t < kBins) atomicAdd(&bins[t], local[t]);
}

__global__ void sum_global(const int* input, int* output, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(output, input[i]);
}

__global__ void sum_block_local(const int* input, int* output, std::size_t n) {
    extern __shared__ int values[];
    const int t = threadIdx.x;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + t;
    values[t] = i < n ? input[i] : 0;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (t < offset) values[t] += values[t + offset];
        __syncthreads();
    }
    if (t == 0) atomicAdd(output, values[0]);
}

void launch(bool local, int block, const int* input, int* bins) {
    const unsigned grid = static_cast<unsigned>((kCount + block - 1) / block);
    if (local) histogram_local<<<grid, block>>>(input, bins, kCount);
    else histogram_global<<<grid, block>>>(input, bins, kCount);
    CUDA_CHECK(cudaGetLastError());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
}

void benchmark_sum(bool local, int block, const int* input, int reference,
                   const cudaDeviceProp& device, const char* pattern) {
    DeviceBuffer<int> output(1);
    Events events;
    cudaFuncAttributes attr{};
    int active_blocks = 0;
    const std::size_t dynamic_bytes = local ? static_cast<std::size_t>(block) * sizeof(int) : 0;
    if (local) {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, sum_block_local));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, sum_block_local, block, dynamic_bytes));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, sum_global));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, sum_global, block, 0));
    }
    auto launch_sum = [&]() {
        const unsigned grid = static_cast<unsigned>((kCount + block - 1) / block);
        if (local) sum_block_local<<<grid, block, dynamic_bytes>>>(input, output.get(), kCount);
        else sum_global<<<grid, block>>>(input, output.get(), kCount);
        CUDA_CHECK(cudaGetLastError());
    };
    for (int rep = 0; rep < 3; ++rep) {
        CUDA_CHECK(cudaMemset(output.get(), 0, sizeof(int)));
        launch_sum();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> samples;
    for (int rep = 0; rep < 10; ++rep) {
        CUDA_CHECK(cudaMemset(output.get(), 0, sizeof(int)));
        events.start();
        launch_sum();
        samples.push_back(events.stop());
    }
    int actual = 0;
    CUDA_CHECK(cudaMemcpy(&actual, output.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (actual != reference) throw std::runtime_error("sum CPU/GPU mismatch");
    const double predicted_occupancy = static_cast<double>(active_blocks * block) /
                                       device.maxThreadsPerMultiProcessor;
    std::printf("pattern=%s operation=sum mode=%s block=%d median_ms=%.4f "
                "registers_per_thread=%d shared_bytes=%zu local_bytes=%zu predicted_occupancy=%.3f PASS\n",
                pattern, local ? "block_local" : "global_atomic", block, median(samples),
                attr.numRegs, attr.sharedSizeBytes + dynamic_bytes, attr.localSizeBytes,
                predicted_occupancy);
}

void benchmark_one(bool local, int block, const int* input,
                   const std::vector<int>& reference, const cudaDeviceProp& device,
                   const char* pattern) {
    DeviceBuffer<int> bins(kBins);
    Events events;
    cudaFuncAttributes attr{};
    int active_blocks = 0;
    if (local) {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, histogram_local));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, histogram_local, block, 0));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, histogram_global));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, histogram_global, block, 0));
    }
    const double predicted_occupancy = static_cast<double>(active_blocks * block) /
                                       device.maxThreadsPerMultiProcessor;
    for (int rep = 0; rep < 3; ++rep) {
        CUDA_CHECK(cudaMemset(bins.get(), 0, kBins * sizeof(int)));
        launch(local, block, input, bins.get());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> samples;
    for (int rep = 0; rep < 10; ++rep) {
        CUDA_CHECK(cudaMemset(bins.get(), 0, kBins * sizeof(int)));
        events.start();
        launch(local, block, input, bins.get());
        samples.push_back(events.stop());
    }
    std::vector<int> actual(kBins);
    CUDA_CHECK(cudaMemcpy(actual.data(), bins.get(), kBins * sizeof(int), cudaMemcpyDeviceToHost));
    if (actual != reference) throw std::runtime_error("histogram CPU/GPU mismatch");
    std::printf("pattern=%s operation=histogram mode=%s block=%d median_ms=%.4f registers_per_thread=%d "
                "static_shared_bytes=%zu local_bytes=%zu predicted_occupancy=%.3f PASS\n",
                pattern, local ? "block_local" : "global_atomic", block, median(samples),
                attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes, predicted_occupancy);
}

void run_pattern(bool all_zero, const cudaDeviceProp& device) {
    const char* pattern = all_zero ? "all_zero" : "cycling_0_to_15";
    std::vector<int> input(kCount), reference(kBins, 0);
    for (std::size_t i = 0; i < kCount; ++i) {
        input[i] = all_zero ? 0 : static_cast<int>(i % kBins);
        ++reference[static_cast<std::size_t>(input[i])];
    }
    DeviceBuffer<int> din(kCount);
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), kCount * sizeof(int), cudaMemcpyHostToDevice));
    int sum_reference = 0;
    for (int value : input) sum_reference += value;
    for (int block : {64, 128, 256}) {
        benchmark_sum(false, block, din.get(), sum_reference, device, pattern);
        benchmark_sum(true, block, din.get(), sum_reference, device, pattern);
        benchmark_one(false, block, din.get(), reference, device, pattern);
        benchmark_one(true, block, din.get(), reference, device, pattern);
    }
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        std::printf("GPU=%s SM_count=%d max_threads_per_SM=%d\n",
                    device.name, device.multiProcessorCount, device.maxThreadsPerMultiProcessor);
        run_pattern(false, device);
        run_pattern(true, device);
        std::puts("chapter 15 resource tradeoffs: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 15 resource tradeoffs: FAIL: %s\n", e.what());
        return 1;
    }
}
