#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

constexpr std::size_t kItems = 1003;
constexpr int kIterations = 100;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

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

class Graph {
public:
    void capture(cudaStream_t stream, const int* input, int* temp, int* output);
    ~Graph() {
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
    }
    cudaGraphExec_t get() const { return exec_; }
private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

__global__ void double_values(const int* input, int* temp, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) temp[i] = 2 * input[i];
}

__global__ void add_one(const int* temp, int* output, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = temp[i] + 1;
}

void launch_pair(cudaStream_t stream, const int* input, int* temp, int* output) {
    const unsigned blocks = static_cast<unsigned>((kItems + 127) / 128);
    double_values<<<blocks, 128, 0, stream>>>(input, temp, kItems);
    CUDA_CHECK(cudaGetLastError());
    add_one<<<blocks, 128, 0, stream>>>(temp, output, kItems);
    CUDA_CHECK(cudaGetLastError());
}

void Graph::capture(cudaStream_t stream, const int* input, int* temp, int* output) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_pair(stream, input, temp, output);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void validate(const int* device_output, const std::vector<int>& input) {
    std::vector<int> actual(kItems);
    CUDA_CHECK(cudaMemcpy(actual.data(), device_output, kItems * sizeof(int), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < kItems; ++i)
        if (actual[i] != 2 * input[i] + 1) throw std::runtime_error("CPU/GPU mismatch");
}

void run() {
    std::vector<int> input(kItems);
    for (std::size_t i = 0; i < kItems; ++i) input[i] = static_cast<int>(i % 101) - 50;
    DeviceBuffer<int> din(kItems), dtemp(kItems), dout(kItems);
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), kItems * sizeof(int), cudaMemcpyHostToDevice));
    Stream stream;
    std::vector<double> direct_ms;
    for (int trial = 0; trial < 5; ++trial) {
        const auto start = Clock::now();
        for (int rep = 0; rep < kIterations; ++rep)
            launch_pair(stream.get(), din.get(), dtemp.get(), dout.get());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        direct_ms.push_back(Milliseconds(Clock::now() - start).count());
    }
    validate(dout.get(), input);

    const auto setup_start = Clock::now();
    Graph graph;
    graph.capture(stream.get(), din.get(), dtemp.get(), dout.get());
    const double graph_setup_ms = Milliseconds(Clock::now() - setup_start).count();
    CUDA_CHECK(cudaGraphLaunch(graph.get(), stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get())); // graph upload/first-use warmup
    std::vector<double> graph_ms;
    for (int trial = 0; trial < 5; ++trial) {
        const auto start = Clock::now();
        for (int rep = 0; rep < kIterations; ++rep)
            CUDA_CHECK(cudaGraphLaunch(graph.get(), stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        graph_ms.push_back(Milliseconds(Clock::now() - start).count());
    }
    validate(dout.get(), input);
    std::printf("direct_100_pairs_median_ms=%.4f graph_100_launches_median_ms=%.4f "
                "graph_capture_instantiate_ms=%.4f PASS\n",
                median(direct_ms), median(graph_ms), graph_setup_ms);

    int pool_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&pool_supported, cudaDevAttrMemoryPoolsSupported, 0));
    if (!pool_supported) {
        std::puts("stream_ordered_allocator: SKIP (device reports unsupported)");
        return;
    }
    int* temporary = nullptr;
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&temporary), kItems * sizeof(int), stream.get()));
    launch_pair(stream.get(), din.get(), temporary, dout.get());
    CUDA_CHECK(cudaFreeAsync(temporary, stream.get())); // after both kernels in same stream
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    validate(dout.get(), input);
    std::puts("stream_ordered_allocator: PASS");
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        run();
        std::puts("chapter 18 repeated work: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 18 repeated work: FAIL: %s\n", e.what());
        return 1;
    }
}
