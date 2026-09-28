#include <cuda_runtime.h>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(int))); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    int* get() const { return data_; }
private:
    int* data_ = nullptr;
};

__global__ void safe_index(int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2 * i;
}

__global__ void safe_shared_sum(int* out) {
    __shared__ int values[32];
    const int t = threadIdx.x;
    values[t] = t;
    __syncthreads();
    if (t == 0) {
        int sum = 0;
        for (int j = 0; j < 32; ++j) sum += values[j];
        out[0] = sum;
    }
}

__global__ void safe_initialized_read(const int* input, int* out) {
    const int i = threadIdx.x;
    if (i < 8) out[i] = input[i] + i;
}

__global__ void safe_warp_mask(int* out) {
    const int t = threadIdx.x;
    const unsigned mask = __ballot_sync(0xffffffffu, t < 16);
    if (t < 16) __syncwarp(mask);
    if (t == 0) out[0] = 1;
}

// The following kernels are intentionally wrong. They run only with explicit CLI modes.
__global__ void bad_oob(int* out) {
    if (threadIdx.x == 0) out[8] = 42; // allocation has only indices 0..7
}

__global__ void bad_shared_race(int* out) {
    __shared__ volatile int slot;
    slot = threadIdx.x; // 32 writes to the same shared address, no ordering
    __syncthreads();
    if (threadIdx.x == 0) out[0] = slot;
}

__global__ void bad_uninitialized(const int* input, int* out) {
    const int i = threadIdx.x;
    if (i < 8) out[i] = input[i] + i; // input was allocated but never written
}

__global__ void bad_warp_mask(int* out) {
    const int t = threadIdx.x;
    if (t < 17) __syncwarp(0x0000ffffu); // thread 16 does not name itself
    if (t == 0) out[0] = 1;
}

void launch_checked() {
    CUDA_CHECK(cudaGetLastError()); // launch/configuration error
    CUDA_CHECK(cudaDeviceSynchronize()); // asynchronous execution error
}

void run_safe() {
    DeviceBuffer out(32), input(8);
    safe_index<<<1, 32>>>(out.get(), 8);
    launch_checked();
    std::vector<int> actual(8);
    CUDA_CHECK(cudaMemcpy(actual.data(), out.get(), 8 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i)
        if (actual[i] != 2 * i) throw std::runtime_error("safe_index mismatch");

    safe_shared_sum<<<1, 32>>>(out.get());
    launch_checked();
    int sum = -1;
    CUDA_CHECK(cudaMemcpy(&sum, out.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (sum != 496) throw std::runtime_error("safe_shared_sum mismatch");

    CUDA_CHECK(cudaMemset(input.get(), 0, 8 * sizeof(int)));
    safe_initialized_read<<<1, 32>>>(input.get(), out.get());
    launch_checked();
    CUDA_CHECK(cudaMemcpy(actual.data(), out.get(), 8 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i)
        if (actual[i] != i) throw std::runtime_error("safe_initialized_read mismatch");

    safe_warp_mask<<<1, 32>>>(out.get());
    launch_checked();
    int flag = 0;
    CUDA_CHECK(cudaMemcpy(&flag, out.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (flag != 1) throw std::runtime_error("safe_warp_mask mismatch");
    std::puts("safe cases: PASS (index, shared, initialized, warp mask)");
}

void run_bad(const std::string& mode) {
    DeviceBuffer out(8);
    if (mode == "oob") {
        bad_oob<<<1, 32>>>(out.get());
    } else if (mode == "race") {
        bad_shared_race<<<1, 32>>>(out.get());
    } else if (mode == "init") {
        DeviceBuffer input(8);
        bad_uninitialized<<<1, 32>>>(input.get(), out.get());
        launch_checked(); // keep input alive until the kernel finishes
        std::puts("intentional uninitialized read completed; inspect initcheck report");
        return;
    } else if (mode == "sync") {
        bad_warp_mask<<<1, 32>>>(out.get());
    } else {
        throw std::invalid_argument("mode must be safe, oob, race, init, or sync");
    }
    launch_checked();
    std::printf("intentional %s fault completed; inspect sanitizer report\n", mode.c_str());
}

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: debug_cases [safe|oob|race|init|sync]");
        const std::string mode = argc == 2 ? argv[1] : "safe";
        if (mode == "safe") run_safe();
        else run_bad(mode);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "debug_cases: FAIL: %s\n", e.what());
        return 1;
    }
}
