#include "../../../common/cuda_support.cuh"
#include <chrono>

__global__ void square(const int* input, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * input[i];
}

__global__ void increment(int* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ++values[i];
}

__global__ void sum_packed(const int* packed, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = packed[i] + packed[n + i];
}

std::vector<int> square_task(const std::vector<int>& input) {
    DeviceBuffer<int> device_input(input.size()), device_output(input.size());
    device_input.upload(input);
    square<<<1, 128>>>(device_input.data, device_output.data, input.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return device_output.download();
}

int main() {
    return guarded([] {
        std::vector<int> input{-3, -2, -1, 0, 1, 2, 3};
        std::vector<int> squared{9, 4, 1, 0, 1, 4, 9};
        std::vector<int> got;
        constexpr int warmups = 5, repeats = 20;
        for (int i = 0; i < warmups; ++i) got = square_task(input);
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) got = square_task(input);
        auto end = std::chrono::steady_clock::now();
        verify("square", got, squared);
        std::printf("square_end_to_end_mean_ms=%.6f (warmup=%d repeats=%d; allocation through free)\n",
                    std::chrono::duration<double, std::milli>(end - start).count() / repeats,
                    warmups, repeats);

        const int n = 1003, rounds = 5;
        std::vector<int> values(n), expected(n);
        for (int i = 0; i < n; ++i) {
            values[i] = i % 11;
            expected[i] = values[i] + rounds;
        }
        DeviceBuffer<int> reused(n);
        reused.upload(values);
        for (int step = 0; step < rounds; ++step) {
            increment<<<(n + 127) / 128, 128>>>(reused.data, n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("reused_buffer", reused.download(), expected);

        const int pair_count = 7;
        std::vector<int> packed(2 * pair_count), sum_reference(pair_count);
        for (int i = 0; i < pair_count; ++i) {
            packed[i] = i;
            packed[pair_count + i] = 10 + i;
            sum_reference[i] = 10 + 2 * i;
        }
        DeviceBuffer<int> device_packed(packed.size()), device_sum(pair_count);
        device_packed.upload(packed); // One H2D copy for both arrays.
        sum_packed<<<1, 128>>>(device_packed.data, device_sum.data, pair_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("packed_inputs", device_sum.download(), sum_reference);
    });
}
