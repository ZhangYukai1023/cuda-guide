#include "../../../common/cuda_support.cuh"

__global__ void warp_layout(int* warps, int* lanes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        warps[i] = threadIdx.x / warpSize;
        lanes[i] = threadIdx.x % warpSize;
    }
}

__global__ void classify(const int* input, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (input[i] >= 0) output[i] = 2 * input[i] + 1;
        else output[i] = 3 * input[i] - 1;
    }
}

__global__ void parity_paths(const int* input, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((i & 1) == 0) output[i] = input[i] + 10;
        else output[i] = input[i] - 10;
    }
}

int predicate_mixed_warps(const std::vector<int>& input, int warp_size) {
    int mixed = 0;
    for (int begin = 0; begin < static_cast<int>(input.size()); begin += warp_size) {
        bool positive = false, negative = false;
        for (int i = begin; i < std::min(begin + warp_size, static_cast<int>(input.size())); ++i) {
            positive |= input[i] >= 0;
            negative |= input[i] < 0;
        }
        mixed += positive && negative;
    }
    return mixed;
}

void run_classification(const char* name, const std::vector<int>& input, int block_size,
                        int warp_size) {
    const int n = static_cast<int>(input.size());
    std::vector<int> expected(n);
    for (int i = 0; i < n; ++i)
        expected[i] = input[i] >= 0 ? 2 * input[i] + 1 : 3 * input[i] - 1;
    DeviceBuffer<int> device_input(n), device_output(n);
    device_input.upload(input);
    classify<<<(n + block_size - 1) / block_size, block_size>>>(
        device_input.data, device_output.data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    verify(name, device_output.download(), expected);
    std::printf("%s: predicate_mixed_warps=%d (host model, not measured branch instructions)\n",
                name, predicate_mixed_warps(input, warp_size));
}

int main() {
    return guarded([] {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("warp_size=%d sm_count=%d\n", prop.warpSize, prop.multiProcessorCount);

        const int n = 64;
        DeviceBuffer<int> warps(n), lanes(n);
        warp_layout<<<1, n>>>(warps.data, lanes.data, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> expected_warps(n), expected_lanes(n);
        for (int i = 0; i < n; ++i) {
            expected_warps[i] = i / prop.warpSize;
            expected_lanes[i] = i % prop.warpSize;
        }
        verify("warp_ids", warps.download(), expected_warps);
        verify("lane_ids", lanes.download(), expected_lanes);

        run_classification("uniform", std::vector<int>(n, 5), 64, prop.warpSize);

        std::vector<int> values(n), parity_expected(n);
        for (int i = 0; i < n; ++i) {
            values[i] = i;
            parity_expected[i] = (i & 1) == 0 ? i + 10 : i - 10;
        }
        DeviceBuffer<int> device_values(n), parity_output(n);
        device_values.upload(values);
        parity_paths<<<1, n>>>(device_values.data, parity_output.data, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("parity", parity_output.download(), parity_expected);

        std::vector<int> alternating(n), grouped(n);
        for (int i = 0; i < n; ++i) {
            alternating[i] = (i & 1) == 0 ? 5 : -5;
            grouped[i] = i < n / 2 ? 5 : -5;
        }
        run_classification("data_alternating", alternating, 64, prop.warpSize);
        run_classification("data_grouped", grouped, 64, prop.warpSize);
        run_classification("data_grouped_block128", grouped, 128, prop.warpSize);
        std::vector<int> irregular(1003);
        for (int i = 0; i < static_cast<int>(irregular.size()); ++i)
            irregular[i] = i % 7 - 3;
        run_classification("irregular_block64", irregular, 64, prop.warpSize);
        run_classification("irregular_block128", irregular, 128, prop.warpSize);
    });
}
