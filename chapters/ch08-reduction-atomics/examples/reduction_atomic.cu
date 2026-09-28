#include "../../../common/cuda_support.cuh"
#include <climits>

struct Candidate { int value, index; };

__host__ __device__ bool better(Candidate a, Candidate b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
}

__global__ void sum_partials(const int* input, int* partial, int n) {
    __shared__ int values[128];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    values[t] = i < n ? input[i] : 0;
    __syncthreads();
    for (int offset = 64; offset > 0; offset /= 2) {
        if (t < offset) values[t] += values[t + offset];
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = values[0];
}

__global__ void max_partials(const int* input, Candidate* partial, int n) {
    __shared__ Candidate values[128];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    values[t] = i < n ? Candidate{input[i], i} : Candidate{INT_MIN, INT_MAX};
    __syncthreads();
    for (int offset = 64; offset > 0; offset /= 2) {
        if (t < offset && better(values[t + offset], values[t]))
            values[t] = values[t + offset];
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = values[0];
}

__global__ void max_finish(const Candidate* partial, Candidate* answer, int n) {
    __shared__ Candidate values[128];
    int t = threadIdx.x;
    values[t] = t < n ? partial[t] : Candidate{INT_MIN, INT_MAX};
    __syncthreads();
    for (int offset = 64; offset > 0; offset /= 2) {
        if (t < offset && better(values[t + offset], values[t]))
            values[t] = values[t + offset];
        __syncthreads();
    }
    if (t == 0) answer[0] = values[0];
}

__global__ void histogram16(const int* input, int* bins, int n) {
    __shared__ int local[16];
    int t = threadIdx.x;
    if (t < 16) local[t] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + t;
    if (i < n) atomicAdd(&local[input[i]], 1);
    __syncthreads();
    if (t < 16) atomicAdd(&bins[t], local[t]);
}

void run_reductions(int n) {
    std::vector<int> input(n);
    int cpu_sum = 0;
    Candidate cpu_max{INT_MIN, INT_MAX};
    for (int i = 0; i < n; ++i) {
        input[i] = i % 11 - 5;
        cpu_sum += input[i];
        Candidate item{input[i], i};
        if (better(item, cpu_max)) cpu_max = item;
    }
    int blocks = (n + 127) / 128;
    DeviceBuffer<int> device_input(n), sum_partial(blocks), sum_final(1);
    device_input.upload(input);
    sum_partials<<<blocks, 128>>>(device_input.data, sum_partial.data, n);
    CUDA_CHECK(cudaGetLastError());
    sum_partials<<<1, 128>>>(sum_partial.data, sum_final.data, blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("sum", sum_final.download(), std::vector<int>{cpu_sum});

    DeviceBuffer<Candidate> max_partial(blocks), max_final(1);
    max_partials<<<blocks, 128>>>(device_input.data, max_partial.data, n);
    CUDA_CHECK(cudaGetLastError());
    max_finish<<<1, 128>>>(max_partial.data, max_final.data, blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Candidate got = max_final.download()[0];
    if (got.value != cpu_max.value || got.index != cpu_max.index)
        throw std::runtime_error("max_index verification failed");
    std::printf("max_index: n=%d value=%d index=%d PASS\n", n, got.value, got.index);
}

void run_histogram(int n, bool all_zero) {
    constexpr int bin_count = 16;
    std::vector<int> input(n), expected(bin_count, 0);
    for (int i = 0; i < n; ++i) {
        input[i] = all_zero ? 0 : i % bin_count;
        ++expected[input[i]];
    }
    DeviceBuffer<int> device_input(n), bins(bin_count);
    device_input.upload(input);
    CUDA_CHECK(cudaMemset(bins.data, 0, bin_count * sizeof(int)));
    histogram16<<<(n + 127) / 128, 128>>>(device_input.data, bins.data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    verify(all_zero ? "histogram_all_zero" : "histogram_cycle",
           bins.download(), expected);
}

int main() {
    return guarded([] {
        for (int n : {1, 7, 128, 1003}) run_reductions(n);
        run_histogram(64, false);
        run_histogram(1003, true);
    });
}
