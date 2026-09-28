#include "../../../common/cuda_support.cuh"

__global__ void write_42(int* out) { out[0] = 42; }
__global__ void add_scalars(int a, int b, int* out) { out[0] = a + b; }
__global__ void write_ids(int* out) { out[threadIdx.x] = threadIdx.x; }

int main() {
    return guarded([] {
        DeviceBuffer<int> one(1);
        write_42<<<1, 1>>>(one.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("write_42", one.download(), std::vector<int>{42});

        add_scalars<<<1, 1>>>(7, 5, one.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("add_scalars", one.download(), std::vector<int>{12});

        DeviceBuffer<int> eight(8);
        write_ids<<<1, 8>>>(eight.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("write_ids", eight.download(), std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
    });
}
