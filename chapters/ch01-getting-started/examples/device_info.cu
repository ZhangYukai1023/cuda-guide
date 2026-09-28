#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return EXIT_FAILURE;
    }
    std::printf("CUDA devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, i));
        std::printf("Device %d: %s, compute capability %d.%d\n",
                    i, prop.name, prop.major, prop.minor);
    }
    return EXIT_SUCCESS;
}
