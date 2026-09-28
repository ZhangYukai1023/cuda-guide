#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 本章使用立即失败的错误处理，适合独立的小程序。
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

bool run_case(int n, bool print_values) {
    std::vector<int> a(n), b(n), result(n), reference(n);
    for (int i = 0; i < n; ++i) {
        a[i] = i + 1;
        b[i] = 10 * (i + 1);
        reference[i] = a[i] + b[i];
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    int *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    check(cudaMalloc(&device_a, bytes));
    check(cudaMalloc(&device_b, bytes));
    check(cudaMalloc(&device_c, bytes));
    check(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(device_a, device_b, device_c, n);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_a));
    check(cudaFree(device_b));
    check(cudaFree(device_c));
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (result[i] != reference[i]) ++mismatches;
    }
    if (print_values) {
        std::printf("result:");
        for (int value : result) std::printf(" %d", value);
        std::printf("\n");
    }
    std::printf("n=%d, mismatches=%d, %s\n", n, mismatches,
                mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int main() {
    bool passed = true;
    // 1：单元素；8：可手算；129：跨 block 且非整块；1000：稍大规模。
    for (int n : {1, 8, 129, 1000}) {
        if (!run_case(n, n == 8)) passed = false;
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
