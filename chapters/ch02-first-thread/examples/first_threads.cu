#include <cuda_runtime.h>
#include <cstdio>

__global__ void write_42(int* out) { out[0] = 42; }
__global__ void add_scalars(int a, int b, int* out) { out[0] = a + b; }
__global__ void write_ids(int* out) { out[threadIdx.x] = threadIdx.x; }

bool check(cudaError_t status, const char* step) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(status));
    return false;
}

bool run(int* device) {
    int host[8] = {};

    write_42<<<1, 1>>>(device);
    if (!check(cudaGetLastError(), "write_42 launch") ||
        !check(cudaDeviceSynchronize(), "write_42 execution") ||
        !check(cudaMemcpy(host, device, sizeof(int), cudaMemcpyDeviceToHost), "copy 42")) return false;
    if (host[0] != 42) return false;
    std::puts("write_42: got=42 expected=42 PASS");

    add_scalars<<<1, 1>>>(7, 5, device);
    if (!check(cudaGetLastError(), "add_scalars launch") ||
        !check(cudaDeviceSynchronize(), "add_scalars execution") ||
        !check(cudaMemcpy(host, device, sizeof(int), cudaMemcpyDeviceToHost), "copy sum")) return false;
    if (host[0] != 7 + 5) return false;
    std::puts("add_scalars: got=12 expected=12 PASS");

    write_ids<<<1, 8>>>(device);
    if (!check(cudaGetLastError(), "write_ids launch") ||
        !check(cudaDeviceSynchronize(), "write_ids execution") ||
        !check(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost), "copy ids")) return false;
    for (int i = 0; i < 8; ++i) {
        if (host[i] != i) {
            std::fprintf(stderr, "write_ids: index=%d got=%d expected=%d FAIL\n", i, host[i], i);
            return false;
        }
    }
    std::puts("write_ids: got=0,1,2,3,4,5,6,7 PASS");
    return true;
}

int main() {
    int* device = nullptr;
    if (!check(cudaMalloc(&device, 8 * sizeof(int)), "cudaMalloc")) return 1;
    const bool result_ok = run(device);
    const bool free_ok = check(cudaFree(device), "cudaFree");
    return result_ok && free_ok ? 0 : 1;
}
