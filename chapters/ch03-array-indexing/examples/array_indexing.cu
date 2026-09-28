#include "../../../common/cuda_support.cuh"

__global__ void multiply(const int* input, int* output, int n, int factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * factor;
}

__global__ void vector_add(const int* a, const int* b, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) output[i] = a[i] + b[i];
}

__global__ void coordinates(int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) output[y * width + x] = 100 * y + x;
}

int main() {
    return guarded([] {
        for (int n : {1, 7, 256, 1003}) {
            std::vector<int> a(n), b(n), scaled(n), added(n);
            for (int i = 0; i < n; ++i) {
                a[i] = i % 13 - 6;
                b[i] = 2 * i + 1;
                scaled[i] = 3 * a[i];
                added[i] = a[i] + b[i];
            }
            DeviceBuffer<int> da(n), db(n), out(n);
            da.upload(a);
            db.upload(b);
            multiply<<<(n + 127) / 128, 128>>>(da.data, out.data, n, 3);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("multiply", out.download(), scaled);

            vector_add<<<2, 128>>>(da.data, db.data, out.data, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("vector_add", out.download(), added);
        }
        for (auto shape : std::vector<std::pair<int, int>>{{3, 2}, {1, 1}, {37, 19}, {19, 37}}) {
            int width = shape.first, height = shape.second;
            DeviceBuffer<int> out(width * height);
            dim3 block(16, 8);
            dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);
            coordinates<<<grid, block>>>(out.data, width, height);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<int> expected(width * height);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    expected[y * width + x] = 100 * y + x;
            verify("coordinates", out.download(), expected);
        }
    });
}
