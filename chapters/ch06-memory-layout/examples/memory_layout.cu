#include "../../../common/cuda_support.cuh"

struct Particle { int x, y; };

__global__ void read_contiguous(const int* input, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2 * input[i];
}

__global__ void read_strided(const int* input, int* output, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2 * input[i * stride];
}

__global__ void copy_rows(const int* padded, int* compact, int width, int height,
                          int row_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        compact[y * width + x] = padded[y * row_stride + x];
}

__global__ void sum_aos(const Particle* input, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i].x + input[i].y;
}

__global__ void sum_soa(const int* x, const int* y, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = x[i] + y[i];
}

__global__ void add_one(int* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ++values[i];
}

int main() {
    return guarded([] {
        constexpr int stride = 4;
        for (int n : {7, 1003}) {
            std::vector<int> input(n * stride), contiguous(n), strided(n);
            for (int i = 0; i < n * stride; ++i) input[i] = i;
            for (int i = 0; i < n; ++i) {
                contiguous[i] = 2 * input[i];
                strided[i] = 2 * input[i * stride];
            }
            DeviceBuffer<int> device_input(input.size()), output(n);
            device_input.upload(input);
            read_contiguous<<<(n + 127) / 128, 128>>>(device_input.data, output.data, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("contiguous", output.download(), contiguous);
            read_strided<<<(n + 127) / 128, 128>>>(device_input.data, output.data, n, stride);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("strided", output.download(), strided);
        }

        const int width = 5, height = 3, row_stride = 8;
        std::vector<int> padded(height * row_stride, -1), compact(width * height);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                padded[y * row_stride + x] = 100 * y + x;
                compact[y * width + x] = 100 * y + x;
            }
        DeviceBuffer<int> device_padded(padded.size()), device_compact(compact.size());
        device_padded.upload(padded);
        copy_rows<<<dim3(1, 1), dim3(8, 4)>>>(device_padded.data,
                                                 device_compact.data,
                                                 width, height, row_stride);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("row_stride", device_compact.download(), compact);

        for (int n : {7, 1003}) {
            std::vector<Particle> aos(n);
            std::vector<int> x(n), y(n), expected(n);
            for (int i = 0; i < n; ++i) {
                x[i] = i;
                y[i] = 3 * i + 1;
                aos[i] = {x[i], y[i]};
                expected[i] = x[i] + y[i];
            }
            DeviceBuffer<Particle> device_aos(n);
            DeviceBuffer<int> device_x(n), device_y(n), output(n);
            device_aos.upload(aos);
            device_x.upload(x);
            device_y.upload(y);
            sum_aos<<<(n + 127) / 128, 128>>>(device_aos.data, output.data, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("aos", output.download(), expected);
            sum_soa<<<(n + 127) / 128, 128>>>(device_x.data, device_y.data, output.data, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("soa", output.download(), expected);
        }

        const int n = 7;
        std::vector<int> original(n), expected(n);
        for (int i = 0; i < n; ++i) {
            original[i] = i - 3;
            expected[i] = original[i] + 1;
        }
        DeviceBuffer<int> explicit_values(n);
        explicit_values.upload(original);
        add_one<<<1, 128>>>(explicit_values.data, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("explicit_transfer", explicit_values.download(), expected);

        int* managed = nullptr;
        CUDA_CHECK(cudaMallocManaged(&managed, n * sizeof(int)));
        try {
            for (int i = 0; i < n; ++i) managed[i] = original[i];
            add_one<<<1, 128>>>(managed, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("managed_memory", std::vector<int>(managed, managed + n), expected);
        } catch (...) {
            cudaFree(managed);
            throw;
        }
        CUDA_CHECK(cudaFree(managed));
    });
}
