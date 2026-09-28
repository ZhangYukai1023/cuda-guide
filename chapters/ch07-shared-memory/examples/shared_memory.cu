#include "../../../common/cuda_support.cuh"

__global__ void reverse_block(const int* input, int* output) {
    __shared__ int tile[8];
    int t = threadIdx.x;
    tile[t] = input[t];
    __syncthreads();
    output[t] = tile[7 - t];
}

__global__ void reverse_warp(const int* input, int* output) {
    __shared__ int tile[32];
    int t = threadIdx.x;
    tile[t] = input[t];
    __syncwarp();
    output[t] = tile[31 - t];
}

__global__ void transpose_tiled(const int* input, int* output, int width, int height) {
    __shared__ int tile[16][17];
    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    __syncthreads();
    int output_x = blockIdx.y * 16 + threadIdx.x;
    int output_y = blockIdx.x * 16 + threadIdx.y;
    if (output_x < height && output_y < width)
        output[output_y * height + output_x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void block_sums(const int* input, int* partial, int n) {
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

int main() {
    return guarded([] {
        std::vector<int> eight{0, 1, 2, 3, 4, 5, 6, 7};
        DeviceBuffer<int> in8(8), out8(8);
        in8.upload(eight);
        reverse_block<<<1, 8>>>(in8.data, out8.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("reverse_block", out8.download(), std::vector<int>{7, 6, 5, 4, 3, 2, 1, 0});

        std::vector<int> thirty_two(32), reversed(32);
        for (int i = 0; i < 32; ++i) {
            thirty_two[i] = i;
            reversed[i] = 31 - i;
        }
        DeviceBuffer<int> in32(32), out32(32);
        in32.upload(thirty_two);
        reverse_warp<<<1, 32>>>(in32.data, out32.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("reverse_warp", out32.download(), reversed);

        for (auto shape : std::vector<std::pair<int, int>>{{3, 2}, {1, 1}, {31, 17}}) {
            int width = shape.first, height = shape.second;
            std::vector<int> input(width * height), expected(width * height);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    input[y * width + x] = 100 * y + x;
                    expected[x * height + y] = input[y * width + x];
                }
            DeviceBuffer<int> source(input.size()), output(input.size());
            source.upload(input);
            transpose_tiled<<<dim3((width + 15) / 16, (height + 15) / 16),
                              dim3(16, 16)>>>(source.data, output.data, width, height);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("transpose_tiled", output.download(), expected);
        }

        for (int n : {1, 7, 128, 1003}) {
            std::vector<int> input(n);
            for (int i = 0; i < n; ++i) input[i] = i % 11 - 5;
            int blocks = (n + 127) / 128;
            std::vector<int> expected(blocks, 0);
            for (int i = 0; i < n; ++i) expected[i / 128] += input[i];
            DeviceBuffer<int> source(n), partial(blocks);
            source.upload(input);
            block_sums<<<blocks, 128>>>(source.data, partial.data, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("block_sums", partial.download(), expected);
        }
    });
}
