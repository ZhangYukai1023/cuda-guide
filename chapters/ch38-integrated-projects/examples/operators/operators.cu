#include "operators.cuh"
#include <cmath>

namespace {

__global__ void row_softmax(const float* input, float* output, int cols) {
    __shared__ float scratch[256];
    int row = blockIdx.x, col = threadIdx.x;
    float value = col < cols ? input[row * cols + col] : -INFINITY;
    scratch[col] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (col < stride) scratch[col] = fmaxf(scratch[col], scratch[col + stride]);
        __syncthreads();
    }
    float maximum = scratch[0];
    scratch[col] = col < cols ? expf(value - maximum) : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (col < stride) scratch[col] += scratch[col + stride];
        __syncthreads();
    }
    float denominator = scratch[0];
    if (col < cols) output[row * cols + col] = expf(value - maximum) / denominator;
}

__global__ void row_layernorm(const float* input, float* output, int cols, float epsilon) {
    __shared__ float scratch[256];
    int row = blockIdx.x, col = threadIdx.x;
    float value = col < cols ? input[row * cols + col] : 0.0f;
    scratch[col] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (col < stride) scratch[col] += scratch[col + stride];
        __syncthreads();
    }
    float mean = scratch[0] / cols;
    float centered = col < cols ? value - mean : 0.0f;
    scratch[col] = centered * centered;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (col < stride) scratch[col] += scratch[col + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(scratch[0] / cols + epsilon);
    if (col < cols) output[row * cols + col] = centered * inverse;
}

__global__ void row_major_gemm(const float* input, const float* weights, float* output,
                               int rows, int inner, int out_cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * out_cols) return;
    int row = index / out_cols, col = index % out_cols;
    float sum = 0.0f;
    for (int k = 0; k < inner; ++k) sum += input[row * inner + k] * weights[k * out_cols + col];
    output[index] = sum;
}

} // namespace

void launch_softmax(const float* input, float* output, int rows, int cols, cudaStream_t stream) {
    row_softmax<<<rows, 256, 0, stream>>>(input, output, cols);
}

void launch_layernorm(const float* input, float* output, int rows, int cols, float epsilon, cudaStream_t stream) {
    row_layernorm<<<rows, 256, 0, stream>>>(input, output, cols, epsilon);
}

void launch_gemm(const float* input, const float* weights, float* output,
                 int rows, int inner, int out_cols, cudaStream_t stream) {
    row_major_gemm<<<(rows * out_cols + 255) / 256, 256, 0, stream>>>(input, weights, output, rows, inner, out_cols);
}
