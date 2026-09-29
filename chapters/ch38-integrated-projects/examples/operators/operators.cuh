#pragma once
#include <cuda_runtime.h>

void launch_softmax(const float* input, float* output, int rows, int cols, cudaStream_t stream);
void launch_layernorm(const float* input, float* output, int rows, int cols, float epsilon, cudaStream_t stream);
void launch_gemm(const float* input, const float* weights, float* output,
                 int rows, int inner, int out_cols, cudaStream_t stream);
