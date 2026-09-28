#include <cuda_runtime.h>
#include <cstddef>

namespace guide {
__device__ float affine_device(float x, float scale, float bias);

__global__ void affine_kernel(const float* input, float* output,
                              std::size_t count, float scale, float bias) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = affine_device(input[i], scale, bias);
}

void launch_affine(const float* input, float* output, std::size_t count,
                   float scale, float bias, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((count + 127) / 128);
    affine_kernel<<<blocks, 128, 0, stream>>>(input, output, count, scale, bias);
}
} // namespace guide
