#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

__global__ void scale_kernel(int n, const float* input, float factor, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * factor;
}

__global__ void scale_relu_kernel(int n, const float* input, float factor, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = input[i] * factor;
        output[i] = value > 0.f ? value : 0.f;
    }
}

void validate(const at::Tensor& x, double factor) {
    TORCH_CHECK(x.is_cuda(), "cuda_guide_ops requires a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "cuda_guide_ops supports float32 only");
    TORCH_CHECK(std::isfinite(factor) && std::isfinite(static_cast<float>(factor)),
                "factor must be finite in float32");
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(), "too many elements for this teaching kernel");
}

at::Tensor run(const at::Tensor& x, double factor, bool activation) {
    validate(x, factor);
    c10::cuda::CUDAGuard guard(x.device());
    const at::Tensor contiguous = x.contiguous();
    at::Tensor output = at::empty(x.sizes(), x.options());
    const int n = static_cast<int>(x.numel());
    if (n == 0) return output;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (n - 1) / 256 + 1;
    if (activation)
        scale_relu_kernel<<<blocks, 256, 0, stream>>>(
            n, contiguous.data_ptr<float>(), static_cast<float>(factor), output.data_ptr<float>());
    else
        scale_kernel<<<blocks, 256, 0, stream>>>(
            n, contiguous.data_ptr<float>(), static_cast<float>(factor), output.data_ptr<float>());
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "cuda_guide_ops kernel launch: ", cudaGetErrorString(error));
    return output;
}

at::Tensor scale_cuda(const at::Tensor& x, double factor) { return run(x, factor, false); }
at::Tensor scale_relu_cuda(const at::Tensor& x, double factor) { return run(x, factor, true); }

TORCH_LIBRARY_IMPL(cuda_guide_ops, CUDA, m) {
    m.impl("scale", &scale_cuda);
    m.impl("scale_relu", &scale_relu_cuda);
}
