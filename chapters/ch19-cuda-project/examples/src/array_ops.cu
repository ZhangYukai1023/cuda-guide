#include "array_ops.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

namespace guide {
void launch_affine(const float* input, float* output, std::size_t count,
                   float scale, float bias, cudaStream_t stream);

void check(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    throw std::runtime_error("CUDA operation failed");
}

ArrayTransformer::ArrayTransformer(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) throw std::invalid_argument("capacity must be positive");
    try {
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check(cudaMalloc(reinterpret_cast<void**>(&device_input_), capacity * sizeof(float)), "cudaMalloc input");
        check(cudaMalloc(reinterpret_cast<void**>(&device_output_), capacity * sizeof(float)), "cudaMalloc output");
    } catch (...) {
        if (device_input_) cudaFree(device_input_);
        if (device_output_) cudaFree(device_output_);
        if (stream_) cudaStreamDestroy(stream_);
        throw;
    }
}

ArrayTransformer::~ArrayTransformer() {
    if (stream_) cudaStreamSynchronize(stream_);
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
    if (stream_) cudaStreamDestroy(stream_);
}

void ArrayTransformer::enqueue(const float* pinned_input, float* pinned_output,
                               std::size_t count, float scale, float bias) {
    if (busy_) throw std::logic_error("call wait before reusing the transformer");
    if (count > capacity_) throw std::out_of_range("count exceeds capacity");
    if (count == 0) return;
    if (!pinned_input || !pinned_output) throw std::invalid_argument("null host buffer");
    const std::size_t bytes = count * sizeof(float);
    try {
        check(cudaMemcpyAsync(device_input_, pinned_input, bytes,
                              cudaMemcpyHostToDevice, stream_), "H2D");
        launch_affine(device_input_, device_output_, count, scale, bias, stream_);
        check(cudaGetLastError(), "affine kernel launch");
        check(cudaMemcpyAsync(pinned_output, device_output_, bytes,
                              cudaMemcpyDeviceToHost, stream_), "D2H");
        busy_ = true;
    } catch (...) {
        cudaStreamSynchronize(stream_); // drain any partial enqueue before reuse/destruction
        busy_ = false;
        throw;
    }
}

void ArrayTransformer::wait() {
    if (!busy_) return;
    const cudaError_t result = cudaStreamSynchronize(stream_);
    busy_ = false;
    check(result, "cudaStreamSynchronize");
}
} // namespace guide
