#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

namespace guide {

// Owns device buffers and one CUDA stream. Not thread-safe.
// Caller owns pinned host buffers and keeps them alive until wait() returns.
class ArrayTransformer {
public:
    explicit ArrayTransformer(std::size_t capacity);
    ~ArrayTransformer();
    ArrayTransformer(const ArrayTransformer&) = delete;
    ArrayTransformer& operator=(const ArrayTransformer&) = delete;

    void enqueue(const float* pinned_input, float* pinned_output,
                 std::size_t count, float scale, float bias);
    void wait();
    std::size_t capacity() const { return capacity_; }

private:
    std::size_t capacity_ = 0;
    float* device_input_ = nullptr;
    float* device_output_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool busy_ = false;
};

} // namespace guide
