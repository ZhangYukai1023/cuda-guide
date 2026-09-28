#include "array_ops.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace {
constexpr std::size_t kCapacity = 1003;

class HostPinned {
public:
    HostPinned() {
        if (cudaHostAlloc(reinterpret_cast<void**>(&data_), kCapacity * sizeof(float), cudaHostAllocDefault)
            != cudaSuccess) throw std::runtime_error("cudaHostAlloc failed");
    }
    ~HostPinned() { if (data_) cudaFreeHost(data_); }
    float* get() const { return data_; }
private:
    float* data_ = nullptr;
};

void verify(const float* input, const float* output, std::size_t n, float scale, float bias) {
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = input[i] * scale + bias;
        if (std::fabs(output[i] - expected) > 1e-6f)
            throw std::runtime_error("CPU/GPU mismatch");
    }
}
}

int main() {
    try {
        HostPinned input, output;
        for (std::size_t i = 0; i < kCapacity; ++i)
            input.get()[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
        guide::ArrayTransformer transformer(kCapacity);
        transformer.enqueue(input.get(), output.get(), kCapacity, 2.0f, 1.0f);
        bool rejected_overlap = false;
        try { transformer.enqueue(input.get(), output.get(), 7, 1.0f, 0.0f); }
        catch (const std::logic_error&) { rejected_overlap = true; }
        if (!rejected_overlap) throw std::runtime_error("overlapping enqueue was accepted");
        transformer.wait();
        verify(input.get(), output.get(), kCapacity, 2.0f, 1.0f);
        transformer.enqueue(input.get(), output.get(), 7, -1.0f, 0.5f);
        transformer.wait();
        verify(input.get(), output.get(), 7, -1.0f, 0.5f);
        transformer.enqueue(nullptr, nullptr, 0, 1.0f, 0.0f); // documented no-op
        bool rejected_capacity = false;
        try { transformer.enqueue(input.get(), output.get(), kCapacity + 1, 1.0f, 0.0f); }
        catch (const std::out_of_range&) { rejected_capacity = true; }
        if (!rejected_capacity) throw std::runtime_error("oversize input was accepted");
        std::puts("chapter 19 reusable array module: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 19 reusable array module: FAIL: %s\n", e.what());
        return 1;
    }
}
