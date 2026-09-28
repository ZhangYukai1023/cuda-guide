#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int threads = 256;

void check(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
}

template<class T> struct Format;

template<> struct Format<__half> {
    static __host__ __device__ float to_float(__half value) { return __half2float(value); }
    static __host__ __device__ __half from_float(float value) { return __float2half_rn(value); }
    static constexpr const char* name = "FP16";
    static constexpr double tolerance = 2e-3;
    static constexpr double row_sum_tolerance = 5e-3;
};

template<> struct Format<__nv_bfloat16> {
    static __host__ __device__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
    static __host__ __device__ __nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }
    static constexpr const char* name = "BF16";
    static constexpr double tolerance = 2e-2;
    static constexpr double row_sum_tolerance = 3e-2;
};

template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (pointer_) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return pointer_; }
    void upload(const std::vector<T>& host) {
        if (host.size() != count_) throw std::runtime_error("upload count mismatch");
        check(cudaMemcpy(pointer_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> host(count_);
        check(cudaMemcpy(host.data(), pointer_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return host;
    }
private:
    size_t count_;
    T* pointer_ = nullptr;
};

template<class T> __global__ void softmax_rows(const T* input, T* output, int width) {
    __shared__ float scratch[threads];
    const int row = blockIdx.x, tid = threadIdx.x;
    const T* src = input + static_cast<size_t>(row) * width;
    T* dst = output + static_cast<size_t>(row) * width;
    float local_max = -INFINITY;
    for (int x = tid; x < width; x += threads)
        local_max = fmaxf(local_max, Format<T>::to_float(src[x]));
    scratch[tid] = local_max;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];
    float local_sum = 0.0f;
    for (int x = tid; x < width; x += threads)
        local_sum += expf(Format<T>::to_float(src[x]) - maximum);
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (int x = tid; x < width; x += threads) {
        const float probability = expf(Format<T>::to_float(src[x]) - maximum) / denominator;
        dst[x] = Format<T>::from_float(probability);
    }
}

template<class T> void run_case(int rows, int width, bool extreme) {
    const size_t count = static_cast<size_t>(rows) * width;
    std::vector<T> input(count);
    for (int row = 0; row < rows; ++row) for (int col = 0; col < width; ++col) {
        const float value = extreme && row == 0
            ? 1000.0f + static_cast<float>((col % 7) - 3) * 0.25f
            : static_cast<float>(((row * 17 + col * 13) % 29) - 14) * 0.35f;
        input[static_cast<size_t>(row) * width + col] = Format<T>::from_float(value);
    }
    std::vector<T> expected(count);
    for (int row = 0; row < rows; ++row) {
        double maximum = -INFINITY;
        for (int col = 0; col < width; ++col)
            maximum = std::max(maximum, static_cast<double>(Format<T>::to_float(input[static_cast<size_t>(row) * width + col])));
        double denominator = 0.0;
        for (int col = 0; col < width; ++col)
            denominator += std::exp(static_cast<double>(Format<T>::to_float(input[static_cast<size_t>(row) * width + col])) - maximum);
        for (int col = 0; col < width; ++col) {
            const double probability = std::exp(static_cast<double>(Format<T>::to_float(input[static_cast<size_t>(row) * width + col])) - maximum) / denominator;
            expected[static_cast<size_t>(row) * width + col] = Format<T>::from_float(static_cast<float>(probability));
        }
    }
    DeviceBuffer<T> device_input(count), device_output(count);
    device_input.upload(input);
    check(cudaMemset(device_output.data(), 0xa5, count * sizeof(T)), "poison output");
    softmax_rows<T><<<rows, threads>>>(device_input.data(), device_output.data(), width);
    check(cudaGetLastError(), "mixed Softmax launch");
    const auto actual = device_output.download();
    double max_error = 0.0, max_row_sum_error = 0.0;
    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < width; ++col) {
            const size_t index = static_cast<size_t>(row) * width + col;
            const double got = Format<T>::to_float(actual[index]);
            const double want = Format<T>::to_float(expected[index]);
            if (!std::isfinite(got)) throw std::runtime_error("nonfinite mixed Softmax output");
            max_error = std::max(max_error, std::abs(got - want));
            row_sum += got;
        }
        max_row_sum_error = std::max(max_row_sum_error, std::abs(row_sum - 1.0));
    }
    std::cout << Format<T>::name << " rows=" << rows << " width=" << width
              << " extreme=" << extreme << " max_error=" << max_error
              << " max_row_sum_error=" << max_row_sum_error << '\n';
    if (max_error > Format<T>::tolerance || max_row_sum_error > Format<T>::row_sum_tolerance)
        throw std::runtime_error(std::string(Format<T>::name) + " Softmax differs from quantized CPU reference");
}

} // namespace

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        run_case<__half>(1, 5, false);
        run_case<__half>(3, 257, true);
        if (properties.major >= 8) {
            run_case<__nv_bfloat16>(1, 5, false);
            run_case<__nv_bfloat16>(3, 257, true);
        } else {
            std::cout << "SKIP BF16: this example requires compute capability 8.0 or newer\n";
        }
        std::cout << "chapter 28 mixed-precision Softmax: PASS (BF16 may be SKIP)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chapter 28 mixed-precision Softmax: FAIL: " << error.what() << '\n';
        return 1;
    }
}
