#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) : size_(n) { check(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(float)), "cudaMalloc"); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* data() const { return data_; }
    void upload(const std::vector<float>& host) {
        if (host.size() != size_) throw std::runtime_error("upload size mismatch");
        check(cudaMemcpy(data_, host.data(), size_ * sizeof(float), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<float> download() const {
        std::vector<float> host(size_);
        check(cudaMemcpy(host.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost), "download");
        return host;
    }
private:
    size_t size_;
    float* data_ = nullptr;
};

class EventTimer {
public:
    EventTimer() {
        check(cudaEventCreate(&start_), "cudaEventCreate start");
        try { check(cudaEventCreate(&stop_), "cudaEventCreate stop"); }
        catch (...) { cudaEventDestroy(start_); throw; }
    }
    ~EventTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { check(cudaEventRecord(start_), "record start"); }
    float stop() {
        check(cudaEventRecord(stop_), "record stop");
        check(cudaEventSynchronize(stop_), "synchronize stop");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_, stop_), "elapsed time");
        return ms;
    }
private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

__global__ void step_1d(const float* old, float* next, int n, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0 || i == n - 1) { next[i] = 0.0f; return; }
    next[i] = old[i] + r * (old[i - 1] - 2.0f * old[i] + old[i + 1]);
}

__global__ void step_2d_naive(const float* old, float* next, int width, int height, float r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) { next[i] = 0.0f; return; }
    next[i] = old[i] + r * (old[i - 1] + old[i + 1] + old[i - width] + old[i + width] - 4.0f * old[i]);
}

constexpr int tile_side = 16;

__global__ void step_2d_tiled(const float* old, float* next, int width, int height, float r) {
    __shared__ float tile[tile_side + 2][tile_side + 2];
    constexpr int side = tile_side + 2;
    int lane = threadIdx.y * tile_side + threadIdx.x;
    for (int q = lane; q < side * side; q += tile_side * tile_side) {
        int tx = q % side, ty = q / side;
        int gx = blockIdx.x * tile_side + tx - 1;
        int gy = blockIdx.y * tile_side + ty - 1;
        tile[ty][tx] = gx >= 0 && gx < width && gy >= 0 && gy < height ? old[gy * width + gx] : 0.0f;
    }
    __syncthreads();
    int x = blockIdx.x * tile_side + threadIdx.x;
    int y = blockIdx.y * tile_side + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) { next[i] = 0.0f; return; }
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
    float center = tile[ty][tx];
    next[i] = center + r * (tile[ty][tx - 1] + tile[ty][tx + 1]
                            + tile[ty - 1][tx] + tile[ty + 1][tx] - 4.0f * center);
}

std::vector<float> initial_1d(int n) {
    std::vector<float> out(n, 0.0f);
    for (int i = n / 3; i < 2 * n / 3; ++i) out[i] = 1.0f;
    return out;
}

std::vector<float> initial_2d(int width, int height) {
    std::vector<float> out(width * height, 0.0f);
    for (int y = height / 3; y < 2 * height / 3; ++y)
        for (int x = width / 3; x < 2 * width / 3; ++x) out[y * width + x] = 1.0f;
    return out;
}

std::vector<double> cpu_1d(const std::vector<float>& input, int steps, double r) {
    int n = static_cast<int>(input.size());
    std::vector<double> old(input.begin(), input.end()), next(n, 0.0);
    for (int step = 0; step < steps; ++step) {
        next.front() = next.back() = 0.0;
        for (int i = 1; i < n - 1; ++i)
            next[i] = old[i] + r * (old[i - 1] - 2.0 * old[i] + old[i + 1]);
        std::swap(old, next);
    }
    return old;
}

std::vector<double> cpu_2d(const std::vector<float>& input, int width, int height, int steps, double r) {
    std::vector<double> old(input.begin(), input.end()), next(width * height, 0.0);
    for (int step = 0; step < steps; ++step) {
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            next[i] = x == 0 || y == 0 || x == width - 1 || y == height - 1 ? 0.0
                : old[i] + r * (old[i - 1] + old[i + 1] + old[i - width] + old[i + width] - 4.0 * old[i]);
        }
        std::swap(old, next);
    }
    return old;
}

void verify(const std::string& name, const std::vector<float>& actual,
            const std::vector<double>& reference, double tolerance) {
    if (actual.size() != reference.size()) throw std::runtime_error(name + " size mismatch");
    double max_error = 0.0, min_value = 1e100, max_value = -1e100;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error(name + " nonfinite value");
        max_error = std::max(max_error, std::abs(static_cast<double>(actual[i]) - reference[i]));
        min_value = std::min(min_value, static_cast<double>(actual[i]));
        max_value = std::max(max_value, static_cast<double>(actual[i]));
    }
    std::cout << name << " max_abs_error=" << max_error << " range=[" << min_value << ',' << max_value
              << "] tolerance=" << tolerance << '\n';
    if (max_error > tolerance || min_value < -1e-5 || max_value > 1.00001) throw std::runtime_error(name + " failed");
}

void one_dimensional() {
    constexpr int n = 128, steps = 100;
    constexpr float r = 0.2f;
    auto input = initial_1d(n);
    auto reference = cpu_1d(input, steps, static_cast<double>(r));
    DeviceBuffer a(n), b(n);
    a.upload(input);
    float *old = a.data(), *next = b.data();
    EventTimer timer;
    timer.start();
    for (int step = 0; step < steps; ++step) {
        step_1d<<<(n + 255) / 256, 256>>>(old, next, n, r);
        std::swap(old, next);
    }
    check(cudaGetLastError(), "1D kernel launch");
    float ms = timer.stop();
    auto output = old == a.data() ? a.download() : b.download();
    verify("1D diffusion", output, reference, 3e-5);
    std::cout << "1D steps=" << steps << " r=" << r << " kernel_time_ms=" << ms << '\n';
}

std::vector<float> two_dimensional(bool tiled, const std::vector<float>& input,
                                   const std::vector<double>& reference, int width, int height,
                                   int steps, float r) {
    DeviceBuffer a(input.size()), b(input.size());
    a.upload(input);
    float *old = a.data(), *next = b.data();
    dim3 block(tile_side, tile_side), grid((width + tile_side - 1) / tile_side,
                                           (height + tile_side - 1) / tile_side);
    EventTimer timer;
    timer.start();
    for (int step = 0; step < steps; ++step) {
        if (tiled) step_2d_tiled<<<grid, block>>>(old, next, width, height, r);
        else step_2d_naive<<<grid, block>>>(old, next, width, height, r);
        std::swap(old, next);
    }
    check(cudaGetLastError(), tiled ? "tiled launch" : "naive launch");
    float ms = timer.stop();
    auto output = old == a.data() ? a.download() : b.download();
    verify(tiled ? "2D tiled diffusion" : "2D naive diffusion", output, reference, 5e-5);
    std::cout << (tiled ? "2D tiled" : "2D naive") << " steps=" << steps
              << " r=" << r << " kernel_time_ms=" << ms << '\n';
    return output;
}

} // namespace

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        std::cout << "GPU=" << prop.name << '\n';
        one_dimensional();
        constexpr int width = 64, height = 48, steps = 80;
        constexpr float r = 0.2f;
        auto input = initial_2d(width, height);
        auto reference = cpu_2d(input, width, height, steps, static_cast<double>(r));
        auto naive = two_dimensional(false, input, reference, width, height, steps, r);
        auto tiled = two_dimensional(true, input, reference, width, height, steps, r);
        double difference = 0.0;
        for (size_t i = 0; i < naive.size(); ++i) difference = std::max(difference, std::abs(static_cast<double>(naive[i] - tiled[i])));
        std::cout << "2D naive/tiled max_abs_difference=" << difference << '\n';
        if (difference > 5e-5) throw std::runtime_error("naive and tiled results differ");
        constexpr int edge_width = 67, edge_height = 51, edge_steps = 40;
        auto edge_input = initial_2d(edge_width, edge_height);
        auto edge_reference = cpu_2d(edge_input, edge_width, edge_height, edge_steps, static_cast<double>(r));
        auto edge_naive = two_dimensional(false, edge_input, edge_reference, edge_width, edge_height, edge_steps, r);
        auto edge_tiled = two_dimensional(true, edge_input, edge_reference, edge_width, edge_height, edge_steps, r);
        double edge_difference = 0.0;
        for (size_t i = 0; i < edge_naive.size(); ++i)
            edge_difference = std::max(edge_difference, std::abs(static_cast<double>(edge_naive[i] - edge_tiled[i])));
        std::cout << "2D partial-tile max_abs_difference=" << edge_difference << '\n';
        if (edge_difference > 5e-5) throw std::runtime_error("partial-tile results differ");
        std::cout << "chapter 33 stencil and heat: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "chapter 33 FAIL: " << e.what() << '\n';
        return 1;
    }
}
