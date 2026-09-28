#include "image_io.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)

using Byte = std::uint8_t;
using Clock = std::chrono::steady_clock;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

enum Border { ZERO = 0, REPLICATE = 1, REFLECT101 = 2 };

__host__ __device__ int border_index(int i, int n, Border border) {
    if (border == ZERO) return (i < 0 || i >= n) ? -1 : i;
    if (border == REPLICATE || n == 1) return i < 0 ? 0 : (i >= n ? n - 1 : i);
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}

__host__ __device__ Byte sample(const Byte* src, int width, int height, int stride,
                                int x, int y, Border border) {
    const int bx = border_index(x, width, border), by = border_index(y, height, border);
    return bx < 0 || by < 0 ? 0 : src[by * stride + bx];
}

__global__ void mean3_direct(const Byte* src, Byte* dst, int width, int height, int stride, Border border) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int sum = 0;
    for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx)
            sum += sample(src, width, height, stride, x + kx, y + ky, border);
    dst[y * stride + x] = static_cast<Byte>((sum + 4) / 9);
}

__global__ void gaussian5_direct(const Byte* src, Byte* dst, int width, int height,
                                 int stride, Border border) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int weight[5] = {1, 4, 6, 4, 1};
    int sum = 0;
    for (int ky = -2; ky <= 2; ++ky)
        for (int kx = -2; kx <= 2; ++kx)
            sum += weight[ky + 2] * weight[kx + 2] *
                   sample(src, width, height, stride, x + kx, y + ky, border);
    dst[y * stride + x] = static_cast<Byte>((sum + 128) / 256);
}

__global__ void gaussian5_horizontal(const Byte* src, int* temp, int width, int height,
                                     int stride, Border border) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int weight[5] = {1, 4, 6, 4, 1};
    int sum = 0;
    for (int k = -2; k <= 2; ++k)
        sum += weight[k + 2] * sample(src, width, height, stride, x + k, y, border);
    temp[y * stride + x] = sum;
}

__global__ void gaussian5_vertical(const int* temp, Byte* dst, int width, int height,
                                   int stride, Border border) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int weight[5] = {1, 4, 6, 4, 1};
    int sum = 0;
    for (int k = -2; k <= 2; ++k) {
        const int yy = border_index(y + k, height, border);
        if (yy >= 0) sum += weight[k + 2] * temp[yy * stride + x];
    }
    dst[y * stride + x] = static_cast<Byte>((sum + 128) / 256);
}

__global__ void mean3_shared_replicate(const Byte* src, Byte* dst, int width, int height, int stride) {
    __shared__ Byte tile[18][18];
    const int ox = blockIdx.x * 16, oy = blockIdx.y * 16;
    for (int ly = threadIdx.y; ly < 18; ly += 16)
        for (int lx = threadIdx.x; lx < 18; lx += 16)
            tile[ly][lx] = sample(src, width, height, stride, ox + lx - 1, oy + ly - 1, REPLICATE);
    __syncthreads();
    const int x = ox + threadIdx.x, y = oy + threadIdx.y;
    if (x >= width || y >= height) return;
    int sum = 0;
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx)
            sum += tile[threadIdx.y + ky][threadIdx.x + kx];
    dst[y * stride + x] = static_cast<Byte>((sum + 4) / 9);
}

std::vector<Byte> cpu_filter(const std::vector<Byte>& src, int width, int height,
                             int stride, Border border, bool gaussian) {
    std::vector<Byte> dst(src.size(), 0xa5);
    const int radius = gaussian ? 2 : 1;
    const int weight[5] = {1, 4, 6, 4, 1};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int ky = -radius; ky <= radius; ++ky)
                for (int kx = -radius; kx <= radius; ++kx) {
                    const int coeff = gaussian ? weight[ky + 2] * weight[kx + 2] : 1;
                    sum += coeff * sample(src.data(), width, height, stride, x + kx, y + ky, border);
                }
            dst[y * stride + x] = static_cast<Byte>(gaussian ? (sum + 128) / 256 : (sum + 4) / 9);
        }
    return dst;
}

guide_image::Image compact(const std::vector<Byte>& data, int width, int height, int stride) {
    guide_image::Image image{width, height, 1, std::vector<Byte>(static_cast<std::size_t>(width) * height)};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.pixels[static_cast<std::size_t>(y) * width + x] = data[static_cast<std::size_t>(y) * stride + x];
    return image;
}

void verify(const char* name, const std::vector<Byte>& actual, const std::vector<Byte>& expected,
            int width, int height, int stride, const std::filesystem::path& output_dir) {
    std::size_t mismatches = 0;
    std::vector<Byte> diff(actual.size(), 0xa5);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i];
        diff[i] = static_cast<Byte>(actual[i] > expected[i] ? actual[i] - expected[i] : expected[i] - actual[i]);
    }
    guide_image::write_pnm((output_dir / (std::string(name) + ".pgm")).string(), compact(actual, width, height, stride));
    guide_image::write_pnm((output_dir / (std::string(name) + "-diff.pgm")).string(), compact(diff, width, height, stride));
    std::printf("%s width=%d height=%d mismatches=%zu %s\n", name, width, height,
                mismatches, mismatches ? "FAIL" : "PASS");
    if (mismatches) throw std::runtime_error(std::string(name) + " CPU/GPU mismatch");
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }

void run_case(int width, int height, int stride, const std::filesystem::path& output_dir,
              bool save_images) {
    std::vector<Byte> src(static_cast<std::size_t>(height) * stride, 0xee);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            src[static_cast<std::size_t>(y) * stride + x] =
                static_cast<Byte>((13 * x + 47 * y + ((x ^ y) & 3) * 19) & 255);
    if (save_images) guide_image::write_pnm((output_dir / "input.pgm").string(), compact(src, width, height, stride));
    DeviceBuffer<Byte> din(src.size()), dout(src.size());
    DeviceBuffer<int> temp(src.size());
    const dim3 threads(16, 16), blocks((width + 15) / 16, (height + 15) / 16);
    auto execute = [&](const char* name, const std::vector<Byte>& expected, auto launch) {
        const auto task_start = Clock::now();
        CUDA_CHECK(cudaMemcpy(din.get(), src.data(), src.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dout.get(), 0xa5, src.size()));
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float kernel_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        std::vector<Byte> actual(src.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size(), cudaMemcpyDeviceToHost));
        const double task_ms = std::chrono::duration<double, std::milli>(Clock::now() - task_start).count();
        if (save_images) verify(name, actual, expected, width, height, stride, output_dir);
        else {
            std::size_t bad = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) bad += actual[i] != expected[i];
            std::printf("%s width=%d height=%d mismatches=%zu %s\n", name, width, height,
                        bad, bad ? "FAIL" : "PASS");
            if (bad) throw std::runtime_error(std::string(name) + " CPU/GPU mismatch");
        }
        std::printf("%s kernel_ms=%.4f task_ms=%.4f (single run; no file I/O)\n",
                    name, static_cast<double>(kernel_ms), task_ms);
    };
    for (Border border : {ZERO, REPLICATE, REFLECT101}) {
        const char* name = border == ZERO ? "mean3_zero" : border == REPLICATE ? "mean3_replicate" : "mean3_reflect101";
        const auto expected = cpu_filter(src, width, height, stride, border, false);
        execute(name, expected, [&] { mean3_direct<<<blocks, threads>>>(din.get(), dout.get(), width, height, stride, border); });
    }
    const auto gaussian = cpu_filter(src, width, height, stride, REPLICATE, true);
    execute("gaussian5_direct", gaussian, [&] {
        gaussian5_direct<<<blocks, threads>>>(din.get(), dout.get(), width, height, stride, REPLICATE);
    });
    execute("gaussian5_separable", gaussian, [&] {
        gaussian5_horizontal<<<blocks, threads>>>(din.get(), temp.get(), width, height, stride, REPLICATE);
        gaussian5_vertical<<<blocks, threads>>>(temp.get(), dout.get(), width, height, stride, REPLICATE);
    });
    const auto mean = cpu_filter(src, width, height, stride, REPLICATE, false);
    execute("mean3_shared", mean, [&] {
        mean3_shared_replicate<<<blocks, threads>>>(din.get(), dout.get(), width, height, stride);
    });
}

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: mean_gaussian [output-dir]");
        const std::filesystem::path output_dir = argc == 2 ? argv[1] : ".";
        std::filesystem::create_directories(output_dir);
        run_case(5, 5, 8, output_dir, true);
        run_case(17, 19, 24, output_dir, false);
        run_case(513, 257, 520, output_dir, false);
        std::puts("chapter 22 mean and Gaussian: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 22 mean and Gaussian: FAIL: %s\n", e.what());
        return 1;
    }
}
