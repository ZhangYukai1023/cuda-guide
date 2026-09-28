#include "image_io.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
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

__host__ __device__ Byte zero_at(const Byte* src, int width, int height, int x, int y) {
    return x < 0 || y < 0 || x >= width || y >= height ? 0 : src[y * width + x];
}

__host__ __device__ Byte sobel_at(const Byte* src, int width, int height, int x, int y) {
    const int gx = -zero_at(src,width,height,x-1,y-1) + zero_at(src,width,height,x+1,y-1)
                   -2*zero_at(src,width,height,x-1,y) + 2*zero_at(src,width,height,x+1,y)
                   -zero_at(src,width,height,x-1,y+1) + zero_at(src,width,height,x+1,y+1);
    const int gy = -zero_at(src,width,height,x-1,y-1) - 2*zero_at(src,width,height,x,y-1)
                   -zero_at(src,width,height,x+1,y-1) + zero_at(src,width,height,x-1,y+1)
                   +2*zero_at(src,width,height,x,y+1) + zero_at(src,width,height,x+1,y+1);
    const int ax = gx < 0 ? -gx : gx, ay = gy < 0 ? -gy : gy;
    const int magnitude = ax + ay;
    return static_cast<Byte>(magnitude > 255 ? 255 : magnitude);
}

__global__ void sobel_kernel(const Byte* src, Byte* dst, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) dst[y * width + x] = sobel_at(src, width, height, x, y);
}

__host__ __device__ Byte morph_at(const Byte* src, int width, int height,
                                  int x, int y, bool dilate) {
    Byte result = dilate ? 0 : 255;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const Byte v = zero_at(src, width, height, x + dx, y + dy);
            if (dilate) result = v > result ? v : result;
            else result = v < result ? v : result;
        }
    return result;
}

__global__ void morph_kernel(const Byte* src, Byte* dst, int width, int height, bool dilate) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        dst[y * width + x] = morph_at(src, width, height, x, y, dilate);
}

__global__ void histogram_kernel(const Byte* src, int n, unsigned* hist) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&hist[src[i]], 1u);
}

__global__ void equalization_lut_kernel(const unsigned* hist, Byte* lut, int n) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned cdf = 0, cdf_min = 0;
    for (int v = 0; v < 256; ++v) {
        cdf += hist[v];
        if (!cdf_min && cdf) cdf_min = cdf;
        if (n == static_cast<int>(cdf_min)) lut[v] = static_cast<Byte>(v);
        else lut[v] = cdf <= cdf_min ? 0 :
            static_cast<Byte>((static_cast<unsigned long long>(cdf - cdf_min) * 255 +
                               (n - cdf_min) / 2) / (n - cdf_min));
    }
}

__global__ void apply_lut_kernel(const Byte* src, Byte* dst, int n, const Byte* lut) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[i] = lut[src[i]];
}

std::vector<Byte> cpu_sobel(const std::vector<Byte>& src, int width, int height) {
    std::vector<Byte> dst(src.size());
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) dst[y * width + x] = sobel_at(src.data(), width, height, x, y);
    return dst;
}

std::vector<Byte> cpu_morph(const std::vector<Byte>& src, int width, int height, bool dilate) {
    std::vector<Byte> dst(src.size());
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) dst[y * width + x] = morph_at(src.data(), width, height, x, y, dilate);
    return dst;
}

std::vector<Byte> cpu_equalize(const std::vector<Byte>& src) {
    std::array<unsigned, 256> hist{};
    for (Byte v : src) ++hist[v];
    std::array<Byte, 256> lut{};
    unsigned cdf = 0, cdf_min = 0;
    for (int v = 0; v < 256; ++v) {
        cdf += hist[v];
        if (!cdf_min && cdf) cdf_min = cdf;
        if (src.size() == cdf_min) lut[v] = static_cast<Byte>(v);
        else lut[v] = cdf <= cdf_min ? 0 :
            static_cast<Byte>((static_cast<unsigned long long>(cdf - cdf_min) * 255 +
                               (src.size() - cdf_min) / 2) / (src.size() - cdf_min));
    }
    std::vector<Byte> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) dst[i] = lut[src[i]];
    return dst;
}

void save(const std::filesystem::path& directory, const std::string& label,
          const std::vector<Byte>& actual, const std::vector<Byte>& expected,
          int width, int height) {
    guide_image::write_pnm((directory / (label + ".pgm")).string(), {width, height, 1, actual});
    std::vector<Byte> diff(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        diff[i] = static_cast<Byte>(std::abs(static_cast<int>(actual[i]) - expected[i]));
    guide_image::write_pnm((directory / (label + "-diff.pgm")).string(), {width, height, 1, diff});
    if (width <= 16 && height <= 16) {
        guide_image::Image zoom{width * 16, height * 16, 1,
                                std::vector<Byte>(static_cast<std::size_t>(width) * 16 * height * 16)};
        for (int y = 0; y < zoom.height; ++y)
            for (int x = 0; x < zoom.width; ++x)
                zoom.pixels[y * zoom.width + x] = actual[(y / 16) * width + x / 16];
        guide_image::write_pnm((directory / (label + "-zoom.pgm")).string(), zoom);
    }
}

template <typename Launch>
void execute(const char* label, const std::vector<Byte>& input, const std::vector<Byte>& expected,
             DeviceBuffer<Byte>& din, DeviceBuffer<Byte>& dout, int width, int height,
             const std::filesystem::path& directory, Launch launch) {
    const auto task_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout.get(), 0xa5, expected.size()));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<Byte> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size(), cudaMemcpyDeviceToHost));
    const double task_ms = std::chrono::duration<double, std::milli>(Clock::now() - task_start).count();
    std::size_t bad = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) bad += actual[i] != expected[i];
    save(directory, label, actual, expected, width, height);
    std::printf("%s %dx%d mismatches=%zu kernel_ms=%.4f task_ms=%.4f %s\n",
                label, width, height, bad, static_cast<double>(kernel_ms), task_ms,
                bad ? "FAIL" : "PASS");
    if (bad) throw std::runtime_error(std::string(label) + " CPU/GPU mismatch");
}

void run_image(const std::vector<Byte>& gray, int width, int height,
               const std::filesystem::path& directory, const std::string& prefix) {
    const int n = width * height;
    guide_image::write_pnm((directory / (prefix + "input.pgm")).string(), {width, height, 1, gray});
    std::vector<Byte> binary(gray.size());
    for (int i = 0; i < n; ++i) binary[i] = gray[i] >= 128 ? 255 : 0;
    guide_image::write_pnm((directory / (prefix + "binary.pgm")).string(), {width, height, 1, binary});
    DeviceBuffer<Byte> din(n), dout(n), temp(n);
    DeviceBuffer<unsigned> hist(256);
    DeviceBuffer<Byte> lut(256);
    const dim3 threads(16, 16), blocks((width + 15) / 16, (height + 15) / 16);
    const int linear_blocks = std::min(256, (n + 255) / 256);
    execute((prefix + "sobel").c_str(), gray, cpu_sobel(gray, width, height), din, dout,
            width, height, directory,
            [&] { sobel_kernel<<<blocks, threads>>>(din.get(), dout.get(), width, height); });
    execute((prefix + "dilate").c_str(), binary, cpu_morph(binary, width, height, true), din, dout,
            width, height, directory,
            [&] { morph_kernel<<<blocks, threads>>>(din.get(), dout.get(), width, height, true); });
    execute((prefix + "erode").c_str(), binary, cpu_morph(binary, width, height, false), din, dout,
            width, height, directory,
            [&] { morph_kernel<<<blocks, threads>>>(din.get(), dout.get(), width, height, false); });
    const auto opened = cpu_morph(cpu_morph(binary, width, height, false), width, height, true);
    execute((prefix + "open").c_str(), binary, opened, din, dout, width, height, directory,
            [&] {
                morph_kernel<<<blocks, threads>>>(din.get(), temp.get(), width, height, false);
                morph_kernel<<<blocks, threads>>>(temp.get(), dout.get(), width, height, true);
            });
    const auto closed = cpu_morph(cpu_morph(binary, width, height, true), width, height, false);
    execute((prefix + "close").c_str(), binary, closed, din, dout, width, height, directory,
            [&] {
                morph_kernel<<<blocks, threads>>>(din.get(), temp.get(), width, height, true);
                morph_kernel<<<blocks, threads>>>(temp.get(), dout.get(), width, height, false);
            });
    execute((prefix + "equalized").c_str(), gray, cpu_equalize(gray), din, dout,
            width, height, directory,
            [&] {
                CUDA_CHECK(cudaMemset(hist.get(), 0, 256 * sizeof(unsigned)));
                histogram_kernel<<<linear_blocks, 256>>>(din.get(), n, hist.get());
                equalization_lut_kernel<<<1, 1>>>(hist.get(), lut.get(), n);
                apply_lut_kernel<<<linear_blocks, 256>>>(din.get(), dout.get(), n, lut.get());
            });
}

int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("usage: image_features [output-dir] [input.pgm]");
        const std::filesystem::path directory = argc >= 2 ? argv[1] : ".";
        std::filesystem::create_directories(directory);
        const std::vector<Byte> small = {
              0,  0,  0,  0,  0,
              0, 50, 50, 50,  0,
              0, 50,200, 50,  0,
              0, 50, 50, 50,  0,
              0,  0,  0,  0,  0
        };
        if (cpu_sobel(small, 5, 5)[2 * 5 + 2] != 0 ||
            cpu_morph(std::vector<Byte>{0,0,0,0,255,0,0,0,0}, 3, 3, true)[0] != 255)
            throw std::runtime_error("small hand-check failed");
        run_image(small, 5, 5, directory, "small-");
        const std::vector<Byte> constant(7 * 9, 73);
        if (cpu_equalize(constant) != constant)
            throw std::runtime_error("constant histogram policy failed");
        if (cpu_equalize(std::vector<Byte>{0, 0, 128, 255}) !=
            std::vector<Byte>{0, 0, 128, 255})
            throw std::runtime_error("small histogram hand-check failed");
        run_image(constant, 7, 9, directory, "constant-");
        std::vector<Byte> medium(257 * 193);
        for (int y = 0; y < 193; ++y)
            for (int x = 0; x < 257; ++x)
                medium[y * 257 + x] = static_cast<Byte>((x / 3 + y / 2 + (x * y) % 17) & 255);
        run_image(medium, 257, 193, directory, "medium-");
        if (argc == 3) {
            const auto image = guide_image::read_pnm(argv[2]);
            if (image.channels != 1 || image.width > 2048 || image.height > 2048)
                throw std::invalid_argument("optional input must be P5 and at most 2048x2048");
            run_image(image.pixels, image.width, image.height, directory, "user-");
        }
        std::puts("chapter 25 image features: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 25 image features: FAIL: %s\n", e.what());
        return 1;
    }
}
