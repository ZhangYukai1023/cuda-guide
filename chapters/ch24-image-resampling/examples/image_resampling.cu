#include "image_io.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
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

class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n)); }
    ~DeviceBytes() { if (ptr_) cudaFree(ptr_); }
    Byte* get() const { return ptr_; }
private:
    Byte* ptr_ = nullptr;
};

enum Method { NEAREST, BILINEAR, BICUBIC, AREA, ROTATE90 };

__host__ __device__ int clamp_index(int v, int limit) {
    return v < 0 ? 0 : (v >= limit ? limit - 1 : v);
}

__host__ __device__ float value_at(const Byte* src, int width, int height, int stride, int x, int y) {
    return src[clamp_index(y, height) * stride + clamp_index(x, width)];
}

__host__ __device__ Byte round_byte(float value) {
    int rounded = static_cast<int>(floorf(value + 0.5f));
    rounded = rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded);
    return static_cast<Byte>(rounded);
}

__host__ __device__ float cubic_weight(float distance) {
    const float a = -0.5f; // Catmull-Rom
    const float t = fabsf(distance);
    if (t < 1.f) return (a + 2.f) * t * t * t - (a + 3.f) * t * t + 1.f;
    if (t < 2.f) return a * t * t * t - 5.f * a * t * t + 8.f * a * t - 4.f * a;
    return 0.f;
}

__host__ __device__ Byte sample_point(const Byte* src, int sw, int sh, int stride,
                                      float sx, float sy, Method method) {
    if (method == NEAREST) {
        return static_cast<Byte>(value_at(src, sw, sh, stride,
                                          static_cast<int>(floorf(sx + 0.5f)),
                                          static_cast<int>(floorf(sy + 0.5f))));
    }
    const int x0 = static_cast<int>(floorf(sx)), y0 = static_cast<int>(floorf(sy));
    if (method == BILINEAR || method == ROTATE90) {
        const float fx = sx - x0, fy = sy - y0;
        const float top = (1.f - fx) * value_at(src, sw, sh, stride, x0, y0) +
                          fx * value_at(src, sw, sh, stride, x0 + 1, y0);
        const float bottom = (1.f - fx) * value_at(src, sw, sh, stride, x0, y0 + 1) +
                             fx * value_at(src, sw, sh, stride, x0 + 1, y0 + 1);
        return round_byte((1.f - fy) * top + fy * bottom);
    }
    float sum = 0.f;
    for (int j = -1; j <= 2; ++j)
        for (int i = -1; i <= 2; ++i)
            sum += cubic_weight(sx - (x0 + i)) * cubic_weight(sy - (y0 + j)) *
                   value_at(src, sw, sh, stride, x0 + i, y0 + j);
    return round_byte(sum);
}

__host__ __device__ Byte area_sample(const Byte* src, int sw, int sh, int stride,
                                     int dw, int dh, int x, int y) {
    const float x0 = static_cast<float>(x) * sw / dw, x1 = static_cast<float>(x + 1) * sw / dw;
    const float y0 = static_cast<float>(y) * sh / dh, y1 = static_cast<float>(y + 1) * sh / dh;
    float sum = 0.f, weight_sum = 0.f;
    for (int iy = static_cast<int>(floorf(y0)); iy < static_cast<int>(ceilf(y1)); ++iy)
        for (int ix = static_cast<int>(floorf(x0)); ix < static_cast<int>(ceilf(x1)); ++ix) {
            const float left = fmaxf(x0, static_cast<float>(ix));
            const float right = fminf(x1, static_cast<float>(ix + 1));
            const float top = fmaxf(y0, static_cast<float>(iy));
            const float bottom = fminf(y1, static_cast<float>(iy + 1));
            const float area = fmaxf(0.f, right - left) * fmaxf(0.f, bottom - top);
            if (area > 0.f && ix < sw && iy < sh) {
                sum += area * value_at(src, sw, sh, stride, ix, iy);
                weight_sum += area;
            }
        }
    return round_byte(sum / weight_sum);
}

__host__ __device__ Byte resample_one(const Byte* src, int sw, int sh, int stride,
                                      int dw, int dh, int x, int y, Method method) {
    if (method == AREA) return area_sample(src, sw, sh, stride, dw, dh, x, y);
    if (method == ROTATE90) return sample_point(src, sw, sh, stride,
                                                static_cast<float>(y), static_cast<float>(sh - 1 - x),
                                                BILINEAR);
    const float sx = (x + 0.5f) * static_cast<float>(sw) / dw - 0.5f;
    const float sy = (y + 0.5f) * static_cast<float>(sh) / dh - 0.5f;
    return sample_point(src, sw, sh, stride, sx, sy, method);
}

__global__ void resample_kernel(const Byte* src, int sw, int sh, int stride,
                                Byte* dst, int dw, int dh, Method method) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    dst[y * dw + x] = resample_one(src, sw, sh, stride, dw, dh, x, y, method);
}

std::vector<Byte> cpu_resample(const std::vector<Byte>& src, int sw, int sh, int stride,
                               int dw, int dh, Method method) {
    std::vector<Byte> expected(static_cast<std::size_t>(dw) * dh);
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            expected[y * dw + x] = resample_one(src.data(), sw, sh, stride, dw, dh, x, y, method);
    return expected;
}

guide_image::Image compact(const std::vector<Byte>& padded, int w, int h, int stride) {
    guide_image::Image image{w, h, 1, std::vector<Byte>(static_cast<std::size_t>(w) * h)};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) image.pixels[y * w + x] = padded[y * stride + x];
    return image;
}

void save(const std::filesystem::path& directory, const std::string& label,
          const std::vector<Byte>& actual, const std::vector<Byte>& expected, int w, int h) {
    guide_image::write_pnm((directory / (label + ".pgm")).string(), {w, h, 1, actual});
    std::vector<Byte> diff(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        diff[i] = static_cast<Byte>(std::abs(static_cast<int>(actual[i]) - expected[i]));
    guide_image::write_pnm((directory / (label + "-diff.pgm")).string(), {w, h, 1, diff});
    if (w <= 16 && h <= 16) {
        guide_image::Image zoom{w * 16, h * 16, 1, std::vector<Byte>(static_cast<std::size_t>(w) * 16 * h * 16)};
        for (int y = 0; y < zoom.height; ++y)
            for (int x = 0; x < zoom.width; ++x)
                zoom.pixels[y * zoom.width + x] = actual[(y / 16) * w + x / 16];
        guide_image::write_pnm((directory / (label + "-zoom.pgm")).string(), zoom);
    }
}

void run_case(const std::string& label, const std::vector<Byte>& src, int sw, int sh,
              int stride, int dw, int dh, Method method,
              const std::filesystem::path& directory, int tolerance = 1) {
    const auto expected = cpu_resample(src, sw, sh, stride, dw, dh, method);
    DeviceBytes din(src.size()), dout(expected.size());
    const auto task_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(din.get(), src.data(), src.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout.get(), 0xa5, expected.size()));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    const dim3 threads(16, 16), blocks((dw + 15) / 16, (dh + 15) / 16);
    resample_kernel<<<blocks, threads>>>(din.get(), sw, sh, stride, dout.get(), dw, dh, method);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<Byte> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size(), cudaMemcpyDeviceToHost));
    const double task_ms = std::chrono::duration<double, std::milli>(Clock::now() - task_start).count();
    int max_error = 0, beyond = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const int error = std::abs(static_cast<int>(actual[i]) - expected[i]);
        max_error = std::max(max_error, error);
        beyond += error > tolerance;
    }
    save(directory, label, actual, expected, dw, dh);
    std::printf("%s %dx%d -> %dx%d max_error=%d beyond_tolerance=%d kernel_ms=%.4f "
                "task_ms=%.4f %s\n", label.c_str(), sw, sh, dw, dh, max_error, beyond,
                static_cast<double>(kernel_ms), task_ms, beyond ? "FAIL" : "PASS");
    if (beyond) throw std::runtime_error(label + " CPU/GPU mismatch");
}

int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("usage: image_resampling [output-dir] [input.pgm]");
        const std::filesystem::path directory = argc >= 2 ? argv[1] : ".";
        std::filesystem::create_directories(directory);
        std::vector<Byte> small(4 * 6, 0xee);
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) small[y * 6 + x] = static_cast<Byte>(20 + 17 * x + 43 * y);
        guide_image::write_pnm((directory / "source-4x4.pgm").string(), compact(small, 4, 4, 6));
        const auto nearest_hand = cpu_resample(small, 4, 4, 6, 7, 5, NEAREST);
        const auto bilinear_hand = cpu_resample(small, 4, 4, 6, 7, 5, BILINEAR);
        const auto rotate_hand = cpu_resample(small, 4, 4, 6, 4, 4, ROTATE90);
        if (nearest_hand[2 * 7 + 3] != 140 || bilinear_hand[2 * 7 + 3] != 110 ||
            rotate_hand[0] != 149 || rotate_hand[3] != 20 ||
            rotate_hand[12] != 200 || rotate_hand[15] != 71)
            throw std::runtime_error("small image hand-calculated coordinates mismatch");
        run_case("nearest-4to7", small, 4, 4, 6, 7, 5, NEAREST, directory, 0);
        run_case("bilinear-4to7", small, 4, 4, 6, 7, 5, BILINEAR, directory);
        run_case("bicubic-4to7", small, 4, 4, 6, 7, 5, BICUBIC, directory);
        run_case("rotate90-4x4", small, 4, 4, 6, 4, 4, ROTATE90, directory, 0);
        std::vector<Byte> checker(8 * 10, 0xee);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) checker[y * 10 + x] = ((x + y) & 1) ? 255 : 0;
        guide_image::write_pnm((directory / "checker-8x8.pgm").string(), compact(checker, 8, 8, 10));
        run_case("checker-nearest-3x3", checker, 8, 8, 10, 3, 3, NEAREST, directory, 0);
        run_case("checker-area-3x3", checker, 8, 8, 10, 3, 3, AREA, directory);
        std::vector<Byte> constant(3 * 8, 0xee);
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 5; ++x) constant[y * 8 + x] = 73;
        for (Method method : {NEAREST, BILINEAR, BICUBIC, AREA}) {
            const char* names[] = {"nearest", "bilinear", "bicubic", "area"};
            const auto expected = cpu_resample(constant, 5, 3, 8, 7, 5, method);
            if (!std::all_of(expected.begin(), expected.end(), [](Byte value) { return value == 73; }))
                throw std::runtime_error("constant image did not remain constant");
            run_case(std::string("constant-") + names[method], constant, 5, 3, 8,
                     7, 5, method, directory, 0);
        }
        if (argc == 3) {
            const auto image = guide_image::read_pnm(argv[2]);
            if (image.channels != 1 || image.width > 2048 || image.height > 2048)
                throw std::invalid_argument("optional input must be P5 and at most 2048x2048");
            run_case("user-bilinear-2x", image.pixels, image.width, image.height, image.width,
                     image.width * 2, image.height * 2, BILINEAR, directory);
        }
        std::puts("chapter 24 image resampling: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 24 image resampling: FAIL: %s\n", e.what());
        return 1;
    }
}
