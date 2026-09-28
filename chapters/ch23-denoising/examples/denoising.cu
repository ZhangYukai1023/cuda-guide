#include "image_io.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
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
constexpr int width = 64, height = 64, pixels = width * height;
constexpr std::uint32_t seed = 20260928;

class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n)); }
    ~DeviceBytes() { if (ptr_) cudaFree(ptr_); }
    Byte* get() const { return ptr_; }
private:
    Byte* ptr_ = nullptr;
};

__host__ __device__ int clamp_index(int x, int limit) {
    return x < 0 ? 0 : (x >= limit ? limit - 1 : x);
}

__host__ __device__ Byte at(const Byte* input, int x, int y) {
    return input[clamp_index(y, height) * width + clamp_index(x, width)];
}

__global__ void median3_kernel(const Byte* input, Byte* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    Byte values[9];
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) values[n++] = at(input, x + dx, y + dy);
    for (int i = 1; i < 9; ++i) {
        const Byte value = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) { values[j] = values[j - 1]; --j; }
        values[j] = value;
    }
    output[y * width + x] = values[4];
}

__global__ void gaussian5_kernel(const Byte* input, Byte* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int weight[5] = {1, 4, 6, 4, 1};
    int sum = 0;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
            sum += weight[dy + 2] * weight[dx + 2] * at(input, x + dx, y + dy);
    output[y * width + x] = static_cast<Byte>((sum + 128) / 256);
}

__constant__ float spatial_weight[25];
__constant__ float range_weight[256];

__global__ void bilateral5_kernel(const Byte* input, Byte* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int center = input[y * width + x];
    float weighted = 0.f, total = 0.f;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            const int value = at(input, x + dx, y + dy);
            const int difference = value > center ? value - center : center - value;
            const float w = spatial_weight[(dy + 2) * 5 + dx + 2] * range_weight[difference];
            weighted += w * value;
            total += w;
        }
    int rounded = static_cast<int>(weighted / total + 0.5f);
    rounded = rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded);
    output[y * width + x] = static_cast<Byte>(rounded);
}

std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

std::vector<Byte> make_clean() {
    std::vector<Byte> clean(pixels);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const int base = x < 32 ? 45 : 180;
            const int pattern = ((x / 4 + y / 4) & 1) ? 16 : -16;
            clean[y * width + x] = static_cast<Byte>(base + pattern + (y / 8) * 2);
        }
    return clean;
}

std::vector<Byte> salt_pepper(const std::vector<Byte>& clean) {
    auto noisy = clean;
    for (int i = 0; i < pixels; ++i) {
        const unsigned r = mix(seed ^ static_cast<unsigned>(i)) % 1000;
        if (r < 35) noisy[i] = 0;
        else if (r < 70) noisy[i] = 255;
    }
    return noisy;
}

std::vector<Byte> gaussian_noise(const std::vector<Byte>& clean) {
    auto noisy = clean;
    for (int i = 0; i < pixels; ++i) {
        const double u1 = (static_cast<double>(mix(seed + 2u * i)) + 0.5) / 4294967296.0;
        const double u2 = (static_cast<double>(mix(seed + 2u * i + 1u)) + 0.5) / 4294967296.0;
        const double normal = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
        const int value = static_cast<int>(std::lround(clean[i] + 18.0 * normal));
        noisy[i] = static_cast<Byte>(std::max(0, std::min(255, value)));
    }
    return noisy;
}

struct Tables { std::array<float, 25> spatial; std::array<float, 256> range; };

Tables make_tables(float sigma_spatial, float sigma_range) {
    Tables t{};
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
            t.spatial[(dy + 2) * 5 + dx + 2] =
                static_cast<float>(std::exp(-(dx * dx + dy * dy) / (2.0 * sigma_spatial * sigma_spatial)));
    for (int d = 0; d < 256; ++d)
        t.range[d] = static_cast<float>(std::exp(-(d * d) / (2.0 * sigma_range * sigma_range)));
    return t;
}

std::vector<Byte> cpu_median3(const std::vector<Byte>& input) {
    std::vector<Byte> output(pixels);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            std::array<Byte, 9> values{};
            int n = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) values[n++] = at(input.data(), x + dx, y + dy);
            std::sort(values.begin(), values.end());
            output[y * width + x] = values[4];
        }
    return output;
}

std::vector<Byte> cpu_gaussian5(const std::vector<Byte>& input) {
    std::vector<Byte> output(pixels);
    const int weight[5] = {1, 4, 6, 4, 1};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx)
                    sum += weight[dy + 2] * weight[dx + 2] * at(input.data(), x + dx, y + dy);
            output[y * width + x] = static_cast<Byte>((sum + 128) / 256);
        }
    return output;
}

std::vector<Byte> cpu_bilateral5(const std::vector<Byte>& input, const Tables& t) {
    std::vector<Byte> output(pixels);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const int center = input[y * width + x];
            float weighted = 0.f, total = 0.f;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    const int value = at(input.data(), x + dx, y + dy);
                    const int difference = std::abs(value - center);
                    const float w = t.spatial[(dy + 2) * 5 + dx + 2] * t.range[difference];
                    weighted += w * value;
                    total += w;
                }
            const int rounded = static_cast<int>(weighted / total + 0.5f);
            output[y * width + x] = static_cast<Byte>(std::max(0, std::min(255, rounded)));
        }
    return output;
}

double psnr(const std::vector<Byte>& reference, const std::vector<Byte>& candidate) {
    double sq = 0;
    for (int i = 0; i < pixels; ++i) {
        const double d = static_cast<double>(reference[i]) - candidate[i];
        sq += d * d;
    }
    const double mse = sq / pixels;
    return mse == 0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(255.0 * 255.0 / mse);
}

double ssim8(const std::vector<Byte>& reference, const std::vector<Byte>& candidate) {
    const double c1 = (0.01 * 255) * (0.01 * 255);
    const double c2 = (0.03 * 255) * (0.03 * 255);
    double sum_ssim = 0;
    for (int by = 0; by < height; by += 8)
        for (int bx = 0; bx < width; bx += 8) {
            double mx = 0, my = 0;
            for (int dy = 0; dy < 8; ++dy)
                for (int dx = 0; dx < 8; ++dx) {
                    const int i = (by + dy) * width + bx + dx;
                    mx += reference[i]; my += candidate[i];
                }
            mx /= 64; my /= 64;
            double vx = 0, vy = 0, cov = 0;
            for (int dy = 0; dy < 8; ++dy)
                for (int dx = 0; dx < 8; ++dx) {
                    const int i = (by + dy) * width + bx + dx;
                    const double a = reference[i] - mx, b = candidate[i] - my;
                    vx += a * a; vy += b * b; cov += a * b;
                }
            vx /= 64; vy /= 64; cov /= 64;
            sum_ssim += ((2 * mx * my + c1) * (2 * cov + c2)) /
                        ((mx * mx + my * my + c1) * (vx + vy + c2));
        }
    return sum_ssim / ((width / 8) * (height / 8));
}

guide_image::Image image_of(const std::vector<Byte>& pixels_data) {
    return {width, height, 1, pixels_data};
}

void save(const std::filesystem::path& directory, const std::string& stem,
          const std::vector<Byte>& data, const std::vector<Byte>& reference) {
    guide_image::write_pnm((directory / (stem + ".pgm")).string(), image_of(data));
    std::vector<Byte> diff(pixels);
    for (int i = 0; i < pixels; ++i)
        diff[i] = static_cast<Byte>(std::abs(static_cast<int>(data[i]) - reference[i]));
    guide_image::write_pnm((directory / (stem + "-diff.pgm")).string(), image_of(diff));
    guide_image::Image zoom{128, 128, 1, std::vector<Byte>(128 * 128)};
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x)
            zoom.pixels[y * 128 + x] = data[(24 + y / 8) * width + 24 + x / 8];
    guide_image::write_pnm((directory / (stem + "-edge-zoom.pgm")).string(), zoom);
}

template <typename Launch>
void run_filter(const char* noise_name, const char* filter_name,
                const std::vector<Byte>& noisy, const std::vector<Byte>& clean,
                const std::vector<Byte>& expected, DeviceBytes& din, DeviceBytes& dout,
                const std::filesystem::path& directory, int tolerance, Launch launch) {
    const auto task_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(din.get(), noisy.data(), pixels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout.get(), 0xa5, pixels));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<Byte> actual(pixels);
    CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), pixels, cudaMemcpyDeviceToHost));
    const double task_ms = std::chrono::duration<double, std::milli>(Clock::now() - task_start).count();
    int max_error = 0, beyond = 0;
    for (int i = 0; i < pixels; ++i) {
        const int error = std::abs(static_cast<int>(actual[i]) - expected[i]);
        max_error = std::max(max_error, error);
        beyond += error > tolerance;
    }
    const std::string label = std::string(noise_name) + "-" + filter_name;
    save(directory, label, actual, clean);
    std::printf("%s CPU_max_error=%d beyond_tolerance=%d PSNR=%.3f dB SSIM8=%.6f "
                "kernel_ms=%.4f task_ms=%.4f %s\n",
                label.c_str(), max_error, beyond, psnr(clean, actual), ssim8(clean, actual),
                static_cast<double>(kernel_ms), task_ms, beyond ? "FAIL" : "PASS");
    if (beyond) throw std::runtime_error(label + " CPU/GPU mismatch");
}

int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("usage: denoising [output-dir] [clean-64x64.pgm]");
        const std::filesystem::path directory = argc >= 2 ? argv[1] : ".";
        std::filesystem::create_directories(directory);
        std::vector<Byte> clean = make_clean();
        if (argc == 3) {
            const auto image = guide_image::read_pnm(argv[2]);
            if (image.width != width || image.height != height || image.channels != 1)
                throw std::invalid_argument("clean input must be P5 8-bit 64x64");
            clean = image.pixels;
        }
        const auto impulse = salt_pepper(clean), normal = gaussian_noise(clean);
        const Tables tables = make_tables(1.4f, 32.f);
        CUDA_CHECK(cudaMemcpyToSymbol(spatial_weight, tables.spatial.data(), sizeof(float) * 25));
        CUDA_CHECK(cudaMemcpyToSymbol(range_weight, tables.range.data(), sizeof(float) * 256));
        save(directory, "clean", clean, clean);
        save(directory, "salt-pepper", impulse, clean);
        save(directory, "gaussian-noise", normal, clean);
        DeviceBytes din(pixels), dout(pixels);
        const dim3 threads(16, 16), blocks(4, 4);
        for (const auto& entry : {std::pair<const char*, const std::vector<Byte>*>("salt-pepper", &impulse),
                                  std::pair<const char*, const std::vector<Byte>*>("gaussian-noise", &normal)}) {
            const auto& noisy = *entry.second;
            std::printf("%s input PSNR=%.3f dB SSIM8=%.6f\n",
                        entry.first, psnr(clean, noisy), ssim8(clean, noisy));
            run_filter(entry.first, "median3", noisy, clean, cpu_median3(noisy), din, dout, directory, 0,
                       [&] { median3_kernel<<<blocks, threads>>>(din.get(), dout.get()); });
            run_filter(entry.first, "gaussian5", noisy, clean, cpu_gaussian5(noisy), din, dout, directory, 0,
                       [&] { gaussian5_kernel<<<blocks, threads>>>(din.get(), dout.get()); });
            run_filter(entry.first, "bilateral5", noisy, clean, cpu_bilateral5(noisy, tables), din, dout,
                       directory, 1,
                       [&] { bilateral5_kernel<<<blocks, threads>>>(din.get(), dout.get()); });
        }
        std::puts("chapter 23 denoising: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 23 denoising: FAIL: %s\n", e.what());
        return 1;
    }
}
