#include "image_io.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (false)

using Byte = std::uint8_t;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T))); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

__device__ __host__ Byte clamp_byte(int value) {
    return static_cast<Byte>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__global__ void invert_gray_roi(const Byte* input, int in_stride, Byte* output, int out_stride,
                                int roi_x, int roi_y, int roi_w, int roi_h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < roi_w && y < roi_h) {
        const int ix = roi_x + x, iy = roi_y + y;
        output[iy * out_stride + ix] = 255 - input[iy * in_stride + ix];
    }
}

__global__ void brightness_gray(const Byte* input, int in_stride, Byte* output, int out_stride,
                                int width, int height, int delta) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        output[y * out_stride + x] = clamp_byte(static_cast<int>(input[y * in_stride + x]) + delta);
}

__global__ void threshold_gray(const Byte* input, int in_stride, Byte* output, int out_stride,
                               int width, int height, int threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        output[y * out_stride + x] = input[y * in_stride + x] >= threshold ? 255 : 0;
}

__global__ void rgb_to_gray(const Byte* input, int in_stride, Byte* output, int out_stride,
                            int width, int height, bool bgr) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const Byte* p = input + y * in_stride + 3 * x;
    const int r = bgr ? p[2] : p[0], g = p[1], b = bgr ? p[0] : p[2];
    output[y * out_stride + x] = static_cast<Byte>((77 * r + 150 * g + 29 * b + 128) >> 8);
}

__global__ void brighten_rgba(const Byte* input, int in_stride, Byte* output, int out_stride,
                              int width, int height, int delta) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const Byte* src = input + y * in_stride + 4 * x;
    Byte* dst = output + y * out_stride + 4 * x;
    for (int c = 0; c < 3; ++c) dst[c] = clamp_byte(static_cast<int>(src[c]) + delta);
    dst[3] = src[3];
}

void launch_check() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::size_t differences(const std::vector<Byte>& actual, const std::vector<Byte>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("size mismatch");
    std::size_t count = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) count += actual[i] != expected[i];
    return count;
}

guide_image::Image compact_gray(const std::vector<Byte>& padded, int width, int height, int stride) {
    guide_image::Image image{width, height, 1, std::vector<Byte>(static_cast<std::size_t>(width) * height)};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.pixels[static_cast<std::size_t>(y) * width + x] = padded[static_cast<std::size_t>(y) * stride + x];
    return image;
}

guide_image::Image zoom_nearest(const guide_image::Image& image, int factor = 16) {
    guide_image::Image zoom{image.width * factor, image.height * factor, image.channels,
                            std::vector<Byte>(static_cast<std::size_t>(image.width) * factor *
                                              image.height * factor * image.channels)};
    for (int y = 0; y < zoom.height; ++y)
        for (int x = 0; x < zoom.width; ++x)
            for (int c = 0; c < zoom.channels; ++c)
                zoom.pixels[(static_cast<std::size_t>(y) * zoom.width + x) * zoom.channels + c] =
                    image.pixels[(static_cast<std::size_t>(y / factor) * image.width + x / factor) * image.channels + c];
    return zoom;
}

void write_gray_result(const std::filesystem::path& out_dir, const std::string& stem,
                       const std::vector<Byte>& actual, const std::vector<Byte>& expected,
                       int width, int height, int stride) {
    std::vector<Byte> difference(expected.size(), 0);
    for (std::size_t i = 0; i < expected.size(); ++i)
        difference[i] = actual[i] > expected[i] ? actual[i] - expected[i] : expected[i] - actual[i];
    const auto image = compact_gray(actual, width, height, stride);
    guide_image::write_pnm((out_dir / (stem + ".pgm")).string(), image);
    guide_image::write_pnm((out_dir / (stem + "-zoom.pgm")).string(), zoom_nearest(image));
    guide_image::write_pnm((out_dir / (stem + "-diff.pgm")).string(),
                           compact_gray(difference, width, height, stride));
}

void run_gray(const std::filesystem::path& out_dir) {
    constexpr int w = 5, h = 3, stride = 8;
    std::vector<Byte> input(stride * h, 0xee);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            input[y * stride + x] = static_cast<Byte>(20 + y * 70 + x * 20);
    const auto input_image = compact_gray(input, w, h, stride);
    guide_image::write_pnm((out_dir / "gray-input.pgm").string(), input_image);
    guide_image::write_pnm((out_dir / "gray-input-zoom.pgm").string(), zoom_nearest(input_image));
    DeviceBuffer<Byte> din(input.size()), dout(input.size());
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
    const dim3 threads(16, 16), blocks(1, 1);
    auto check_and_write = [&](const char* label, const std::vector<Byte>& expected,
                               const std::string& filename) {
        std::vector<Byte> actual(input.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size(), cudaMemcpyDeviceToHost));
        const std::size_t bad = differences(actual, expected);
        std::printf("gray %s 5x3 stride8 mismatches=%zu %s\n", label, bad, bad ? "FAIL" : "PASS");
        write_gray_result(out_dir, filename.substr(0, filename.size() - 4), actual, expected, w, h, stride);
        if (bad) throw std::runtime_error("gray operation mismatch");
    };
    std::vector<Byte> roi = input;
    for (int y = 1; y < h; ++y)
        for (int x = 1; x < 4; ++x) roi[y * stride + x] = 255 - input[y * stride + x];
    CUDA_CHECK(cudaMemcpy(dout.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
    invert_gray_roi<<<blocks, threads>>>(din.get(), stride, dout.get(), stride, 1, 1, 3, 2);
    launch_check();
    check_and_write("roi_invert", roi, "gray-roi-invert.pgm");

    std::vector<Byte> bright = input;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) bright[y * stride + x] = clamp_byte(input[y * stride + x] + 40);
    CUDA_CHECK(cudaMemcpy(dout.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
    brightness_gray<<<blocks, threads>>>(din.get(), stride, dout.get(), stride, w, h, 40);
    launch_check();
    check_and_write("brightness", bright, "gray-bright.pgm");

    std::vector<Byte> binary = input;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) binary[y * stride + x] = input[y * stride + x] >= 128 ? 255 : 0;
    CUDA_CHECK(cudaMemcpy(dout.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
    threshold_gray<<<blocks, threads>>>(din.get(), stride, dout.get(), stride, w, h, 128);
    launch_check();
    check_and_write("threshold", binary, "gray-threshold.pgm");
}

void run_rgb(const std::filesystem::path& out_dir) {
    constexpr int w = 5, h = 3, rgb_stride = 20, gray_stride = 8;
    std::vector<Byte> rgb(rgb_stride * h, 0xdd), bgr(rgb_stride * h, 0xdd);
    guide_image::Image compact{w, h, 3, std::vector<Byte>(w * h * 3)};
    std::vector<Byte> expected(gray_stride * h, 0xcc);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int r = (x * 50 + y * 10) % 256;
            const int g = (x * 20 + y * 60) % 256;
            const int b = (x * 10 + y * 30) % 256;
            Byte* rp = rgb.data() + y * rgb_stride + 3 * x;
            Byte* bp = bgr.data() + y * rgb_stride + 3 * x;
            rp[0] = static_cast<Byte>(r); rp[1] = static_cast<Byte>(g); rp[2] = static_cast<Byte>(b);
            bp[0] = static_cast<Byte>(b); bp[1] = static_cast<Byte>(g); bp[2] = static_cast<Byte>(r);
            for (int c = 0; c < 3; ++c) compact.pixels[(y * w + x) * 3 + c] = rp[c];
            expected[y * gray_stride + x] = static_cast<Byte>((77 * r + 150 * g + 29 * b + 128) >> 8);
        }
    guide_image::write_pnm((out_dir / "rgb-input.ppm").string(), compact);
    guide_image::write_pnm((out_dir / "rgb-input-zoom.ppm").string(), zoom_nearest(compact));
    DeviceBuffer<Byte> din(rgb.size()), dout(expected.size());
    const dim3 threads(16, 16), blocks(1, 1);
    for (bool is_bgr : {false, true}) {
        const auto& input = is_bgr ? bgr : rgb;
        CUDA_CHECK(cudaMemcpy(din.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dout.get(), 0xcc, expected.size()));
        rgb_to_gray<<<blocks, threads>>>(din.get(), rgb_stride, dout.get(), gray_stride, w, h, is_bgr);
        launch_check();
        std::vector<Byte> actual(expected.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size(), cudaMemcpyDeviceToHost));
        const auto bad = differences(actual, expected);
        std::printf("%s_to_gray 5x3 stride20 mismatches=%zu %s\n",
                    is_bgr ? "bgr" : "rgb", bad, bad ? "FAIL" : "PASS");
        write_gray_result(out_dir, is_bgr ? "bgr-to-gray" : "rgb-to-gray", actual, expected,
                          w, h, gray_stride);
        if (bad) throw std::runtime_error("RGB/BGR conversion mismatch");
    }
}

void run_rgba(const std::filesystem::path& out_dir) {
    constexpr int w = 5, h = 3, stride = 24;
    std::vector<Byte> input(stride * h, 0xaa), expected = input;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            Byte* src = input.data() + y * stride + 4 * x;
            Byte* ref = expected.data() + y * stride + 4 * x;
            src[0] = static_cast<Byte>(x * 40); src[1] = static_cast<Byte>(y * 70);
            src[2] = static_cast<Byte>(x * 20 + y * 10); src[3] = static_cast<Byte>(100 + x + y);
            for (int c = 0; c < 3; ++c) ref[c] = clamp_byte(src[c] + 30);
            ref[3] = src[3];
        }
    DeviceBuffer<Byte> din(input.size()), dout(input.size());
    CUDA_CHECK(cudaMemcpy(din.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dout.get(), input.data(), input.size(), cudaMemcpyHostToDevice));
    brighten_rgba<<<1, dim3(16, 16)>>>(din.get(), stride, dout.get(), stride, w, h, 30);
    launch_check();
    std::vector<Byte> actual(input.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), dout.get(), actual.size(), cudaMemcpyDeviceToHost));
    const auto bad = differences(actual, expected);
    std::printf("rgba_brightness_alpha_preserved mismatches=%zu %s\n", bad, bad ? "FAIL" : "PASS");
    guide_image::Image visual{w, h, 3, std::vector<Byte>(w * h * 3)};
    guide_image::Image diff{w, h, 1, std::vector<Byte>(w * h)};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c)
                visual.pixels[(y * w + x) * 3 + c] = actual[y * stride + 4 * x + c];
            int max_diff = 0;
            for (int c = 0; c < 4; ++c) {
                const int delta = static_cast<int>(actual[y * stride + 4 * x + c]) -
                                  static_cast<int>(expected[y * stride + 4 * x + c]);
                if (delta > max_diff) max_diff = delta;
                if (-delta > max_diff) max_diff = -delta;
            }
            diff.pixels[y * w + x] = static_cast<Byte>(max_diff);
        }
    guide_image::write_pnm((out_dir / "rgba-bright-rgb.ppm").string(), visual);
    guide_image::write_pnm((out_dir / "rgba-bright-rgb-zoom.ppm").string(), zoom_nearest(visual));
    guide_image::write_pnm((out_dir / "rgba-bright-diff.pgm").string(), diff);
    if (bad) throw std::runtime_error("RGBA brightness mismatch");
}

void run_user_pgm(const std::string& path, const std::filesystem::path& out_dir) {
    const auto read_start = Clock::now();
    const auto image = guide_image::read_pnm(path);
    const double file_read_ms = Milliseconds(Clock::now() - read_start).count();
    if (image.channels != 1) throw std::runtime_error("optional input must be P5 grayscale");
    const std::size_t bytes = image.pixels.size();
    DeviceBuffer<Byte> din(bytes), dout(bytes);
    std::vector<Byte> result(bytes);
    const auto task_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(din.get(), image.pixels.data(), bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    const dim3 threads(16, 16), blocks((image.width + 15) / 16, (image.height + 15) / 16);
    invert_gray_roi<<<blocks, threads>>>(din.get(), image.width, dout.get(), image.width,
                                         0, 0, image.width, image.height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaMemcpy(result.data(), dout.get(), bytes, cudaMemcpyDeviceToHost));
    const double task_ms = Milliseconds(Clock::now() - task_start).count();
    std::size_t bad = 0;
    for (std::size_t i = 0; i < bytes; ++i) bad += result[i] != 255 - image.pixels[i];
    if (bad) throw std::runtime_error("optional PGM CPU/GPU mismatch");
    const auto write_start = Clock::now();
    guide_image::write_pnm((out_dir / "user-invert.pgm").string(),
                           {image.width, image.height, 1, std::move(result)});
    const double file_write_ms = Milliseconds(Clock::now() - write_start).count();
    std::printf("user_pgm width=%d height=%d kernel_ms=%.4f task_ms=%.4f "
                "file_read_ms=%.4f file_write_ms=%.4f mismatches=0 PASS\n",
                image.width, image.height, static_cast<double>(kernel_ms), task_ms,
                file_read_ms, file_write_ms);
}

int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("usage: image_layout [output-dir] [input.pgm]");
        const std::filesystem::path output_dir = argc >= 2 ? argv[1] : ".";
        std::filesystem::create_directories(output_dir);
        run_gray(output_dir);
        run_rgb(output_dir);
        run_rgba(output_dir);
        if (argc == 3) run_user_pgm(argv[2], output_dir);
        std::puts("chapter 21 image layout: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 21 image layout: FAIL: %s\n", e.what());
        return 1;
    }
}
