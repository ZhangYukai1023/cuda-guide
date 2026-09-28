#include "image_io.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
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

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t n) { CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), n * sizeof(T))); }
    ~PinnedBuffer() { if (ptr_) cudaFreeHost(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { if (event_) cudaEventDestroy(event_); }
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

__host__ __device__ int clamp_index(int v, int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

__host__ __device__ Byte median_at(const Byte* src, int width, int height, int x, int y) {
    Byte value[9];
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            value[count++] = src[clamp_index(y + dy, height) * width + clamp_index(x + dx, width)];
    for (int i = 1; i < 9; ++i) {
        const Byte key = value[i];
        int j = i;
        while (j > 0 && value[j - 1] > key) { value[j] = value[j - 1]; --j; }
        value[j] = key;
    }
    return value[4];
}

__host__ __device__ Byte bilinear_at(const Byte* src, int sw, int sh, int dw, int dh,
                                     int x, int y) {
    const float sx = (x + 0.5f) * static_cast<float>(sw) / dw - 0.5f;
    const float sy = (y + 0.5f) * static_cast<float>(sh) / dh - 0.5f;
    const int x0 = static_cast<int>(floorf(sx)), y0 = static_cast<int>(floorf(sy));
    const float fx = sx - x0, fy = sy - y0;
    const int ax = clamp_index(x0, sw), bx = clamp_index(x0 + 1, sw);
    const int ay = clamp_index(y0, sh), by = clamp_index(y0 + 1, sh);
    const float top = (1.f - fx) * src[ay * sw + ax] + fx * src[ay * sw + bx];
    const float bottom = (1.f - fx) * src[by * sw + ax] + fx * src[by * sw + bx];
    int rounded = static_cast<int>(floorf((1.f - fy) * top + fy * bottom + 0.5f));
    rounded = rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded);
    return static_cast<Byte>(rounded);
}

__host__ __device__ float normalize(Byte value) {
    return 2.f * static_cast<float>(value) / 255.f - 1.f;
}

__global__ void median_kernel(const Byte* src, Byte* dst, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        dst[y * width + x] = median_at(src, width, height, x, y);
}

__global__ void scale_kernel(const Byte* src, Byte* dst, int sw, int sh, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dw && y < dh)
        dst[y * dw + x] = bilinear_at(src, sw, sh, dw, dh, x, y);
}

__global__ void normalize_kernel(const Byte* src, float* dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[i] = normalize(src[i]);
}

struct Frame { std::string name; guide_image::Image image; };
struct Expected { std::vector<Byte> preview; std::vector<float> normalized; };
struct Result { std::vector<Byte> preview; std::vector<float> normalized; float kernel_ms = 0; double frame_task_ms = 0; };

Expected cpu_reference(const guide_image::Image& image, int dw, int dh) {
    std::vector<Byte> denoised(image.pixels.size());
    for (int y = 0; y < image.height; ++y)
        for (int x = 0; x < image.width; ++x)
            denoised[y * image.width + x] = median_at(image.pixels.data(), image.width, image.height, x, y);
    Expected expected;
    expected.preview.resize(static_cast<std::size_t>(dw) * dh);
    expected.normalized.resize(expected.preview.size());
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x) {
            const int i = y * dw + x;
            expected.preview[i] = bilinear_at(denoised.data(), image.width, image.height, dw, dh, x, y);
            expected.normalized[i] = normalize(expected.preview[i]);
        }
    return expected;
}

std::vector<Frame> synthetic_frames() {
    std::vector<Frame> frames;
    for (int index = 0; index < 6; ++index) {
        guide_image::Image image{65, 49, 1, std::vector<Byte>(65 * 49)};
        for (int y = 0; y < image.height; ++y)
            for (int x = 0; x < image.width; ++x)
                image.pixels[y * image.width + x] =
                    static_cast<Byte>((x * 3 + y * 5 + index * 17 + ((x * y) & 7)) & 255);
        frames.push_back({"synthetic-" + std::to_string(index), std::move(image)});
    }
    return frames;
}

std::vector<Frame> load_frames(const std::filesystem::path& input_dir) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir))
        if (entry.is_regular_file() && entry.path().extension() == ".pgm") paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    if (paths.empty() || paths.size() > 128)
        throw std::invalid_argument("input directory must contain 1..128 .pgm files");
    std::vector<Frame> frames;
    for (const auto& path : paths) {
        auto image = guide_image::read_pnm(path.string());
        if (image.channels != 1 || image.width > 1024 || image.height > 1024)
            throw std::invalid_argument("all inputs must be P5 and at most 1024x1024");
        if (!frames.empty() && (image.width != frames[0].image.width || image.height != frames[0].image.height))
            throw std::invalid_argument("all input images must have identical dimensions");
        frames.push_back({path.filename().string(), std::move(image)});
    }
    return frames;
}

struct Slot {
    Slot(std::size_t input_count, std::size_t output_count)
        : host_input(input_count), host_preview(output_count), host_norm(output_count),
          device_input(input_count), device_median(input_count),
          device_preview(output_count), device_norm(output_count) {}
    PinnedBuffer<Byte> host_input;
    PinnedBuffer<Byte> host_preview;
    PinnedBuffer<float> host_norm;
    DeviceBuffer<Byte> device_input;
    DeviceBuffer<Byte> device_median;
    DeviceBuffer<Byte> device_preview;
    DeviceBuffer<float> device_norm;
    Stream stream;
    Event kernel_start;
    Event kernel_stop;
    Clock::time_point enqueue_start;
    int pending = -1;
};

void enqueue(Slot& slot, int index, const Frame& frame, int dw, int dh) {
    const int sw = frame.image.width, sh = frame.image.height;
    const std::size_t input_bytes = frame.image.pixels.size(), output_count = static_cast<std::size_t>(dw) * dh;
    std::copy(frame.image.pixels.begin(), frame.image.pixels.end(), slot.host_input.get());
    slot.enqueue_start = Clock::now();
    const cudaStream_t stream = slot.stream.get();
    CUDA_CHECK(cudaMemcpyAsync(slot.device_input.get(), slot.host_input.get(), input_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(slot.kernel_start.get(), stream));
    const dim3 threads(16, 16), input_blocks((sw + 15) / 16, (sh + 15) / 16);
    const dim3 output_blocks((dw + 15) / 16, (dh + 15) / 16);
    median_kernel<<<input_blocks, threads, 0, stream>>>(slot.device_input.get(), slot.device_median.get(), sw, sh);
    CUDA_CHECK(cudaGetLastError());
    scale_kernel<<<output_blocks, threads, 0, stream>>>(slot.device_median.get(), slot.device_preview.get(),
                                                         sw, sh, dw, dh);
    CUDA_CHECK(cudaGetLastError());
    normalize_kernel<<<std::min(256, (static_cast<int>(output_count) + 255) / 256), 256, 0, stream>>>(
        slot.device_preview.get(), slot.device_norm.get(), static_cast<int>(output_count));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot.kernel_stop.get(), stream));
    CUDA_CHECK(cudaMemcpyAsync(slot.host_preview.get(), slot.device_preview.get(), output_count,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot.host_norm.get(), slot.device_norm.get(), output_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    slot.pending = index;
}

void finish(Slot& slot, std::vector<Result>& results, std::size_t output_count) {
    if (slot.pending < 0) return;
    CUDA_CHECK(cudaStreamSynchronize(slot.stream.get()));
    Result& result = results[slot.pending];
    result.frame_task_ms = std::chrono::duration<double, std::milli>(Clock::now() - slot.enqueue_start).count();
    CUDA_CHECK(cudaEventElapsedTime(&result.kernel_ms, slot.kernel_start.get(), slot.kernel_stop.get()));
    result.preview.assign(slot.host_preview.get(), slot.host_preview.get() + output_count);
    result.normalized.assign(slot.host_norm.get(), slot.host_norm.get() + output_count);
    slot.pending = -1;
}

std::vector<Result> run_batch(const std::vector<Frame>& frames, int slot_count,
                              int dw, int dh, double& wall_ms) {
    const std::size_t input_count = frames[0].image.pixels.size();
    const std::size_t output_count = static_cast<std::size_t>(dw) * dh;
    std::array<std::unique_ptr<Slot>, 2> slots;
    for (int i = 0; i < slot_count; ++i)
        slots[i] = std::make_unique<Slot>(input_count, output_count);
    std::vector<Result> results(frames.size());
    const auto start = Clock::now();
    for (std::size_t i = 0; i < frames.size(); ++i) {
        Slot& slot = *slots[i % slot_count];
        finish(slot, results, output_count); // previous frame is done before any slot buffer is reused
        enqueue(slot, static_cast<int>(i), frames[i], dw, dh);
    }
    for (int i = 0; i < slot_count; ++i) finish(*slots[i], results, output_count);
    wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return results;
}

void verify(const char* mode, const std::vector<Frame>& frames,
            const std::vector<Expected>& expected, const std::vector<Result>& results) {
    for (std::size_t f = 0; f < frames.size(); ++f) {
        std::size_t bytes_bad = 0, floats_bad = 0;
        float max_float_error = 0;
        for (std::size_t i = 0; i < expected[f].preview.size(); ++i) {
            bytes_bad += results[f].preview[i] != expected[f].preview[i];
            const float error = std::fabs(results[f].normalized[i] - expected[f].normalized[i]);
            if (error > max_float_error) max_float_error = error;
            floats_bad += !(error <= 1e-6f);
        }
        std::printf("%s frame=%zu source=%s preview_mismatches=%zu norm_mismatches=%zu "
                    "norm_max_error=%.8g kernel_ms=%.4f frame_task_ms=%.4f %s\n",
                    mode, f, frames[f].name.c_str(), bytes_bad, floats_bad,
                    static_cast<double>(max_float_error), static_cast<double>(results[f].kernel_ms),
                    results[f].frame_task_ms,
                    (bytes_bad || floats_bad) ? "FAIL" : "PASS");
        if (bytes_bad || floats_bad) throw std::runtime_error("pipeline CPU/GPU mismatch");
    }
}

void save_results(const std::filesystem::path& directory,
                  const std::vector<Frame>& frames, const std::vector<Result>& results,
                  int dw, int dh) {
    const std::uint16_t endian_probe = 1;
    if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
        *reinterpret_cast<const Byte*>(&endian_probe) != 1)
        throw std::runtime_error("binary float output requires little-endian IEEE 32-bit float");
    std::ofstream manifest(directory / "manifest.tsv");
    if (!manifest) throw std::runtime_error("cannot write output manifest");
    manifest << "index\tsource\twidth\theight\tpreview\ttensor_f32_le\n";
    for (std::size_t i = 0; i < frames.size(); ++i) {
        char stem[32];
        std::snprintf(stem, sizeof(stem), "frame-%03zu", i);
        const std::string preview = std::string(stem) + ".pgm";
        const std::string tensor = std::string(stem) + ".f32";
        guide_image::write_pnm((directory / preview).string(), {dw, dh, 1, results[i].preview});
        std::ofstream output(directory / tensor, std::ios::binary);
        if (!output) throw std::runtime_error("cannot write float tensor");
        output.write(reinterpret_cast<const char*>(results[i].normalized.data()),
                     static_cast<std::streamsize>(results[i].normalized.size() * sizeof(float)));
        if (!output) throw std::runtime_error("failed to write float tensor");
        manifest << i << '\t' << frames[i].name << '\t' << dw << '\t' << dh << '\t'
                 << preview << '\t' << tensor << '\n';
    }
    if (!manifest) throw std::runtime_error("failed to write manifest");
}

int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("usage: image_pipeline [output-dir] [input-dir]");
        const std::filesystem::path directory = argc >= 2 ? argv[1] : ".";
        std::filesystem::create_directories(directory);
        const auto file_start = Clock::now();
        const auto frames = argc == 3 ? load_frames(argv[2]) : synthetic_frames();
        const double file_read_ms = std::chrono::duration<double, std::milli>(Clock::now() - file_start).count();
        const int sw = frames[0].image.width, sh = frames[0].image.height;
        const int dw = (sw + 1) / 2, dh = (sh + 1) / 2;
        std::vector<Expected> expected;
        for (const auto& frame : frames) expected.push_back(cpu_reference(frame.image, dw, dh));
        double serial_ms = 0, double_ms = 0;
        const auto serial = run_batch(frames, 1, dw, dh, serial_ms);
        verify("serial", frames, expected, serial);
        const auto double_slot = run_batch(frames, 2, dw, dh, double_ms);
        verify("double_slot", frames, expected, double_slot);
        const auto write_start = Clock::now();
        save_results(directory, frames, double_slot, dw, dh);
        const double file_write_ms = std::chrono::duration<double, std::milli>(Clock::now() - write_start).count();
        std::printf("frames=%zu source=%dx%d output=%dx%d serial_batch_ms=%.4f "
                    "double_slot_batch_ms=%.4f device_buffer_bytes_per_slot=%zu "
                    "file_read_ms=%.4f file_write_ms=%.4f\n",
                    frames.size(), sw, sh, dw, dh, serial_ms, double_ms,
                    static_cast<std::size_t>(2) * sw * sh + static_cast<std::size_t>(5) * dw * dh,
                    file_read_ms, file_write_ms);
        std::puts("chapter 26 image pipeline: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 26 image pipeline: FAIL: %s\n", e.what());
        return 1;
    }
}
