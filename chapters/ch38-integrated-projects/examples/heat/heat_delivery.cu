#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float r = 0.2f;

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

struct Config {
    int width = 67, height = 51, steps = 50;
    std::filesystem::path output;
};

Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("usage: project_c_heat [--width W --height H --steps S --out DIR]");
        std::string key = argv[i];
        if (key == "--width") c.width = std::stoi(argv[i + 1]);
        else if (key == "--height") c.height = std::stoi(argv[i + 1]);
        else if (key == "--steps") c.steps = std::stoi(argv[i + 1]);
        else if (key == "--out") c.output = argv[i + 1];
        else throw std::runtime_error("unknown option: " + key);
    }
    if (c.width < 3 || c.height < 3 || c.width > 512 || c.height > 512 || c.steps < 1 || c.steps > 1000)
        throw std::runtime_error("limits: width/height 3..512, steps 1..1000");
    return c;
}

class Buffer {
public:
    explicit Buffer(size_t n) : count_(n) { check(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(float)), "cudaMalloc"); }
    ~Buffer() { if (p_) cudaFree(p_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    float* data() const { return p_; }
    void upload(const std::vector<float>& values) {
        if (values.size() != count_) throw std::runtime_error("upload size mismatch");
        check(cudaMemcpy(p_, values.data(), count_ * sizeof(float), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<float> download() const {
        std::vector<float> values(count_);
        check(cudaMemcpy(values.data(), p_, count_ * sizeof(float), cudaMemcpyDeviceToHost), "download");
        return values;
    }
private:
    size_t count_;
    float* p_ = nullptr;
};

__global__ void heat_step(const float* old, float* next, int width, int height, float factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) { next[i] = 0.0f; return; }
    next[i] = old[i] + factor * (old[i - 1] + old[i + 1] + old[i - width] + old[i + width] - 4.0f * old[i]);
}

std::vector<float> initial(Config c) {
    std::vector<float> values(c.width * c.height, 0.0f);
    for (int y = c.height / 3; y < 2 * c.height / 3; ++y)
        for (int x = c.width / 3; x < 2 * c.width / 3; ++x)
            values[y * c.width + x] = 1.0f;
    return values;
}

std::vector<double> cpu_reference(const std::vector<float>& input, Config c) {
    std::vector<double> old(input.begin(), input.end()), next(input.size(), 0.0);
    for (int step = 0; step < c.steps; ++step) {
        for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x) {
            int i = y * c.width + x;
            next[i] = x == 0 || y == 0 || x == c.width - 1 || y == c.height - 1 ? 0.0
                : old[i] + static_cast<double>(r) *
                    (old[i - 1] + old[i + 1] + old[i - c.width] + old[i + c.width] - 4.0 * old[i]);
        }
        old.swap(next);
    }
    return old;
}

void write_outputs(const Config& c, const std::vector<float>& values,
                   double max_error, double gpu_sum, double cpu_sum, float kernel_ms) {
    if (c.output.empty()) return;
    const std::uint16_t one = 1;
    if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
        *reinterpret_cast<const std::uint8_t*>(&one) != 1)
        throw std::runtime_error("heat.f32 requires little-endian IEEE float32 host");
    if (std::filesystem::exists(c.output) && !std::filesystem::is_empty(c.output))
        throw std::runtime_error("output directory is not empty; choose a new path");
    std::filesystem::create_directories(c.output);
    std::ofstream raw(c.output / "heat.f32", std::ios::binary);
    if (!raw) throw std::runtime_error("cannot create heat.f32");
    raw.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!raw) throw std::runtime_error("failed writing heat.f32");
    std::ofstream pgm(c.output / "heat.pgm", std::ios::binary);
    if (!pgm) throw std::runtime_error("cannot create heat.pgm");
    pgm << "P5\n" << c.width << ' ' << c.height << "\n255\n";
    for (float v : values) {
        auto byte = static_cast<std::uint8_t>(std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f));
        pgm.put(static_cast<char>(byte));
    }
    if (!pgm) throw std::runtime_error("failed writing heat.pgm");
    std::ofstream metrics(c.output / "metrics.json");
    if (!metrics) throw std::runtime_error("cannot create metrics.json");
    metrics << std::setprecision(12)
            << "{\n  \"width\": " << c.width << ",\n  \"height\": " << c.height
            << ",\n  \"steps\": " << c.steps << ",\n  \"r\": " << r
            << ",\n  \"max_abs_error_vs_cpu\": " << max_error
            << ",\n  \"gpu_sum\": " << gpu_sum << ",\n  \"cpu_sum\": " << cpu_sum
            << ",\n  \"kernel_ms_single_run\": " << kernel_ms << "\n}\n";
    if (!metrics) throw std::runtime_error("failed writing metrics.json");
}

void run(Config c) {
    auto input = initial(c);
    auto reference = cpu_reference(input, c);
    Buffer first(input.size()), second(input.size());
    first.upload(input);
    float *old = first.data(), *next = second.data();
    dim3 block(16, 16), grid((c.width + 15) / 16, (c.height + 15) / 16);
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "start event");
    check(cudaEventCreate(&stop), "stop event");
    check(cudaEventRecord(start), "record start");
    for (int step = 0; step < c.steps; ++step) {
        heat_step<<<grid, block>>>(old, next, c.width, c.height, r);
        std::swap(old, next);
    }
    check(cudaGetLastError(), "heat kernel launch");
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "synchronize heat");
    float kernel_ms = 0.0f;
    check(cudaEventElapsedTime(&kernel_ms, start, stop), "elapsed heat");
    cudaEventDestroy(start); cudaEventDestroy(stop);
    auto output = old == first.data() ? first.download() : second.download();
    double max_error = 0.0, gpu_sum = 0.0, cpu_sum = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i]) || output[i] < -1e-5f || output[i] > 1.00001f)
            throw std::runtime_error("nonfinite or out-of-range heat value");
        max_error = std::max(max_error, std::abs(static_cast<double>(output[i]) - reference[i]));
        gpu_sum += output[i]; cpu_sum += reference[i];
    }
    double sum_tolerance = std::max(5e-3, 5e-5 * std::abs(cpu_sum));
    std::cout << "size=" << c.width << 'x' << c.height << " steps=" << c.steps
              << " r=" << r << " max_abs_error=" << max_error
              << " sum_error=" << std::abs(gpu_sum - cpu_sum) << " sum_tolerance=" << sum_tolerance
              << " kernel_ms_single_run=" << kernel_ms << '\n';
    if (max_error > 5e-5 || std::abs(gpu_sum - cpu_sum) > sum_tolerance)
        throw std::runtime_error("heat result differs from CPU reference");
    write_outputs(c, output, max_error, gpu_sum, cpu_sum, kernel_ms);
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config c = parse(argc, argv);
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        std::cout << "GPU=" << prop.name << '\n';
        run(c);
        std::cout << "chapter 38 project C heat: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "chapter 38 project C FAIL: " << e.what() << '\n';
        return 1;
    }
}
