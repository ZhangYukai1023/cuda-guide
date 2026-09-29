#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int devices_used = 2;
constexpr int images = 4;
constexpr int width = 32, height = 24, pixels = width * height;
constexpr int images_per_device = images / devices_used;
constexpr int local_pixels = images_per_device * pixels;

void check(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
}

int owner(int image) { return image % devices_used; }
int local_index(int image) { return image / devices_used; }

float input_value(int image, int pixel) {
    return static_cast<float>((pixel * 13 + image * 17) % 256) / 255.0f;
}

float transform_cpu(float input, int image) {
    float gain = 1.1f + 0.1f * image;
    return std::min(1.0f, std::max(0.0f, input * gain + 0.05f));
}

std::vector<float> make_input(int device) {
    std::vector<float> data(local_pixels);
    for (int image = 0; image < images; ++image) if (owner(image) == device) {
        int base = local_index(image) * pixels;
        for (int p = 0; p < pixels; ++p) data[base + p] = input_value(image, p);
    }
    return data;
}

double cpu_reference_sum() {
    double sum = 0.0;
    for (int image = 0; image < images; ++image)
        for (int p = 0; p < pixels; ++p) sum += transform_cpu(input_value(image, p), image);
    return sum;
}

void partition_selftest() {
    std::vector<int> seen(images, 0);
    for (int device = 0; device < devices_used; ++device) {
        auto data = make_input(device);
        if (data.size() != local_pixels) throw std::runtime_error("partition input size");
        for (int image = 0; image < images; ++image) if (owner(image) == device) {
            ++seen[image];
            for (int p = 0; p < pixels; ++p)
                if (data[local_index(image) * pixels + p] != input_value(image, p))
                    throw std::runtime_error("partition data mismatch");
        }
    }
    for (int count : seen) if (count != 1) throw std::runtime_error("image assigned more or less than once");
    std::cout << "partition selftest: 4 images, 2 planned devices, " << pixels
              << " pixels/image, expected_sum=" << cpu_reference_sum() << " PASS\n";
}

__global__ void process_images(const float* input, float* output, double* sum,
                               int first_global_image, int image_stride, int total_pixels, int pixels_per_image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pixels) return;
    int local_image = i / pixels_per_image;
    int image = first_global_image + image_stride * local_image;
    float gain = 1.1f + 0.1f * image;
    float value = fminf(1.0f, fmaxf(0.0f, input[i] * gain + 0.05f));
    output[i] = value;
    atomicAdd(sum, static_cast<double>(value));
}

__global__ void add_two_sums(const double* first, const double* second, double* result) {
    result[0] = first[0] + second[0];
}

struct DeviceState {
    explicit DeviceState(int device_id) : id(device_id), host_input(make_input(device_id)), host_output(local_pixels) {}
    ~DeviceState() {
        cudaSetDevice(id);
        if (stream) cudaStreamDestroy(stream);
        if (device_input) cudaFree(device_input);
        if (device_output) cudaFree(device_output);
        if (device_sum) cudaFree(device_sum);
    }
    DeviceState(const DeviceState&) = delete;
    DeviceState& operator=(const DeviceState&) = delete;
    void init() {
        check(cudaSetDevice(id), "cudaSetDevice");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check(cudaMalloc(reinterpret_cast<void**>(&device_input), local_pixels * sizeof(float)), "cudaMalloc input");
        check(cudaMalloc(reinterpret_cast<void**>(&device_output), local_pixels * sizeof(float)), "cudaMalloc output");
        check(cudaMalloc(reinterpret_cast<void**>(&device_sum), sizeof(double)), "cudaMalloc sum");
    }
    void launch() {
        check(cudaSetDevice(id), "cudaSetDevice launch");
        check(cudaMemcpy(device_input, host_input.data(), local_pixels * sizeof(float),
                         cudaMemcpyHostToDevice), "copy input");
        check(cudaMemsetAsync(device_sum, 0, sizeof(double), stream), "zero sum");
        process_images<<<(local_pixels + 255) / 256, 256, 0, stream>>>(device_input, device_output,
                                                                          device_sum, id, devices_used, local_pixels, pixels);
        check(cudaGetLastError(), "process_images launch");
    }
    void wait() {
        check(cudaSetDevice(id), "cudaSetDevice wait");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        check(cudaMemcpy(host_output.data(), device_output, local_pixels * sizeof(float),
                         cudaMemcpyDeviceToHost), "copy output");
        check(cudaMemcpy(&host_sum, device_sum, sizeof(double), cudaMemcpyDeviceToHost), "copy sum");
    }
    int id;
    cudaStream_t stream{};
    float* device_input = nullptr;
    float* device_output = nullptr;
    double* device_sum = nullptr;
    std::vector<float> host_input, host_output;
    double host_sum = 0.0;
};

template<class T> class DeviceArray {
public:
    explicit DeviceArray(size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)), "baseline cudaMalloc"); }
    ~DeviceArray() { if (p_) cudaFree(p_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* data() const { return p_; }
private:
    T* p_ = nullptr;
};

void one_gpu_baseline() {
    check(cudaSetDevice(0), "baseline cudaSetDevice");
    constexpr int count = images * pixels;
    std::vector<float> input(count), output(count);
    for (int image = 0; image < images; ++image)
        for (int p = 0; p < pixels; ++p) input[image * pixels + p] = input_value(image, p);
    DeviceArray<float> d_input(count), d_output(count);
    DeviceArray<double> d_sum(1);
    check(cudaMemcpy(d_input.data(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice), "baseline upload");
    check(cudaMemset(d_sum.data(), 0, sizeof(double)), "baseline zero sum");
    process_images<<<(count + 255) / 256, 256>>>(d_input.data(), d_output.data(), d_sum.data(),
                                                  0, 1, count, pixels);
    check(cudaGetLastError(), "baseline launch");
    double gpu_sum = 0.0;
    check(cudaMemcpy(output.data(), d_output.data(), count * sizeof(float), cudaMemcpyDeviceToHost), "baseline download");
    check(cudaMemcpy(&gpu_sum, d_sum.data(), sizeof(double), cudaMemcpyDeviceToHost), "baseline download sum");
    double max_error = 0.0;
    for (int image = 0; image < images; ++image) for (int p = 0; p < pixels; ++p)
        max_error = std::max(max_error, std::abs(static_cast<double>(output[image * pixels + p])
                                                - transform_cpu(input_value(image, p), image)));
    std::cout << "one-GPU baseline max_pixel_error=" << max_error << " sum=" << gpu_sum << '\n';
    if (max_error > 2e-6 || std::abs(gpu_sum - cpu_reference_sum()) > 1e-6)
        throw std::runtime_error("one-GPU baseline differs from CPU");
}

void verify_pixels(const DeviceState& state) {
    double max_error = 0.0, downloaded_sum = 0.0;
    for (int image = 0; image < images; ++image) if (owner(image) == state.id) {
        int base = local_index(image) * pixels;
        for (int p = 0; p < pixels; ++p) {
            float actual = state.host_output[base + p];
            float expected = transform_cpu(input_value(image, p), image);
            if (!std::isfinite(actual)) throw std::runtime_error("nonfinite image pixel");
            max_error = std::max(max_error, std::abs(static_cast<double>(actual) - expected));
            downloaded_sum += actual;
        }
    }
    std::cout << "device=" << state.id << " max_pixel_error=" << max_error
              << " GPU_sum=" << state.host_sum << " downloaded_sum=" << downloaded_sum << '\n';
    if (max_error > 2e-6 || std::abs(state.host_sum - downloaded_sum) > 1e-7)
        throw std::runtime_error("device image output failed CPU comparison");
}

void two_gpu_test() {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < devices_used) {
        std::cout << "SKIP two-GPU path: found " << count << " CUDA device(s), requires 2\n";
        return;
    }
    cudaDeviceProp properties[devices_used]{};
    for (int id = 0; id < devices_used; ++id) {
        check(cudaGetDeviceProperties(&properties[id], id), "cudaGetDeviceProperties");
        std::cout << "device " << id << "=" << properties[id].name << '\n';
    }
    int peer01 = 0, peer10 = 0;
    check(cudaDeviceCanAccessPeer(&peer01, 0, 1), "P2P 0 to 1");
    check(cudaDeviceCanAccessPeer(&peer10, 1, 0), "P2P 1 to 0");
    std::cout << "P2P access 0->1=" << peer01 << " 1->0=" << peer10 << '\n';
    one_gpu_baseline();
    DeviceState first(0), second(1);
    first.init(); second.init();
    first.launch(); second.launch();
    first.wait(); second.wait();
    verify_pixels(first); verify_pixels(second);
    double cpu_sum = cpu_reference_sum();
    double host_merged = first.host_sum + second.host_sum;
    if (std::abs(host_merged - cpu_sum) > 1e-6) throw std::runtime_error("host merged sum differs from CPU");
    double merged = host_merged;
    const char* merge_path = "host staging (no 0->1 peer access)";
    if (peer01) {
        check(cudaSetDevice(0), "cudaSetDevice P2P");
        check(cudaDeviceEnablePeerAccess(1, 0), "cudaDeviceEnablePeerAccess");
        double *remote_copy = nullptr, *device_merged = nullptr;
        try {
            check(cudaMalloc(reinterpret_cast<void**>(&remote_copy), sizeof(double)), "cudaMalloc peer copy");
            check(cudaMalloc(reinterpret_cast<void**>(&device_merged), sizeof(double)), "cudaMalloc merged");
            check(cudaMemcpyPeer(remote_copy, 0, second.device_sum, 1, sizeof(double)), "cudaMemcpyPeer");
            add_two_sums<<<1, 1>>>(first.device_sum, remote_copy, device_merged);
            check(cudaGetLastError(), "merge launch");
            check(cudaMemcpy(&merged, device_merged, sizeof(double), cudaMemcpyDeviceToHost), "copy merged sum");
        } catch (...) {
            if (remote_copy) cudaFree(remote_copy);
            if (device_merged) cudaFree(device_merged);
            cudaDeviceDisablePeerAccess(1);
            throw;
        }
        cudaFree(remote_copy);
        cudaFree(device_merged);
        check(cudaDeviceDisablePeerAccess(1), "cudaDeviceDisablePeerAccess");
        merge_path = "P2P device1->device0 partial sum then device0 merge";
    }
    std::cout << "merge_path=" << merge_path << " merged=" << merged << " CPU=" << cpu_sum << '\n';
    if (std::abs(merged - cpu_sum) > 1e-6) throw std::runtime_error("merged sum differs from CPU");
    std::cout << "chapter 35 two-GPU images: PASS\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--selftest") {
            partition_selftest();
            return 0;
        }
        if (argc != 1) throw std::runtime_error("usage: multi_gpu_images [--selftest]");
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count < devices_used) {
            std::cout << "SKIP two-GPU path: found " << count << " CUDA device(s), requires 2\n";
            return 77;
        }
        two_gpu_test();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "chapter 35 FAIL: " << e.what() << '\n';
        return 1;
    }
}
