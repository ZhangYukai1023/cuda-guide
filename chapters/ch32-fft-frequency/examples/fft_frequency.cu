#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

void cuda_ok(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
}

void fft_ok(cufftResult code, const char* what) {
    if (code != CUFFT_SUCCESS) throw std::runtime_error(std::string(what) + ": cuFFT status " + std::to_string(code));
}

template <class T> class Buffer {
public:
    explicit Buffer(size_t n) : n_(n) { cuda_ok(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)), "cudaMalloc"); }
    ~Buffer() { if (p_) cudaFree(p_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() const { return p_; }
    void upload(const std::vector<T>& v) {
        if (v.size() != n_) throw std::runtime_error("upload size mismatch");
        cuda_ok(cudaMemcpy(p_, v.data(), n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> v(n_);
        cuda_ok(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
private:
    size_t n_;
    T* p_ = nullptr;
};

class Plan {
public:
    Plan(std::initializer_list<int> shape, cufftType type, int batch = 1) {
        fft_ok(cufftCreate(&handle_), "cufftCreate");
        try {
            fft_ok(cufftSetAutoAllocation(handle_, 0), "cufftSetAutoAllocation");
            if (shape.size() == 1) {
                fft_ok(cufftMakePlan1d(handle_, *shape.begin(), type, batch, &bytes_), "cufftMakePlan1d");
            } else if (shape.size() == 2 && batch == 1) {
                const int* d = shape.begin();
                fft_ok(cufftMakePlan2d(handle_, d[0], d[1], type, &bytes_), "cufftMakePlan2d");
            } else {
                throw std::runtime_error("unsupported plan shape");
            }
            if (bytes_) {
                cuda_ok(cudaMalloc(&work_, bytes_), "plan workspace cudaMalloc");
                fft_ok(cufftSetWorkArea(handle_, work_), "cufftSetWorkArea");
            }
        } catch (...) {
            if (work_) cudaFree(work_);
            cufftDestroy(handle_);
            throw;
        }
    }
    ~Plan() { cufftDestroy(handle_); if (work_) cudaFree(work_); }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    cufftHandle get() const { return handle_; }
    size_t workspace_bytes() const { return bytes_; }
private:
    cufftHandle handle_{};
    void* work_ = nullptr;
    size_t bytes_ = 0;
};

__global__ void scale_complex(cufftComplex* x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i].x *= scale; x[i].y *= scale; }
}

__global__ void scale_real(float* x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

__global__ void lowpass_1d(cufftComplex* x, int n, int max_frequency) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int signed_frequency = i <= n / 2 ? i : i - n;
        if (abs(signed_frequency) > max_frequency) x[i] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

__global__ void lowpass_2d(cufftComplex* x, int width, int height, int max_x, int max_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        int col = i % width, row = i / width;
        int fx = col <= width / 2 ? col : col - width;
        int fy = row <= height / 2 ? row : row - height;
        if (abs(fx) > max_x || abs(fy) > max_y) x[i] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

__global__ void multiply_spectra(cufftComplex* a, const cufftComplex* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float re = a[i].x * b[i].x - a[i].y * b[i].y;
        float im = a[i].x * b[i].y + a[i].y * b[i].x;
        a[i] = make_cuFloatComplex(re, im);
    }
}

void launch_ok(const char* what) { cuda_ok(cudaGetLastError(), what); }

std::complex<double> dft_bin(const std::vector<float>& x, int k) {
    std::complex<double> sum(0.0, 0.0);
    const int n = static_cast<int>(x.size());
    for (int t = 0; t < n; ++t) {
        double angle = -2.0 * kPi * k * t / n;
        sum += static_cast<double>(x[t]) * std::complex<double>(std::cos(angle), std::sin(angle));
    }
    return sum;
}

double check_real(const std::string& label, const std::vector<float>& actual,
                  const std::vector<float>& expected, double tolerance) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + " size mismatch");
    double error = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error(label + " nonfinite at " + std::to_string(i));
        error = std::max(error, std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    std::cout << label << " max_abs_error=" << error << " tolerance=" << tolerance << '\n';
    if (error > tolerance) throw std::runtime_error(label + " failed");
    return error;
}

void check_spectrum(const std::string& label, const std::vector<cufftComplex>& actual,
                    const std::vector<float>& signal, double tolerance) {
    double error = 0.0;
    for (size_t k = 0; k < actual.size(); ++k) {
        std::complex<double> ref = dft_bin(signal, static_cast<int>(k));
        if (!std::isfinite(actual[k].x) || !std::isfinite(actual[k].y)) throw std::runtime_error(label + " nonfinite spectrum");
        error = std::max(error, std::abs(std::complex<double>(actual[k].x, actual[k].y) - ref));
    }
    std::cout << label << " max_spectrum_error=" << error << " tolerance=" << tolerance << '\n';
    if (error > tolerance) throw std::runtime_error(label + " spectrum failed");
}

void batched_roundtrip() {
    constexpr int n = 8, batch = 2;
    std::vector<float> first(n), second(n);
    std::vector<cufftComplex> input(n * batch);
    for (int i = 0; i < n; ++i) {
        first[i] = i == 0 ? 1.0f : 0.0f;
        second[i] = static_cast<float>(0.5 + std::sin(2.0 * kPi * i / n));
        input[i] = make_cuFloatComplex(first[i], 0.0f);
        input[n + i] = make_cuFloatComplex(second[i], 0.0f);
    }
    Buffer<cufftComplex> d(n * batch);
    d.upload(input);
    Plan p({n}, CUFFT_C2C, batch);
    fft_ok(cufftExecC2C(p.get(), d.data(), d.data(), CUFFT_FORWARD), "batched forward");
    auto spectrum = d.download();
    check_spectrum("batch 0 C2C vs CPU DFT", std::vector<cufftComplex>(spectrum.begin(), spectrum.begin() + n), first, 2e-5);
    check_spectrum("batch 1 C2C vs CPU DFT", std::vector<cufftComplex>(spectrum.begin() + n, spectrum.end()), second, 2e-5);
    fft_ok(cufftExecC2C(p.get(), d.data(), d.data(), CUFFT_INVERSE), "batched inverse");
    scale_complex<<<1, 64>>>(d.data(), n * batch, 1.0f / n);
    launch_ok("batched normalization");
    auto restored = d.download();
    std::vector<float> restored_real(n * batch), expected(n * batch);
    for (int i = 0; i < n * batch; ++i) {
        restored_real[i] = restored[i].x;
        if (std::abs(restored[i].y) > 1e-5f) throw std::runtime_error("batched inverse imaginary residue");
        expected[i] = input[i].x;
    }
    check_real("batched C2C roundtrip", restored_real, expected, 2e-5);
    std::cout << "batched plan workspace_bytes=" << p.workspace_bytes() << "\n";
}

void real_roundtrip() {
    constexpr int n = 8, bins = n / 2 + 1;
    std::vector<float> signal(n);
    for (int i = 0; i < n; ++i) signal[i] = static_cast<float>(1.0 + std::cos(2.0 * kPi * i / n));
    Buffer<float> real(n);
    Buffer<cufftComplex> spectrum(bins);
    real.upload(signal);
    Plan forward({n}, CUFFT_R2C), inverse({n}, CUFFT_C2R);
    fft_ok(cufftExecR2C(forward.get(), real.data(), spectrum.data()), "R2C forward");
    check_spectrum("R2C half spectrum vs CPU DFT", spectrum.download(), signal, 2e-5);
    fft_ok(cufftExecC2R(inverse.get(), spectrum.data(), real.data()), "C2R inverse");
    scale_real<<<1, 32>>>(real.data(), n, 1.0f / n);
    launch_ok("real normalization");
    check_real("R2C/C2R roundtrip", real.download(), signal, 2e-5);
    std::cout << "R2C workspace_bytes=" << forward.workspace_bytes()
              << " C2R workspace_bytes=" << inverse.workspace_bytes() << '\n';
}

void one_dimensional_filter() {
    constexpr int n = 64;
    std::vector<cufftComplex> input(n);
    std::vector<float> clean(n);
    for (int i = 0; i < n; ++i) {
        clean[i] = static_cast<float>(std::sin(2.0 * kPi * 3 * i / n));
        input[i] = make_cuFloatComplex(clean[i] + static_cast<float>(0.5 * std::sin(2.0 * kPi * 12 * i / n)), 0.0f);
    }
    Buffer<cufftComplex> d(n);
    d.upload(input);
    Plan p({n}, CUFFT_C2C);
    fft_ok(cufftExecC2C(p.get(), d.data(), d.data(), CUFFT_FORWARD), "1D filter forward");
    lowpass_1d<<<1, 64>>>(d.data(), n, 8);
    launch_ok("1D lowpass");
    fft_ok(cufftExecC2C(p.get(), d.data(), d.data(), CUFFT_INVERSE), "1D filter inverse");
    scale_complex<<<1, 64>>>(d.data(), n, 1.0f / n);
    launch_ok("1D filter normalization");
    auto out = d.download();
    std::vector<float> got(n);
    for (int i = 0; i < n; ++i) {
        if (std::abs(out[i].y) > 1e-4f) throw std::runtime_error("1D filter imaginary residue");
        got[i] = out[i].x;
    }
    check_real("1D remove k=12 retain k=3", got, clean, 1e-4);
}

void two_dimensional_filter() {
    constexpr int width = 16, height = 16, count = width * height;
    std::vector<cufftComplex> input(count);
    std::vector<float> clean(count);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        int i = y * width + x;
        clean[i] = static_cast<float>(0.25 + std::cos(2.0 * kPi * (x + 2 * y) / width));
        input[i] = make_cuFloatComplex(clean[i] + static_cast<float>(0.4 * std::sin(2.0 * kPi * (5 * x + 4 * y) / width)), 0.0f);
    }
    Buffer<cufftComplex> d(count);
    d.upload(input);
    Plan p({height, width}, CUFFT_C2C);
    fft_ok(cufftExecC2C(p.get(), d.data(), d.data(), CUFFT_FORWARD), "2D filter forward");
    lowpass_2d<<<1, 256>>>(d.data(), width, height, 2, 2);
    launch_ok("2D lowpass");
    fft_ok(cufftExecC2C(p.get(), d.data(), d.data(), CUFFT_INVERSE), "2D filter inverse");
    scale_complex<<<1, 256>>>(d.data(), count, 1.0f / count);
    launch_ok("2D normalization");
    auto out = d.download();
    std::vector<float> got(count);
    for (int i = 0; i < count; ++i) {
        if (std::abs(out[i].y) > 1e-4f) throw std::runtime_error("2D filter imaginary residue");
        got[i] = out[i].x;
    }
    check_real("2D lowpass vs analytical clean image", got, clean, 2e-4);
    std::cout << "2D plan workspace_bytes=" << p.workspace_bytes() << '\n';
}

void linear_convolution() {
    constexpr int signal_size = 8, kernel_size = 3, padded = 16, output_size = signal_size + kernel_size - 1;
    const std::vector<float> signal{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> kernel{1, -2, 1};
    std::vector<cufftComplex> a(padded, make_cuFloatComplex(0.0f, 0.0f));
    std::vector<cufftComplex> b = a;
    for (int i = 0; i < signal_size; ++i) a[i].x = signal[i];
    for (int i = 0; i < kernel_size; ++i) b[i].x = kernel[i];
    Buffer<cufftComplex> da(padded), db(padded);
    da.upload(a); db.upload(b);
    Plan p({padded}, CUFFT_C2C);
    fft_ok(cufftExecC2C(p.get(), da.data(), da.data(), CUFFT_FORWARD), "convolution signal forward");
    fft_ok(cufftExecC2C(p.get(), db.data(), db.data(), CUFFT_FORWARD), "convolution kernel forward");
    multiply_spectra<<<1, 32>>>(da.data(), db.data(), padded);
    launch_ok("spectrum product");
    fft_ok(cufftExecC2C(p.get(), da.data(), da.data(), CUFFT_INVERSE), "convolution inverse");
    scale_complex<<<1, 32>>>(da.data(), padded, 1.0f / padded);
    launch_ok("convolution normalization");
    auto out = da.download();
    std::vector<float> expected(output_size, 0.0f), got(output_size);
    for (int i = 0; i < signal_size; ++i) for (int j = 0; j < kernel_size; ++j) expected[i + j] += signal[i] * kernel[j];
    for (int i = 0; i < output_size; ++i) {
        if (std::abs(out[i].y) > 1e-4f) throw std::runtime_error("convolution imaginary residue");
        got[i] = out[i].x;
    }
    check_real("zero-padded FFT vs CPU full convolution", got, expected, 2e-4);
    for (int i = output_size; i < padded; ++i) {
        if (std::abs(out[i].x) > 2e-4f || std::abs(out[i].y) > 2e-4f) throw std::runtime_error("convolution padded tail is nonzero");
    }
    std::cout << "FFT length=" << padded << " full convolution length=" << output_size << '\n';
}

} // namespace

int main() {
    try {
        int device = 0;
        cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop{};
        cuda_ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        std::cout << "GPU=" << prop.name << "\n";
        batched_roundtrip();
        real_roundtrip();
        one_dimensional_filter();
        two_dimensional_filter();
        linear_convolution();
        std::cout << "chapter 32 FFT and frequency: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "chapter 32 FAIL: " << e.what() << '\n';
        return 1;
    }
}
