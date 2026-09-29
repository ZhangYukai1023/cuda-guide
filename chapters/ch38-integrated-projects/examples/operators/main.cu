#include "operators.cuh"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float epsilon = 1e-5f;

void check(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
}

class Buffer {
public:
    explicit Buffer(size_t n) : count_(n) { check(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(float)), "cudaMalloc"); }
    ~Buffer() { if (p_) cudaFree(p_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    float* data() const { return p_; }
    void upload(const std::vector<float>& host) {
        if (host.size() != count_) throw std::runtime_error("upload size mismatch");
        check(cudaMemcpy(p_, host.data(), count_ * sizeof(float), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<float> download() const {
        std::vector<float> host(count_);
        check(cudaMemcpy(host.data(), p_, count_ * sizeof(float), cudaMemcpyDeviceToHost), "download");
        return host;
    }
private:
    size_t count_;
    float* p_ = nullptr;
};

struct Shape { int rows = 7, cols = 11, out_cols = 5; };

Shape parse(int argc, char** argv) {
    Shape shape;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("usage: project_b_operators [--rows N --cols K --out-cols M]");
        int value = std::stoi(argv[i + 1]);
        std::string key = argv[i];
        if (key == "--rows") shape.rows = value;
        else if (key == "--cols") shape.cols = value;
        else if (key == "--out-cols") shape.out_cols = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (shape.rows < 1 || shape.rows > 128 || shape.cols < 1 || shape.cols > 256 ||
        shape.out_cols < 1 || shape.out_cols > 128)
        throw std::runtime_error("shape bounds: rows 1..128, cols 1..256, out-cols 1..128");
    return shape;
}

std::vector<float> make_input(Shape shape) {
    std::vector<float> input(shape.rows * shape.cols);
    for (int row = 0; row < shape.rows; ++row) for (int col = 0; col < shape.cols; ++col) {
        float base = static_cast<float>(((row * 7 + col * 3) % 17) - 8) * 0.37f;
        input[row * shape.cols + col] = base + (row == 0 ? 1000.0f : -2.0f * row);
    }
    return input;
}

std::vector<float> make_weights(Shape shape) {
    std::vector<float> weights(shape.cols * shape.out_cols);
    for (int k = 0; k < shape.cols; ++k) for (int col = 0; col < shape.out_cols; ++col)
        weights[k * shape.out_cols + col] = static_cast<float>(((k * 5 + col * 11) % 19) - 9) * 0.023f;
    return weights;
}

struct Reference {
    std::vector<double> softmax, layernorm, output;
};

Reference cpu_reference(const std::vector<float>& input, const std::vector<float>& weights, Shape shape) {
    Reference ref;
    ref.softmax.resize(shape.rows * shape.cols);
    ref.layernorm.resize(shape.rows * shape.cols);
    ref.output.resize(shape.rows * shape.out_cols);
    for (int row = 0; row < shape.rows; ++row) {
        double maximum = -INFINITY;
        for (int col = 0; col < shape.cols; ++col)
            maximum = std::max(maximum, static_cast<double>(input[row * shape.cols + col]));
        double denominator = 0.0;
        for (int col = 0; col < shape.cols; ++col) {
            int i = row * shape.cols + col;
            ref.softmax[i] = std::exp(static_cast<double>(input[i]) - maximum);
            denominator += ref.softmax[i];
        }
        double mean = 0.0;
        for (int col = 0; col < shape.cols; ++col) {
            int i = row * shape.cols + col;
            ref.softmax[i] /= denominator;
            mean += ref.softmax[i];
        }
        mean /= shape.cols;
        double variance = 0.0;
        for (int col = 0; col < shape.cols; ++col) {
            double centered = ref.softmax[row * shape.cols + col] - mean;
            variance += centered * centered;
        }
        variance /= shape.cols;
        for (int col = 0; col < shape.cols; ++col) {
            int i = row * shape.cols + col;
            ref.layernorm[i] = (ref.softmax[i] - mean) / std::sqrt(variance + epsilon);
        }
    }
    for (int row = 0; row < shape.rows; ++row) for (int col = 0; col < shape.out_cols; ++col) {
        double sum = 0.0;
        for (int k = 0; k < shape.cols; ++k)
            sum += ref.layernorm[row * shape.cols + k] * weights[k * shape.out_cols + col];
        ref.output[row * shape.out_cols + col] = sum;
    }
    return ref;
}

void compare(const char* name, const std::vector<float>& actual, const std::vector<double>& expected, double tolerance) {
    if (actual.size() != expected.size()) throw std::runtime_error(std::string(name) + " shape mismatch");
    double max_error = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error(std::string(name) + " nonfinite result");
        max_error = std::max(max_error, std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    std::cout << name << " max_abs_error=" << max_error << " tolerance=" << tolerance << '\n';
    if (max_error > tolerance) throw std::runtime_error(std::string(name) + " CPU comparison failed");
}

void run(Shape shape) {
    auto input = make_input(shape), weights = make_weights(shape);
    auto reference = cpu_reference(input, weights, shape);
    Buffer d_input(input.size()), d_weights(weights.size());
    Buffer d_softmax(input.size()), d_layernorm(input.size()), d_output(shape.rows * shape.out_cols);
    d_input.upload(input);
    d_weights.upload(weights);
    auto launch_chain = [&] {
        launch_softmax(d_input.data(), d_softmax.data(), shape.rows, shape.cols, 0);
        launch_layernorm(d_softmax.data(), d_layernorm.data(), shape.rows, shape.cols, epsilon, 0);
        launch_gemm(d_layernorm.data(), d_weights.data(), d_output.data(), shape.rows, shape.cols, shape.out_cols, 0);
    };
    launch_chain();
    check(cudaGetLastError(), "operator chain launch");
    auto softmax = d_softmax.download(), layernorm = d_layernorm.download(), output = d_output.download();
    compare("softmax", softmax, reference.softmax, 2e-5);
    compare("layernorm", layernorm, reference.layernorm, 3e-5);
    compare("GEMM", output, reference.output, 4e-5);
    double row_sum_error = 0.0;
    for (int row = 0; row < shape.rows; ++row) {
        double sum = 0.0;
        for (int col = 0; col < shape.cols; ++col) sum += softmax[row * shape.cols + col];
        row_sum_error = std::max(row_sum_error, std::abs(sum - 1.0));
    }
    std::cout << "softmax max_row_sum_error=" << row_sum_error << '\n';
    if (row_sum_error > 2e-6) throw std::runtime_error("softmax rows do not sum to one");
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "event start");
    check(cudaEventCreate(&stop), "event stop");
    for (int i = 0; i < 10; ++i) launch_chain();
    check(cudaDeviceSynchronize(), "warmup");
    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < 100; ++i) launch_chain();
    check(cudaGetLastError(), "repeated chain launch");
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "wait timed chain");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
    cudaEventDestroy(start); cudaEventDestroy(stop);
    std::cout << "shape rows=" << shape.rows << " cols=" << shape.cols << " out_cols=" << shape.out_cols
              << " chain_average_kernel_ms=" << ms / 100.0f << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        Shape shape = parse(argc, argv);
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        std::cout << "GPU=" << prop.name << '\n';
        run(shape);
        std::cout << "chapter 38 project B operators: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "chapter 38 project B FAIL: " << e.what() << '\n';
        return 1;
    }
}
