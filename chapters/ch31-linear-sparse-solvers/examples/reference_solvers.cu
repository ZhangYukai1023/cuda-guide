#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    throw std::runtime_error("CUDA failure"); } } while (false)

template <class T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t count) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T))); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

__global__ void dense_gemv(const float* matrix, const float* x, float* y, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float sum = 0.f;
    for (int col = 0; col < n; ++col) sum += matrix[row * n + col] * x[col];
    y[row] = sum;
}

__global__ void csr_spmv(const int* offsets, const int* columns, const float* values,
                         const float* x, float* y, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float sum = 0.f;
    for (int p = offsets[row]; p < offsets[row + 1]; ++p)
        sum += values[p] * x[columns[p]];
    y[row] = sum;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

void run_dense() {
    const std::vector<float> a{1,2,3, 0,4,5, 6,0,7};
    const std::vector<float> x{1,2,3};
    const std::vector<float> expected{14,23,27};
    Buffer<float> da(9), dx(3), dy(3);
    CUDA_CHECK(cudaMemcpy(da.ptr, a.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx.ptr, x.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
    dense_gemv<<<1, 32>>>(da.ptr, dx.ptr, dy.ptr, 3);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> actual(3);
    CUDA_CHECK(cudaMemcpy(actual.data(), dy.ptr, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 3; ++i)
        if (actual[i] != expected[i]) throw std::runtime_error("dense GEMV mismatch");
    std::puts("dense row-major GEMV 3x3: PASS (14,23,27)");
}

void run_sparse_cg() {
    constexpr int n = 32;
    std::vector<int> offsets(n + 1), columns;
    std::vector<float> values;
    for (int row = 0; row < n; ++row) {
        offsets[row] = static_cast<int>(values.size());
        if (row > 0) { columns.push_back(row - 1); values.push_back(-1.f); }
        columns.push_back(row); values.push_back(4.f);
        if (row + 1 < n) { columns.push_back(row + 1); values.push_back(-1.f); }
    }
    offsets[n] = static_cast<int>(values.size());
    if (offsets[n] != 94) throw std::runtime_error("CSR nnz mismatch");
    std::vector<double> known(n), b(n), x(n, 0), r(n), p(n), ap(n);
    for (int i = 0; i < n; ++i) known[i] = 1. + .01 * i;
    for (int row = 0; row < n; ++row)
        for (int j = offsets[row]; j < offsets[row + 1]; ++j)
            b[row] += values[j] * known[columns[j]];
    Buffer<int> doff(offsets.size()), dcol(columns.size());
    Buffer<float> dval(values.size()), dx(n), dy(n);
    CUDA_CHECK(cudaMemcpy(doff.ptr, offsets.data(), offsets.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dcol.ptr, columns.data(), columns.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dval.ptr, values.data(), values.size()*sizeof(float), cudaMemcpyHostToDevice));
    auto multiply = [&](const std::vector<double>& input) {
        std::vector<float> host_x(n), host_y(n);
        for (int i = 0; i < n; ++i) host_x[i] = static_cast<float>(input[i]);
        CUDA_CHECK(cudaMemcpy(dx.ptr, host_x.data(), n*sizeof(float), cudaMemcpyHostToDevice));
        csr_spmv<<<1, 64>>>(doff.ptr, dcol.ptr, dval.ptr, dx.ptr, dy.ptr, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_y.data(), dy.ptr, n*sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<double> output(n);
        for (int i = 0; i < n; ++i) output[i] = host_y[i];
        return output;
    };
    const auto known_result = multiply(known);
    double spmv_error = 0;
    for (int i = 0; i < n; ++i) spmv_error = std::max(spmv_error, std::fabs(known_result[i] - b[i]));
    if (spmv_error > 1e-6) throw std::runtime_error("CSR SpMV mismatch");
    std::printf("CSR SpMV n=%d nnz=%d max_abs_error=%.9g PASS\n", n, offsets[n], spmv_error);
    r = p = b;
    const double bnorm = std::sqrt(dot(b,b));
    double rr = dot(r,r);
    int iterations = 0;
    for (; iterations < 4*n && std::sqrt(rr)/bnorm >= 1e-6; ++iterations) {
        ap = multiply(p);
        const double pap = dot(p,ap);
        if (!std::isfinite(pap) || pap <= 0.) throw std::runtime_error("CG non-SPD or numerical breakdown");
        const double alpha = rr/pap;
        for (int i = 0; i < n; ++i) { x[i] += alpha*p[i]; r[i] -= alpha*ap[i]; }
        const double rr_next = dot(r,r);
        if (!std::isfinite(rr_next)) throw std::runtime_error("CG nonfinite residual");
        const double beta = rr_next/rr;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta*p[i];
        rr = rr_next;
    }
    if (iterations == 4*n) throw std::runtime_error("CG did not converge");
    // Recompute the true residual on the CPU, independently of the recursive r.
    double true_residual_sq = 0, max_solution_error = 0;
    for (int row = 0; row < n; ++row) {
        double ax = 0;
        for (int j = offsets[row]; j < offsets[row+1]; ++j)
            ax += values[j] * x[columns[j]];
        true_residual_sq += (ax-b[row])*(ax-b[row]);
        max_solution_error = std::max(max_solution_error, std::fabs(x[row]-known[row]));
    }
    const double relative_residual = std::sqrt(true_residual_sq)/bnorm;
    if (relative_residual >= 1e-4 || max_solution_error >= 1e-4)
        throw std::runtime_error("CG solution or true residual mismatch");
    std::printf("hybrid CG n=%d iterations=%d true_relative_residual=%.9g max_solution_error=%.9g PASS\n",
                n, iterations, relative_residual, max_solution_error);
}

int main() {
    try {
        run_dense();
        run_sparse_cg();
        std::puts("chapter 31 reference solvers: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 31 reference solvers: FAIL: %s\n", e.what());
        return 1;
    }
}
