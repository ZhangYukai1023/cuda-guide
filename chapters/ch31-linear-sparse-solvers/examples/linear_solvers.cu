#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d %s: cuBLAS status %d\n", __FILE__, __LINE__, #call, int(s)); \
        throw std::runtime_error("cuBLAS failure"); \
    } \
} while (false)
#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t s = (call); \
    if (s != CUSPARSE_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d %s: cuSPARSE status %d\n", __FILE__, __LINE__, #call, int(s)); \
        throw std::runtime_error("cuSPARSE failure"); \
    } \
} while (false)
#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t s = (call); \
    if (s != CUSOLVER_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d %s: cuSOLVER status %d\n", __FILE__, __LINE__, #call, int(s)); \
        throw std::runtime_error("cuSOLVER failure"); \
    } \
} while (false)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), std::max<std::size_t>(n,1) * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

class Blas {
public:
    Blas() { CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~Blas() { if (handle_) cublasDestroy(handle_); }
    cublasHandle_t get() const { return handle_; }
private:
    cublasHandle_t handle_ = nullptr;
};

class Sparse {
public:
    Sparse() { CUSPARSE_CHECK(cusparseCreate(&handle_)); }
    ~Sparse() { if (handle_) cusparseDestroy(handle_); }
    cusparseHandle_t get() const { return handle_; }
private:
    cusparseHandle_t handle_ = nullptr;
};

class Solver {
public:
    Solver() { CUSOLVER_CHECK(cusolverDnCreate(&handle_)); }
    ~Solver() { if (handle_) cusolverDnDestroy(handle_); }
    cusolverDnHandle_t get() const { return handle_; }
private:
    cusolverDnHandle_t handle_ = nullptr;
};

struct Tridiagonal {
    int n;
    std::vector<float> dense, values, rhs, known;
    std::vector<int> offsets, columns;
};

Tridiagonal make_problem(int n) {
    Tridiagonal p{n, std::vector<float>(n*n,0.f), {}, {}, {}, std::vector<int>(n+1), {}};
    for (int i = 0; i < n; ++i) {
        p.known.push_back(1.f + i * 0.01f);
        p.offsets[i] = static_cast<int>(p.values.size());
        for (int j = std::max(0,i-1); j <= std::min(n-1,i+1); ++j) {
            const float value = j == i ? 4.f : -1.f;
            p.dense[i*n+j] = value;
            p.columns.push_back(j);
            p.values.push_back(value);
        }
    }
    p.offsets[n] = static_cast<int>(p.values.size());
    p.rhs.resize(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) p.rhs[i] += p.dense[i*n+j] * p.known[j];
    return p;
}

double relative_residual(const Tridiagonal& p, const std::vector<float>& x) {
    double norm_b = 0, norm_r = 0;
    for (int i = 0; i < p.n; ++i) {
        double ax = 0;
        for (int j = 0; j < p.n; ++j) ax += static_cast<double>(p.dense[i*p.n+j]) * x[j];
        const double error = ax - p.rhs[i];
        norm_r += error * error;
        norm_b += static_cast<double>(p.rhs[i]) * p.rhs[i];
    }
    return std::sqrt(norm_r / norm_b);
}

double max_solution_error(const Tridiagonal& p, const std::vector<float>& x) {
    double maximum = 0;
    for (int i = 0; i < p.n; ++i)
        maximum = std::max(maximum, std::fabs(static_cast<double>(x[i]) - p.known[i]));
    return maximum;
}

void dense_gemv(Blas& blas) {
    const std::vector<float> a = {1,2,3, 0,4,5, 6,0,7}; // row-major and asymmetric
    const std::vector<float> x = {1,2,3};
    DeviceBuffer<float> da(9), dx(3), dy(3);
    CUDA_CHECK(cudaMemcpy(da.get(), a.data(), 9*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx.get(), x.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
    const float alpha = 1.f, beta = 0.f;
    // Row-major A is column-major A^T; op(T) recovers A*x.
    CUBLAS_CHECK(cublasSgemv(blas.get(), CUBLAS_OP_T, 3, 3, &alpha,
                            da.get(), 3, dx.get(), 1, &beta, dy.get(), 1));
    std::vector<float> actual(3);
    CUDA_CHECK(cudaMemcpy(actual.data(), dy.get(), 3*sizeof(float), cudaMemcpyDeviceToHost));
    const std::vector<float> expected = {14,23,27};
    if (actual != expected) throw std::runtime_error("dense GEMV layout mismatch");
    std::puts("dense asymmetric GEMV 3x3 exact CPU reference: PASS");
}

void run_sparse_and_solvers(const Tridiagonal& p, Blas& blas, Sparse& sparse, Solver& solver) {
    const int n = p.n, nnz = static_cast<int>(p.values.size());
    DeviceBuffer<int> d_offsets(p.offsets.size()), d_columns(p.columns.size());
    DeviceBuffer<float> d_values(p.values.size()), d_known(n), d_b(n), d_x(n), d_r(n), d_direction(n), d_ap(n);
    CUDA_CHECK(cudaMemcpy(d_offsets.get(), p.offsets.data(), p.offsets.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns.get(), p.columns.data(), p.columns.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values.get(), p.values.data(), p.values.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_known.get(), p.known.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), p.rhs.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr, vec_y = nullptr;
    CUSPARSE_CHECK(cusparseCreateCsr(&matrix, n, n, nnz, d_offsets.get(), d_columns.get(),
                                    d_values.get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, n, d_direction.get(), CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, n, d_ap.get(), CUDA_R_32F));
    const float one = 1.f, zero = 0.f;
    std::size_t workspace_bytes = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &one, matrix, vec_x, &zero, vec_y, CUDA_R_32F,
                                          CUSPARSE_SPMV_ALG_DEFAULT, &workspace_bytes));
    DeviceBuffer<char> workspace(workspace_bytes);
    auto spmv = [&] {
        CUSPARSE_CHECK(cusparseSpMV(sparse.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &one, matrix, vec_x, &zero, vec_y, CUDA_R_32F,
                                   CUSPARSE_SPMV_ALG_DEFAULT, workspace.get()));
    };
    CUDA_CHECK(cudaMemcpy(d_direction.get(), d_known.get(), n*sizeof(float), cudaMemcpyDeviceToDevice));
    spmv();
    std::vector<float> actual(n);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_ap.get(), n*sizeof(float), cudaMemcpyDeviceToHost));
    double max_spmv_error = 0;
    for (int i = 0; i < n; ++i)
        max_spmv_error = std::max(max_spmv_error, std::fabs(static_cast<double>(actual[i]) - p.rhs[i]));
    std::printf("CSR SpMV n=%d nnz=%d max_abs_error=%.8g %s\n",
                n, nnz, max_spmv_error, max_spmv_error < 1e-5 ? "PASS" : "FAIL");
    if (max_spmv_error >= 1e-5) throw std::runtime_error("CSR SpMV mismatch");

    DeviceBuffer<float> d_lu(p.dense.size()), d_solve_rhs(n);
    DeviceBuffer<int> d_pivot(n), d_info(1);
    CUDA_CHECK(cudaMemcpy(d_lu.get(), p.dense.data(), p.dense.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_solve_rhs.get(), p.rhs.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(solver.get(), n, n, d_lu.get(), n, &lwork));
    DeviceBuffer<float> lu_workspace(lwork);
    CUSOLVER_CHECK(cusolverDnSgetrf(solver.get(), n, n, d_lu.get(), n,
                                   lu_workspace.get(), d_pivot.get(), d_info.get()));
    int info = -999;
    CUDA_CHECK(cudaMemcpy(&info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) throw std::runtime_error("LU factorization failed: info=" + std::to_string(info));
    CUSOLVER_CHECK(cusolverDnSgetrs(solver.get(), CUBLAS_OP_N, n, 1, d_lu.get(), n,
                                   d_pivot.get(), d_solve_rhs.get(), n, d_info.get()));
    CUDA_CHECK(cudaMemcpy(&info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) throw std::runtime_error("LU solve failed: info=" + std::to_string(info));
    CUDA_CHECK(cudaMemcpy(actual.data(), d_solve_rhs.get(), n*sizeof(float), cudaMemcpyDeviceToHost));
    const double lu_residual = relative_residual(p, actual), lu_error = max_solution_error(p, actual);
    std::printf("cuSOLVER LU n=%d relative_residual=%.8g max_solution_error=%.8g %s\n",
                n, lu_residual, lu_error, lu_residual < 1e-5 && lu_error < 1e-4 ? "PASS" : "FAIL");
    if (lu_residual >= 1e-5 || lu_error >= 1e-4) throw std::runtime_error("LU reference mismatch");

    // CG is valid here because A is symmetric positive definite (4 on diagonal, -1 neighbors).
    CUDA_CHECK(cudaMemset(d_x.get(), 0, n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_r.get(), d_b.get(), n*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_direction.get(), d_r.get(), n*sizeof(float), cudaMemcpyDeviceToDevice));
    float rhs_sq = 0.f, old_sq = 0.f;
    CUBLAS_CHECK(cublasSdot(blas.get(), n, d_b.get(), 1, d_b.get(), 1, &rhs_sq));
    CUBLAS_CHECK(cublasSdot(blas.get(), n, d_r.get(), 1, d_r.get(), 1, &old_sq));
    int iterations = 0;
    const int max_iterations = n * 4;
    bool converged = false;
    for (; iterations < max_iterations; ++iterations) {
        spmv(); // d_ap = A * d_direction
        float denom = 0.f;
        CUBLAS_CHECK(cublasSdot(blas.get(), n, d_direction.get(), 1, d_ap.get(), 1, &denom));
        if (!(denom > 0.f) || !std::isfinite(denom)) throw std::runtime_error("CG invalid p^T A p");
        const float alpha = old_sq / denom, minus_alpha = -alpha;
        CUBLAS_CHECK(cublasSaxpy(blas.get(), n, &alpha, d_direction.get(), 1, d_x.get(), 1));
        CUBLAS_CHECK(cublasSaxpy(blas.get(), n, &minus_alpha, d_ap.get(), 1, d_r.get(), 1));
        float new_sq = 0.f;
        CUBLAS_CHECK(cublasSdot(blas.get(), n, d_r.get(), 1, d_r.get(), 1, &new_sq));
        if (!std::isfinite(new_sq)) throw std::runtime_error("CG residual is not finite");
        const float recursive_relative_residual = std::sqrt(new_sq / rhs_sq);
        std::printf("CG iteration=%d recursive_relative_residual=%.8g\n",
                    iterations + 1, static_cast<double>(recursive_relative_residual));
        if (recursive_relative_residual < 1e-6f) { ++iterations; converged = true; break; }
        const float beta = new_sq / old_sq;
        CUBLAS_CHECK(cublasSscal(blas.get(), n, &beta, d_direction.get(), 1));
        CUBLAS_CHECK(cublasSaxpy(blas.get(), n, &one, d_r.get(), 1, d_direction.get(), 1));
        old_sq = new_sq;
    }
    if (!converged) throw std::runtime_error("CG did not converge");
    CUDA_CHECK(cudaMemcpy(actual.data(), d_x.get(), n*sizeof(float), cudaMemcpyDeviceToHost));
    const double cg_residual = relative_residual(p, actual), cg_error = max_solution_error(p, actual);
    std::printf("CG n=%d iterations=%d relative_residual=%.8g max_solution_error=%.8g %s\n",
                n, iterations, cg_residual, cg_error,
                cg_residual < 1e-5 && cg_error < 1e-4 ? "PASS" : "FAIL");
    if (cg_residual >= 1e-5 || cg_error >= 1e-4) throw std::runtime_error("CG reference mismatch");
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
    CUSPARSE_CHECK(cusparseDestroySpMat(matrix));
}

int main() {
    try {
        Blas blas;
        Sparse sparse;
        Solver solver;
        dense_gemv(blas);
        run_sparse_and_solvers(make_problem(32), blas, sparse, solver);
        std::puts("chapter 31 linear and sparse solvers: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chapter 31 linear and sparse solvers: FAIL: %s\n", e.what());
        return 1;
    }
}
