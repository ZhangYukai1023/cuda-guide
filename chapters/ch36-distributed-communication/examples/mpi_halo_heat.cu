#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int width = 32, height = 24, steps = 40;
constexpr float r = 0.2f;

void gpu_ok(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
}

void mpi_ok(int code, const char* what) {
    if (code != MPI_SUCCESS) throw std::runtime_error(std::string(what) + ": MPI error " + std::to_string(code));
}

int rows_for(int rank, int ranks) { return height / ranks + (rank < height % ranks ? 1 : 0); }
int first_row(int rank, int ranks) { return rank * (height / ranks) + (rank < height % ranks ? rank : height % ranks); }

float initial(int x, int y) {
    return x >= width / 3 && x < 2 * width / 3 && y >= height / 3 && y < 2 * height / 3 ? 1.0f : 0.0f;
}

std::vector<double> cpu_reference() {
    std::vector<double> old(width * height), next(width * height, 0.0);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) old[y * width + x] = initial(x, y);
    for (int step = 0; step < steps; ++step) {
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            next[i] = x == 0 || x == width - 1 || y == 0 || y == height - 1 ? 0.0
                : old[i] + static_cast<double>(r) *
                    (old[i - 1] + old[i + 1] + old[i - width] + old[i + width] - 4.0 * old[i]);
        }
        old.swap(next);
    }
    return old;
}

__global__ void heat_step(const float* old, float* next, int owned_rows, int first_global_row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int local_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= width || local_y > owned_rows) return;
    int global_y = first_global_row + local_y - 1;
    int i = local_y * width + x;
    if (x == 0 || x == width - 1 || global_y == 0 || global_y == height - 1) {
        next[i] = 0.0f;
        return;
    }
    next[i] = old[i] + r * (old[i - 1] + old[i + 1] + old[i - width] + old[i + width] - 4.0f * old[i]);
}

class DeviceField {
public:
    explicit DeviceField(int rows) : count_((rows + 2) * width) {
        gpu_ok(cudaMalloc(reinterpret_cast<void**>(&first_), count_ * sizeof(float)), "cudaMalloc first");
        try { gpu_ok(cudaMalloc(reinterpret_cast<void**>(&second_), count_ * sizeof(float)), "cudaMalloc second"); }
        catch (...) { cudaFree(first_); throw; }
        gpu_ok(cudaMemset(first_, 0, count_ * sizeof(float)), "zero first");
        gpu_ok(cudaMemset(second_, 0, count_ * sizeof(float)), "zero second");
    }
    ~DeviceField() { cudaFree(first_); cudaFree(second_); }
    DeviceField(const DeviceField&) = delete;
    DeviceField& operator=(const DeviceField&) = delete;
    float* current() const { return first_; }
    float* next() const { return second_; }
    void swap() { std::swap(first_, second_); }
private:
    size_t count_;
    float *first_ = nullptr, *second_ = nullptr;
};

void exchange_halos(float* current, int rows, int rank, int ranks, bool cuda_aware) {
    int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int down = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    if (cuda_aware) {
        mpi_ok(MPI_Sendrecv(current + width, width, MPI_FLOAT, up, 10,
                            current, width, MPI_FLOAT, up, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "device halo up");
        mpi_ok(MPI_Sendrecv(current + rows * width, width, MPI_FLOAT, down, 20,
                            current + (rows + 1) * width, width, MPI_FLOAT, down, 10,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE), "device halo down");
    } else {
        std::vector<float> send_up(width), send_down(width), recv_up(width, 0.0f), recv_down(width, 0.0f);
        gpu_ok(cudaMemcpy(send_up.data(), current + width, width * sizeof(float), cudaMemcpyDeviceToHost), "stage upper row");
        gpu_ok(cudaMemcpy(send_down.data(), current + rows * width, width * sizeof(float), cudaMemcpyDeviceToHost), "stage lower row");
        mpi_ok(MPI_Sendrecv(send_up.data(), width, MPI_FLOAT, up, 10,
                            recv_up.data(), width, MPI_FLOAT, up, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "host halo up");
        mpi_ok(MPI_Sendrecv(send_down.data(), width, MPI_FLOAT, down, 20,
                            recv_down.data(), width, MPI_FLOAT, down, 10,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE), "host halo down");
        gpu_ok(cudaMemcpy(current, recv_up.data(), width * sizeof(float), cudaMemcpyHostToDevice), "upload upper halo");
        gpu_ok(cudaMemcpy(current + (rows + 1) * width, recv_down.data(), width * sizeof(float), cudaMemcpyHostToDevice), "upload lower halo");
    }
}

int run(int rank, int ranks, bool cuda_aware, bool require_two_nodes) {
    if (ranks < 2 || ranks > height) {
        if (rank == 0) std::cout << "SKIP: require 2.." << height << " MPI ranks\n";
        return 77;
    }
    char host[MPI_MAX_PROCESSOR_NAME]{};
    int name_length = 0;
    mpi_ok(MPI_Get_processor_name(host, &name_length), "MPI_Get_processor_name");
    std::vector<char> names(ranks * MPI_MAX_PROCESSOR_NAME, 0);
    mpi_ok(MPI_Allgather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names.data(), MPI_MAX_PROCESSOR_NAME,
                         MPI_CHAR, MPI_COMM_WORLD), "MPI_Allgather hostnames");
    std::set<std::string> nodes;
    for (int i = 0; i < ranks; ++i) nodes.emplace(names.data() + i * MPI_MAX_PROCESSOR_NAME);
    if (require_two_nodes && nodes.size() < 2) {
        if (rank == 0) std::cout << "SKIP: --require-two-nodes saw only one node\n";
        return 77;
    }
    MPI_Comm local_comm = MPI_COMM_NULL;
    mpi_ok(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm), "split local ranks");
    int local_rank = 0;
    mpi_ok(MPI_Comm_rank(local_comm, &local_rank), "local rank");
    mpi_ok(MPI_Comm_free(&local_comm), "free local communicator");
    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice) device_count = 0;
    else gpu_ok(device_status, "cudaGetDeviceCount");
    int can_run = local_rank < device_count ? 1 : 0, all_can_run = 0;
    mpi_ok(MPI_Allreduce(&can_run, &all_can_run, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "GPU availability");
    if (!all_can_run) {
        if (rank == 0) std::cout << "SKIP: at least one rank lacks its local GPU\n";
        return 77;
    }
    gpu_ok(cudaSetDevice(local_rank), "cudaSetDevice local rank");
    int rows = rows_for(rank, ranks), first = first_row(rank, ranks);
    DeviceField field(rows);
    std::vector<float> local(rows * width);
    for (int y = 0; y < rows; ++y) for (int x = 0; x < width; ++x)
        local[y * width + x] = initial(x, first + y);
    gpu_ok(cudaMemcpy(field.current() + width, local.data(), local.size() * sizeof(float),
                      cudaMemcpyHostToDevice), "initial local upload");
    mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "start barrier");
    double started = MPI_Wtime();
    dim3 block(16, 8), grid((width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    for (int step = 0; step < steps; ++step) {
        gpu_ok(cudaDeviceSynchronize(), "wait previous step");
        exchange_halos(field.current(), rows, rank, ranks, cuda_aware);
        heat_step<<<grid, block>>>(field.current(), field.next(), rows, first);
        gpu_ok(cudaGetLastError(), "heat_step launch");
        field.swap();
    }
    gpu_ok(cudaDeviceSynchronize(), "wait final step");
    double local_elapsed = MPI_Wtime() - started, elapsed_max = 0.0;
    mpi_ok(MPI_Allreduce(&local_elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "max elapsed");
    gpu_ok(cudaMemcpy(local.data(), field.current() + width, local.size() * sizeof(float),
                      cudaMemcpyDeviceToHost), "download local result");
    double local_sum = 0.0;
    for (float v : local) local_sum += v;
    double global_sum = 0.0;
    mpi_ok(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "global heat sum");
    std::vector<int> counts(ranks), displacements(ranks);
    for (int i = 0; i < ranks; ++i) {
        counts[i] = rows_for(i, ranks) * width;
        displacements[i] = first_row(i, ranks) * width;
    }
    std::vector<float> full(rank == 0 ? width * height : 0);
    mpi_ok(MPI_Gatherv(local.data(), rows * width, MPI_FLOAT, rank == 0 ? full.data() : nullptr,
                       counts.data(), displacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD), "gather full field");
    int passed = 1;
    if (rank == 0) {
        auto reference = cpu_reference();
        double max_error = 0.0, cpu_sum = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(full[i])) passed = 0;
            max_error = std::max(max_error, std::abs(static_cast<double>(full[i]) - reference[i]));
            cpu_sum += reference[i];
        }
        double sum_error = std::abs(global_sum - cpu_sum);
        std::cout << "nodes=" << nodes.size() << " ranks=" << ranks << " mode="
                  << (cuda_aware ? "CUDA-aware MPI" : "host staging") << " steps=" << steps
                  << " max_abs_error=" << max_error << " allreduce_sum_error=" << sum_error
                  << " slowest_rank_seconds=" << elapsed_max << '\n';
        if (max_error > 5e-5 || sum_error > 1e-4) passed = 0;
        if (passed) std::cout << "chapter 36 MPI halo heat: PASS\n";
    }
    mpi_ok(MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD), "broadcast verification");
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        bool cuda_aware = false, require_two_nodes = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--cuda-aware") cuda_aware = true;
            else if (arg == "--require-two-nodes") require_two_nodes = true;
            else throw std::runtime_error("usage: mpi_halo_heat [--cuda-aware] [--require-two-nodes]");
        }
        int code = run(rank, ranks, cuda_aware, require_two_nodes);
        MPI_Finalize();
        return code;
    } catch (const std::exception& e) {
        std::cerr << "rank " << rank << " FAIL: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
