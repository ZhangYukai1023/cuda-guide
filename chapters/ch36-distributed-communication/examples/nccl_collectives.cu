#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void gpu_ok(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
}

void nccl_ok(ncclResult_t code, const char* what) {
    if (code != ncclSuccess) throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(code));
}

void mpi_ok(int code, const char* what) {
    if (code != MPI_SUCCESS) throw std::runtime_error(std::string(what) + ": MPI error " + std::to_string(code));
}

class DeviceFloats {
public:
    explicit DeviceFloats(size_t n) : n_(n) { gpu_ok(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(float)), "cudaMalloc"); }
    ~DeviceFloats() { if (p_) cudaFree(p_); }
    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;
    float* data() const { return p_; }
    void upload(const std::vector<float>& host) {
        if (host.size() != n_) throw std::runtime_error("upload size mismatch");
        gpu_ok(cudaMemcpy(p_, host.data(), n_ * sizeof(float), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<float> download() const {
        std::vector<float> host(n_);
        gpu_ok(cudaMemcpy(host.data(), p_, n_ * sizeof(float), cudaMemcpyDeviceToHost), "download");
        return host;
    }
private:
    size_t n_;
    float* p_ = nullptr;
};

class Communicator {
public:
    Communicator(int ranks, ncclUniqueId id, int rank) { nccl_ok(ncclCommInitRank(&comm_, ranks, id, rank), "ncclCommInitRank"); }
    ~Communicator() { if (comm_) ncclCommDestroy(comm_); }
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    ncclComm_t get() const { return comm_; }
private:
    ncclComm_t comm_{};
};

class Stream {
public:
    Stream() { gpu_ok(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
    void wait() { gpu_ok(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }
private:
    cudaStream_t stream_{};
};

void update_error(double& max_error, float actual, double expected) {
    if (!std::isfinite(actual)) throw std::runtime_error("collective returned nonfinite value");
    max_error = std::max(max_error, std::abs(static_cast<double>(actual) - expected));
}

int run(int rank, int ranks) {
    if (ranks < 2) {
        if (rank == 0) std::cout << "SKIP: NCCL example requires >=2 ranks\n";
        return 77;
    }
    MPI_Comm local = MPI_COMM_NULL;
    mpi_ok(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local), "split local ranks");
    int local_rank = 0;
    mpi_ok(MPI_Comm_rank(local, &local_rank), "local rank");
    mpi_ok(MPI_Comm_free(&local), "free local communicator");
    int visible = 0;
    cudaError_t status = cudaGetDeviceCount(&visible);
    if (status == cudaErrorNoDevice) visible = 0;
    else gpu_ok(status, "cudaGetDeviceCount");
    int available = local_rank < visible ? 1 : 0, all_available = 0;
    mpi_ok(MPI_Allreduce(&available, &all_available, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "GPU availability");
    if (!all_available) {
        if (rank == 0) std::cout << "SKIP: at least one MPI rank lacks its local GPU\n";
        return 77;
    }
    gpu_ok(cudaSetDevice(local_rank), "cudaSetDevice");
    ncclUniqueId id{};
    if (rank == 0) nccl_ok(ncclGetUniqueId(&id), "ncclGetUniqueId");
    mpi_ok(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD), "broadcast NCCL ID");
    Communicator comm(ranks, id, rank);
    Stream stream;
    constexpr int allreduce_count = 8, per_rank = 4;
    DeviceFloats ar_send(allreduce_count), ar_recv(allreduce_count);
    DeviceFloats gather_send(per_rank), gather_recv(per_rank * ranks);
    DeviceFloats scatter_send(per_rank * ranks), scatter_recv(per_rank);
    std::vector<float> ar_input(allreduce_count), gather_input(per_rank), scatter_input(per_rank * ranks);
    for (int i = 0; i < allreduce_count; ++i) ar_input[i] = static_cast<float>(rank + 1 + 0.01 * i);
    for (int i = 0; i < per_rank; ++i) gather_input[i] = static_cast<float>(rank * 10 + i);
    for (int i = 0; i < per_rank * ranks; ++i) scatter_input[i] = static_cast<float>(rank + 1 + 0.001 * i);
    ar_send.upload(ar_input);
    gather_send.upload(gather_input);
    scatter_send.upload(scatter_input);
    nccl_ok(ncclAllReduce(ar_send.data(), ar_recv.data(), allreduce_count, ncclFloat,
                          ncclSum, comm.get(), stream.get()), "ncclAllReduce");
    nccl_ok(ncclAllGather(gather_send.data(), gather_recv.data(), per_rank, ncclFloat,
                          comm.get(), stream.get()), "ncclAllGather");
    nccl_ok(ncclReduceScatter(scatter_send.data(), scatter_recv.data(), per_rank,
                              ncclFloat, ncclSum, comm.get(), stream.get()), "ncclReduceScatter");
    stream.wait();
    auto ar = ar_recv.download(), gathered = gather_recv.download(), scattered = scatter_recv.download();
    double max_error = 0.0;
    double rank_sum = static_cast<double>(ranks) * (ranks + 1) / 2;
    for (int i = 0; i < allreduce_count; ++i)
        update_error(max_error, ar[i], rank_sum + ranks * 0.01 * i);
    for (int source = 0; source < ranks; ++source)
        for (int i = 0; i < per_rank; ++i)
            update_error(max_error, gathered[source * per_rank + i], source * 10 + i);
    for (int i = 0; i < per_rank; ++i)
        update_error(max_error, scattered[i], rank_sum + ranks * 0.001 * (rank * per_rank + i));
    double global_max = 0.0;
    mpi_ok(MPI_Allreduce(&max_error, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "global max error");
    if (rank == 0) std::cout << "NCCL ranks=" << ranks << " AllReduce/AllGather/ReduceScatter max_abs_error="
                             << global_max << '\n';
    if (global_max > 1e-4) throw std::runtime_error("NCCL collective differs from CPU formulas");
    if (rank == 0) std::cout << "chapter 36 NCCL collectives: PASS\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        int result = run(rank, ranks);
        MPI_Finalize();
        return result;
    } catch (const std::exception& e) {
        std::cerr << "rank " << rank << " FAIL: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
