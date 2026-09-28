#include "../../../common/cuda_support.cuh"
#include <chrono>
__global__ void affine(const float* a,float* b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) b[i]=2.0f*a[i]+1.0f;
}
int main() { return guarded([] {
    constexpr int n=1<<20, warmup=5, repeats=100;
    std::vector<float> input(n),ref(n);
    for(int i=0;i<n;++i) { input[i]=float(i%101)/8; ref[i]=2*input[i]+1; }
    DeviceBuffer<float> a(n),b(n); a.upload(input);
    for(int i=0;i<warmup;++i) affine<<<(n+255)/256,256>>>(a.data,b.data,n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    Event start,stop;
    CUDA_CHECK(cudaEventRecord(start.value));
    for(int i=0;i<repeats;++i) affine<<<(n+255)/256,256>>>(a.data,b.data,n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop.value)); CUDA_CHECK(cudaEventSynchronize(stop.value));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));
    verify("affine",b.download(),ref);
    // 端到端范围：已分配的缓冲区之间 H2D + kernel + D2H；不含分配。
    std::vector<float> output(n);
    auto execute=[&] {
        CUDA_CHECK(cudaMemcpy(a.data,input.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        affine<<<(n+255)/256,256>>>(a.data,b.data,n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(output.data(),b.data,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    for(int i=0;i<warmup;++i) execute();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<repeats;++i) execute();
    auto end=std::chrono::steady_clock::now();
    verify("end_to_end",output,ref);
    std::printf("n=%d warmup=%d repeats=%d\n",n,warmup,repeats);
    std::printf("kernel_sequence_mean_ms=%.6f\n",ms/repeats);
    std::printf("end_to_end_mean_ms=%.6f (H2D+kernel+D2H, allocation excluded)\n",
        std::chrono::duration<double,std::milli>(end-begin).count()/repeats);
}); }
