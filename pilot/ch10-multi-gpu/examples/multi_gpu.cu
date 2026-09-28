#include "../../../common/cuda_support.cuh"
#include <memory>
__global__ void transform(const int* input,int* output,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) output[i]=3*input[i]+1;
}
struct Shard {
    int device,begin,count;
    // device 必须在构造前设为当前设备；显式在正确设备上析构。
    Stream stream;
    DeviceBuffer<int> input,output;
    PinnedBuffer<int> host_input,host_output;
    Shard(int d,int b,int n):device(d),begin(b),count(n),input(n),output(n),
                            host_input(n),host_output(n) {}
};
int main(int argc,char** argv) {
    bool require_two=argc==2 && std::string(argv[1])=="--require-two";
    if(argc>1 && !require_two) { std::fprintf(stderr,"usage: multi_gpu [--require-two]\n"); return 1; }
    int count=0;
    cudaError_t status=cudaGetDeviceCount(&count);
    if(status!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(status)); return 1; }
    if(require_two && count<2) { std::puts("SKIP: two CUDA devices required"); return 77; }
    if(count<1) { std::fprintf(stderr,"no CUDA device\n"); return 1; }
    return guarded([&] {
        int devices=std::min(count,2);
        for(int n:{1,17,1003}) {
            std::vector<std::unique_ptr<Shard>> shards;
            // 即使异常退出，也先切换设备并同步，再销毁与该设备关联的资源。
            auto cleanup=[&] {
                for(auto& s:shards) if(s) {
                    cudaSetDevice(s->device); cudaStreamSynchronize(s->stream.value); s.reset();
                }
            };
            try {
                for(int d=0;d<devices;++d) {
                    int begin=n*d/devices,end=n*(d+1)/devices;
                    if(begin==end) continue;
                    CUDA_CHECK(cudaSetDevice(d));
                    shards.emplace_back(std::make_unique<Shard>(d,begin,end-begin));
                    auto& s=*shards.back();
                    for(int i=0;i<s.count;++i) s.host_input.data[i]=s.begin+i;
                    CUDA_CHECK(cudaMemcpyAsync(s.input.data,s.host_input.data,s.count*sizeof(int),
                                              cudaMemcpyHostToDevice,s.stream.value));
                    transform<<<(s.count+127)/128,128,0,s.stream.value>>>
                        (s.input.data,s.output.data,s.count);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpyAsync(s.host_output.data,s.output.data,s.count*sizeof(int),
                                              cudaMemcpyDeviceToHost,s.stream.value));
                }
                std::vector<int> got(n),ref(n);
                for(auto& p:shards) {
                    auto& s=*p; CUDA_CHECK(cudaSetDevice(s.device));
                    CUDA_CHECK(cudaStreamSynchronize(s.stream.value));
                    std::copy(s.host_output.data,s.host_output.data+s.count,got.begin()+s.begin);
                }
                for(int i=0;i<n;++i) ref[i]=3*i+1;
                verify("sharded_transform",got,ref);
                std::printf("devices_used=%d n=%d\n",devices,n);
                cleanup();
            } catch(...) { cleanup(); throw; }
        }
        if(devices==1) std::puts("Single-device fallback only; multi-device path NOT verified.");
    });
}
