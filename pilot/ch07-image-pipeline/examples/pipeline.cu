#include "../../../common/cuda_support.cuh"
#include "../../../common/image_support.hpp"
__global__ void invert(const float* a,float* b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=255-a[i];
}
__global__ void brightness(const float* a,float* b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=fminf(a[i]+20,255);
}
__global__ void fused(const float* a,float* b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=fminf(275-a[i],255);
}
int main(int argc,char** argv) { return guarded([&] {
    std::string dir=argc>1?argv[1]:"build/pipeline-images";
    int w=65,h=49,n=w*h; auto input=test_image(w,h);
    std::vector<float> ref(n); for(int i=0;i<n;++i) ref[i]=std::min(255-input[i]+20,255.0f);
    DeviceBuffer<float> a(n),tmp(n),out(n); a.upload(input);
    invert<<<(n+127)/128,128>>>(a.data,tmp.data,n);
    CUDA_CHECK(cudaGetLastError());
    brightness<<<(n+127)/128,128>>>(tmp.data,out.data,n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    verify("two_stage",out.download(),ref);
    fused<<<(n+127)/128,128>>>(a.data,out.data,n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    verify("fused",out.download(),ref);
    save_image(dir,"input",input,w,h); save_image(dir,"output",out.download(),w,h);

    // 两个槽各自拥有流、设备内存与页锁定主机内存，复用前等待该槽完成。
    Stream streams[2];
    DeviceBuffer<float> da0(n),da1(n),db0(n),db1(n);
    PinnedBuffer<float> hi0(n),hi1(n),ho0(n),ho1(n);
    float* da[]={da0.data,da1.data}; float* db[]={db0.data,db1.data};
    float* hi[]={hi0.data,hi1.data}; float* ho[]={ho0.data,ho1.data};
    int pending[2]={-1,-1};
    auto collect=[&](int slot) {
        if(pending[slot]<0) return;
        CUDA_CHECK(cudaStreamSynchronize(streams[slot].value));
        std::vector<float> got(ho[slot],ho[slot]+n),expected(n);
        for(int i=0;i<n;++i) expected[i]=std::min(275-float((i+pending[slot])%256),255.0f);
        verify("stream_frame",got,expected);
        pending[slot]=-1;
    };
    for(int frame=0;frame<5;++frame) {
        int s=frame%2; collect(s);
        for(int i=0;i<n;++i) hi[s][i]=float((i+frame)%256);
        CUDA_CHECK(cudaMemcpyAsync(da[s],hi[s],n*sizeof(float),cudaMemcpyHostToDevice,streams[s].value));
        fused<<<(n+127)/128,128,0,streams[s].value>>>(da[s],db[s],n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(ho[s],db[s],n*sizeof(float),cudaMemcpyDeviceToHost,streams[s].value));
        pending[s]=frame;
    }
    collect(0); collect(1);
}); }
