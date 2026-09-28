#include "../../../common/cuda_support.cuh"
#include "../../../common/image_support.hpp"
__global__ void step(const float* a,float* b,int w,int h,float alpha) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h) return;
    int i=y*w+x;
    if(x==0 || y==0 || x==w-1 || y==h-1) { b[i]=a[i]; return; }
    b[i]=a[i]+alpha*(a[i-1]+a[i+1]+a[i-w]+a[i+w]-4*a[i]);
}
void run(int w,int h,int steps,const std::string& dir) {
    std::vector<float> input(w*h,0); input[(h/2)*w+w/2]=100;
    DeviceBuffer<float> a(input.size()),b(input.size()); a.upload(input);
    float* src=a.data; float* dst=b.data;
    constexpr float alpha=0.2f;
    // CPU 使用 double；系数取与 GPU 相同的 float 值再提升，隔离算术误差。
    std::vector<double> ref(input.begin(),input.end()),next(ref.size());
    for(int t=0;t<steps;++t) {
        step<<<dim3((w+15)/16,(h+15)/16),dim3(16,16)>>>(src,dst,w,h,alpha);
        CUDA_CHECK(cudaGetLastError()); std::swap(src,dst);
        next=ref;
        for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x) {
            int i=y*w+x;
            next[i]=ref[i]+double(alpha)*(ref[i-1]+ref[i+1]+ref[i-w]+ref[i+w]-4*ref[i]);
        }
        ref.swap(next);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(input.size());
    CUDA_CHECK(cudaMemcpy(got.data(),src,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
    verify("heat",got,ref,2e-5,2e-5);
    for(float v:got) if(v<0 || v>100) throw std::runtime_error("maximum principle failed");
    std::printf("shape=%dx%d steps=%d center=%.6f range=[%.6f,%.6f]\n",
        w,h,steps,got[(h/2)*w+w/2],*std::min_element(got.begin(),got.end()),
        *std::max_element(got.begin(),got.end()));
    if(w==33) {
        // 固定 0..100 映射到灰度 0..255；输入与输出使用相同标尺。
        for(float& v:input) v*=2.55f; for(float& v:got) v*=2.55f;
        save_image(dir,"initial",input,w,h); save_image(dir,"diffused",got,w,h);
    }
}
int main(int argc,char** argv) { return guarded([&] {
    std::string dir=argc>1?argv[1]:"build/heat-images";
    run(3,3,1,dir); run(7,5,0,dir); run(7,5,1,dir);
    run(7,5,2,dir); run(33,25,20,dir);
}); }
