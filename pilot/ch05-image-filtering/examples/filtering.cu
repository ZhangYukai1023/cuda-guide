#include "../../../common/cuda_support.cuh"
#include "../../../common/image_support.hpp"
__global__ void filter3(const float* input,float* output,int w,int h,bool median) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h) return;
    float values[9],sum=0; int k=0;
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        int sx=min(max(x+dx,0),w-1),sy=min(max(y+dy,0),h-1);
        values[k]=input[sy*w+sx]; sum+=values[k++];
    }
    if(median) {
        for(int i=1;i<9;++i) for(int j=i;j>0 && values[j]<values[j-1];--j) {
            float t=values[j]; values[j]=values[j-1]; values[j-1]=t;
        }
    }
    output[y*w+x]=median?values[4]:sum/9;
}
std::vector<double> reference(const std::vector<float>& a,int w,int h,bool median) {
    std::vector<double> out(a.size());
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        std::vector<double> neighbors;
        for(int yy=y-1;yy<=y+1;++yy) for(int xx=x-1;xx<=x+1;++xx)
            neighbors.push_back(a[std::clamp(yy,0,h-1)*w+std::clamp(xx,0,w-1)]);
        std::sort(neighbors.begin(),neighbors.end());
        double sum=0; for(double v:neighbors) sum+=v;
        out[y*w+x]=median?neighbors[4]:sum/9;
    }
    return out;
}
int main(int argc,char** argv) { return guarded([&] {
    std::string dir=argc>1?argv[1]:"build/filtering-images";
    for(auto shape:std::vector<std::pair<int,int>>{{1,1},{3,3},{65,49}}) {
        int w=shape.first,h=shape.second;
        auto clean=test_image(w,h), input=noisy_image(clean);
        if(w==3) { clean.assign(9,10); input=clean; input[4]=255; }
        if(w==65) { save_image(dir,"clean",clean,w,h); save_image(dir,"noisy",input,w,h); }
        DeviceBuffer<float> a(input.size()),b(input.size()); a.upload(input);
        for(bool median:{false,true}) {
            filter3<<<dim3((w+15)/16,(h+15)/16),dim3(16,16)>>>(a.data,b.data,w,h,median);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            auto got=b.download();
            verify(median?"median3":"box3",got,reference(input,w,h,median),3e-5,1e-6);
            if(w==3) std::printf("center_%s=%.6f\n",median?"median":"box",got[4]);
            if(w==65) {
                save_image(dir,median?"median":"box",got,w,h);
                std::printf("%s MSE_to_clean=%.6f (noisy=%.6f)\n",
                    median?"median":"box",mse(got,clean),mse(input,clean));
            }
        }
    }
}); }
