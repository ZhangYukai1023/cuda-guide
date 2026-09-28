#include "../../../common/cuda_support.cuh"
#include "../../../common/image_support.hpp"
// 输出坐标先反向映射到输入；超出输入的坐标夹到边缘。
__global__ void sample(const float* a,float* b,int w,int h,int ow,int oh,
                       float sx,float sy,float tx,float ty,bool linear) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=ow || y>=oh) return;
    float fx=fminf(fmaxf(sx*x+tx,0),float(w-1));
    float fy=fminf(fmaxf(sy*y+ty,0),float(h-1));
    if(!linear) { b[y*ow+x]=a[int(floorf(fy+0.5f))*w+int(floorf(fx+0.5f))]; return; }
    int x0=int(floorf(fx)),y0=int(floorf(fy)),x1=min(x0+1,w-1),y1=min(y0+1,h-1);
    float u=fx-x0,v=fy-y0;
    b[y*ow+x]=(1-v)*((1-u)*a[y0*w+x0]+u*a[y0*w+x1])
                 +v*((1-u)*a[y1*w+x0]+u*a[y1*w+x1]);
}
std::vector<double> reference(const std::vector<float>& a,int w,int h,int ow,int oh,
                              float sx,float sy,float tx,float ty,bool linear) {
    std::vector<double> out(ow*oh);
    for(int y=0;y<oh;++y) for(int x=0;x<ow;++x) {
        double fx=std::clamp(double(sx)*x+tx,0.0,double(w-1));
        double fy=std::clamp(double(sy)*y+ty,0.0,double(h-1));
        if(!linear) { out[y*ow+x]=a[int(std::floor(fy+0.5))*w+int(std::floor(fx+0.5))]; continue; }
        int x0=int(fx),y0=int(fy); double sum=0;
        for(int j=0;j<2;++j) for(int i=0;i<2;++i)
            sum+=a[std::min(y0+j,h-1)*w+std::min(x0+i,w-1)]
                *(i?fx-x0:1-(fx-x0))*(j?fy-y0:1-(fy-y0));
        out[y*ow+x]=sum;
    }
    return out;
}
void run(const std::vector<float>& a,int w,int h,int ow,int oh,
         float sx,float sy,float tx,float ty,bool linear,
         const char* label,const std::string& dir,bool save) {
    DeviceBuffer<float> da(a.size()),db(ow*oh); da.upload(a);
    sample<<<dim3((ow+15)/16,(oh+15)/16),dim3(16,16)>>>
        (da.data,db.data,w,h,ow,oh,sx,sy,tx,ty,linear);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto got=db.download();
    verify(label,got,reference(a,w,h,ow,oh,sx,sy,tx,ty,linear),1e-4,1e-6);
    if(save) save_image(dir,label,got,ow,oh);
    if(ow==1) std::printf("%s_value=%.6f\n",label,got[0]);
}
int main(int argc,char** argv) { return guarded([&] {
    std::string dir=argc>1?argv[1]:"build/resampling-images";
    run({0,10,20,30},2,2,1,1,0,0,0.25f,0.5f,true,"tiny_bilinear",dir,false);
    run({17},1,1,3,2,1,1,-2,-2,true,"singleton",dir,false);
    int w=33,h=25,ow=66,oh=50; auto a=test_image(w,h);
    save_image(dir,"input",a,w,h);
    run(a,w,h,w,h,1,1,0,0,true,"identity",dir,false);
    // half-pixel：输入坐标 (输出下标+0.5)/2-0.5。
    run(a,w,h,ow,oh,0.5f,0.5f,-0.25f,-0.25f,false,"nearest",dir,true);
    run(a,w,h,ow,oh,0.5f,0.5f,-0.25f,-0.25f,true,"bilinear",dir,true);
    // 向右平移 5、向下平移 3：输入坐标 = 输出坐标 - 位移。
    run(a,w,h,w,h,1,1,-5,-3,true,"translated",dir,true);
}); }
