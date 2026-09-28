#include "../../../common/cuda_support.cuh"

__global__ void grid_stride(int* out, int n) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x;
         i < n; i += blockDim.x*gridDim.x) out[i] = 3*i - 7;
}
__global__ void image_coordinates(int* out, int width, int height) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < width && y < height) out[y*width+x] = 100*y+x;
}
int main() { return guarded([] {
    for (int n : {0, 1, 31, 32, 33, 1003}) {
        DeviceBuffer<int> out(n);
        std::vector<int> ref(n);
        for (int i=0; i<n; ++i) ref[i] = 3*i-7;
        if (n) {
            grid_stride<<<2, 32>>>(out.data,n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        verify("grid_stride",out.download(),ref);
    }
    for (auto shape : std::vector<std::pair<int,int>>{{3,2}, {37,19}, {1,1}}) {
        int w=shape.first, h=shape.second;
        DeviceBuffer<int> out(w*h);
        image_coordinates<<<dim3((w+15)/16,(h+7)/8),dim3(16,8)>>>(out.data,w,h);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> ref(w*h);
        for (int y=0;y<h;++y) for (int x=0;x<w;++x) ref[y*w+x]=100*y+x;
        auto got=out.download();
        verify("image_coordinates",got,ref);
        if(w==3) { for(int v:got) std::printf("%d ",v); std::puts(""); }
    }
}); }
