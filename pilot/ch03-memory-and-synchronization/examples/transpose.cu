#include "../../../common/cuda_support.cuh"
constexpr int TILE=16;
__global__ void naive(const int* a,int* b,int w,int h) {
    int x=blockIdx.x*TILE+threadIdx.x, y=blockIdx.y*TILE+threadIdx.y;
    if(x<w && y<h) b[x*h+y]=a[y*w+x];
}
__global__ void tiled(const int* a,int* b,int w,int h) {
    __shared__ int tile[TILE][TILE+1];
    int x=blockIdx.x*TILE+threadIdx.x, y=blockIdx.y*TILE+threadIdx.y;
    if(x<w && y<h) tile[threadIdx.y][threadIdx.x]=a[y*w+x];
    // 所有线程都到达屏障；不能把屏障放进上面的分支。
    __syncthreads();
    int ox=blockIdx.y*TILE+threadIdx.x;
    int oy=blockIdx.x*TILE+threadIdx.y;
    if(ox<h && oy<w) b[oy*h+ox]=tile[threadIdx.x][threadIdx.y];
}
int main() { return guarded([] {
    for(auto shape:std::vector<std::pair<int,int>>{{3,2},{1,1},{31,17},{64,48}}) {
        int w=shape.first,h=shape.second;
        std::vector<int> a(w*h),ref(w*h);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            a[y*w+x]=y*w+x; ref[x*h+y]=a[y*w+x];
        }
        DeviceBuffer<int> da(a.size()),db(a.size()); da.upload(a);
        dim3 block(TILE,TILE),grid((w+TILE-1)/TILE,(h+TILE-1)/TILE);
        naive<<<grid,block>>>(da.data,db.data,w,h);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        verify("naive_transpose",db.download(),ref);
        tiled<<<grid,block>>>(da.data,db.data,w,h);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        auto got=db.download(); verify("tiled_transpose",got,ref);
        if(w==3) { for(int v:got) std::printf("%d ",v); std::puts(""); }
    }
}); }
