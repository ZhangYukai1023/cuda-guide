#include "../../../common/cuda_support.cuh"
__global__ void matmul(const float* a,const float* b,float* c,int m,int n,int k) {
    int col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y*blockDim.y+threadIdx.y;
    if(row>=m || col>=n) return;
    float sum=0; for(int p=0;p<k;++p) sum+=a[row*k+p]*b[p*n+col];
    c[row*n+col]=sum;
}
// 一块处理一行，128 个线程进行两次树形归约，允许列数大于 128。
__global__ void softmax(const float* a,float* b,int cols) {
    __shared__ float work[128];
    int t=threadIdx.x,row=blockIdx.x;
    float maximum=-INFINITY;
    for(int j=t;j<cols;j+=128) maximum=fmaxf(maximum,a[row*cols+j]);
    work[t]=maximum; __syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(t<stride) work[t]=fmaxf(work[t],work[t+stride]);
        __syncthreads();
    }
    maximum=work[0]; __syncthreads(); // 所有人读完最大值后才能覆盖共享数组。
    float sum=0;
    for(int j=t;j<cols;j+=128) sum+=expf(a[row*cols+j]-maximum);
    work[t]=sum; __syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(t<stride) work[t]+=work[t+stride];
        __syncthreads();
    }
    for(int j=t;j<cols;j+=128) b[row*cols+j]=expf(a[row*cols+j]-maximum)/work[0];
}
void test_matmul(int m,int n,int k) {
    std::vector<float> a(m*k),b(k*n);
    for(size_t i=0;i<a.size();++i) a[i]=float(int(i%7)-3)/4;
    for(size_t i=0;i<b.size();++i) b[i]=float(int(i%5)-2)/8;
    if(m==2 && n==2 && k==3) { a={1,2,3,4,5,6}; b={1,2,3,4,5,6}; }
    std::vector<double> ref(m*n);
    for(int i=0;i<m;++i) for(int j=0;j<n;++j)
        for(int p=0;p<k;++p) ref[i*n+j]+=double(a[i*k+p])*b[p*n+j];
    DeviceBuffer<float> da(a.size()),db(b.size()),dc(ref.size()); da.upload(a); db.upload(b);
    matmul<<<dim3((n+15)/16,(m+15)/16),dim3(16,16)>>>(da.data,db.data,dc.data,m,n,k);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto got=dc.download(); verify("matmul",got,ref,1e-5,1e-5);
    if(m==2) { for(float v:got) std::printf("%.0f ",v); std::puts(""); }
}
int main() { return guarded([] {
    test_matmul(2,2,3); test_matmul(17,19,23);
    for(int cols:{1,3,129,257}) {
        int rows=3; std::vector<float> a(rows*cols);
        for(int j=0;j<cols;++j) {
            a[j]=1000+float(j%11); a[cols+j]=-1000-float(j%13); a[2*cols+j]=7;
        }
        std::vector<double> ref(a.size());
        for(int r=0;r<rows;++r) {
            double mx=*std::max_element(a.begin()+r*cols,a.begin()+(r+1)*cols),sum=0;
            for(int j=0;j<cols;++j) sum+=std::exp(double(a[r*cols+j])-mx);
            for(int j=0;j<cols;++j) ref[r*cols+j]=std::exp(double(a[r*cols+j])-mx)/sum;
        }
        DeviceBuffer<float> da(a.size()),db(a.size()); da.upload(a);
        softmax<<<rows,128>>>(da.data,db.data,cols);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        auto got=db.download(); verify("softmax",got,ref,2e-7,2e-5);
        for(int r=0;r<rows;++r) {
            double sum=0; for(int j=0;j<cols;++j) {
                if(got[r*cols+j]<0) throw std::runtime_error("negative probability");
                sum+=got[r*cols+j];
            }
            if(std::abs(sum-1)>2e-6) throw std::runtime_error("row sum mismatch");
        }
        std::printf("softmax cols=%d row_sums PASS\n",cols);
    }
}); }
