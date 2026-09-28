#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    throw std::runtime_error("CUDA failure"); } } while (false)

template <class T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t n) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n*sizeof(T))); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

__global__ void direct_dft(const float2* input, float2* output, int n, int inverse) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float real = 0, imag = 0;
    const float sign = inverse ? 1.f : -1.f;
    for (int t = 0; t < n; ++t) {
        const float phase = sign * 6.2831853071795864769f * k * t / n;
        const float c = cosf(phase), s = sinf(phase);
        const float2 v = input[t];
        real += v.x*c - v.y*s;
        imag += v.x*s + v.y*c;
    }
    const float factor = inverse ? 1.f/n : 1.f;
    output[k] = make_float2(real*factor, imag*factor);
}

__global__ void lowpass(float2* spectrum, int n, int cutoff) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int frequency = k <= n/2 ? k : k-n;
    if (abs(frequency) > cutoff) spectrum[k] = make_float2(0,0);
}

__global__ void multiply(const float2* a, const float2* b, float2* out, int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    out[k] = make_float2(a[k].x*b[k].x-a[k].y*b[k].y,
                         a[k].x*b[k].y+a[k].y*b[k].x);
}

void upload(float2* dst, const std::vector<float2>& src) {
    CUDA_CHECK(cudaMemcpy(dst,src.data(),src.size()*sizeof(float2),cudaMemcpyHostToDevice));
}
std::vector<float2> download(const float2* src, int n) {
    std::vector<float2> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(),src,n*sizeof(float2),cudaMemcpyDeviceToHost));
    return out;
}
void dft(const float2* src, float2* dst, int n, bool inverse=false) {
    direct_dft<<<(n+127)/128,128>>>(src,dst,n,inverse);
    CUDA_CHECK(cudaGetLastError());
}
void assert_error(const char* name, const std::vector<float2>& actual,
                  const std::vector<float2>& expected, double limit) {
    double max_error = 0;
    for (size_t i=0;i<actual.size();++i) {
        max_error=std::max(max_error,std::fabs(double(actual[i].x)-expected[i].x));
        max_error=std::max(max_error,std::fabs(double(actual[i].y)-expected[i].y));
    }
    std::printf("%s n=%zu max_abs_error=%.9g %s\n",name,actual.size(),max_error,
                max_error<=limit ? "PASS":"FAIL");
    if (!(max_error<=limit)) throw std::runtime_error(name);
}
std::vector<float2> cpu_dft(const std::vector<float2>& x) {
    const int n=static_cast<int>(x.size());
    std::vector<float2> out(n);
    for(int k=0;k<n;++k) {
        double re=0,im=0;
        for(int t=0;t<n;++t) {
            double phase=-6.2831853071795864769*double(k)*t/n;
            re+=x[t].x*std::cos(phase)-x[t].y*std::sin(phase);
            im+=x[t].x*std::sin(phase)+x[t].y*std::cos(phase);
        }
        out[k]=make_float2(float(re),float(im));
    }
    return out;
}
void run_dft() {
    constexpr int n=8;
    std::vector<float2> input(n), impulse(n,make_float2(0,0));
    impulse[0]=make_float2(1,0);
    for(int t=0;t<n;++t)
        input[t]=make_float2(.5f+std::sin(6.2831853071795864769*t/n),0);
    Buffer<float2> dx(n),dy(n),dz(n);
    upload(dx.ptr,impulse); dft(dx.ptr,dy.ptr,n);
    assert_error("impulse spectrum",download(dy.ptr,n),cpu_dft(impulse),1e-5);
    upload(dx.ptr,input); dft(dx.ptr,dy.ptr,n);
    assert_error("signal spectrum",download(dy.ptr,n),cpu_dft(input),1e-5);
    dft(dy.ptr,dz.ptr,n,true);
    assert_error("signal roundtrip",download(dz.ptr,n),input,1e-5);
}
void run_lowpass() {
    constexpr int n=64;
    std::vector<float2> input(n),expected(n);
    for(int t=0;t<n;++t) {
        const float low=std::sin(6.2831853071795864769*3*t/n);
        input[t]=make_float2(low+.5f*std::sin(6.2831853071795864769*12*t/n),0);
        expected[t]=make_float2(low,0);
    }
    Buffer<float2> dx(n),ds(n),dy(n);
    upload(dx.ptr,input); dft(dx.ptr,ds.ptr,n);
    lowpass<<<1,128>>>(ds.ptr,n,8); CUDA_CHECK(cudaGetLastError());
    dft(ds.ptr,dy.ptr,n,true);
    assert_error("lowpass k<=8",download(dy.ptr,n),expected,2e-5);
}
void run_convolution() {
    constexpr int n=16;
    std::vector<float2> signal(n,make_float2(0,0)),kernel(n,make_float2(0,0));
    std::vector<float2> expected(n,make_float2(0,0));
    for(int i=0;i<8;++i) signal[i]=make_float2(float(i+1),0);
    kernel[0]=make_float2(1,0);kernel[1]=make_float2(-2,0);kernel[2]=make_float2(1,0);
    for(int i=0;i<8;++i) for(int j=0;j<3;++j) expected[i+j].x+=signal[i].x*kernel[j].x;
    Buffer<float2> ds(n),dk(n),fs(n),fk(n),fm(n),out(n);
    upload(ds.ptr,signal);upload(dk.ptr,kernel);
    dft(ds.ptr,fs.ptr,n);dft(dk.ptr,fk.ptr,n);
    multiply<<<1,128>>>(fs.ptr,fk.ptr,fm.ptr,n);CUDA_CHECK(cudaGetLastError());
    dft(fm.ptr,out.ptr,n,true);
    assert_error("linear convolution padded16",download(out.ptr,n),expected,1e-4);
}
int main() {
    try { run_dft();run_lowpass();run_convolution();
        std::puts("chapter 32 direct frequency: PASS");return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"chapter 32 direct frequency: FAIL: %s\n",e.what());return 1;
    }
}
