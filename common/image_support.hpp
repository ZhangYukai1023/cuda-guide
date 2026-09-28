#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<float> test_image(int w,int h) {
    std::vector<float> a(w*h);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        float v=30+100.0f*x/std::max(1,w-1);
        if(x>w/4 && x<3*w/4 && y>h/4 && y<3*h/4) v=210;
        a[y*w+x]=v;
    }
    return a;
}
inline std::vector<float> noisy_image(const std::vector<float>& clean) {
    auto a=clean;
    for(size_t i=0;i<a.size();++i) {
        // 固定、可重复的椒盐噪声，不依赖随机数库。
        unsigned v=unsigned(i)*1664525u+1013904223u;
        if(v%23==0) a[i]=0;
        else if(v%29==0) a[i]=255;
    }
    return a;
}
inline double mse(const std::vector<float>& a,const std::vector<float>& b) {
    if(a.size()!=b.size() || a.empty()) throw std::runtime_error("invalid MSE input");
    double sum=0; for(size_t i=0;i<a.size();++i) { double d=a[i]-b[i]; sum+=d*d; }
    return sum/a.size();
}
// SVG 是按真实数值逐像素绘制的确定性可视化，不是生成式图像。
inline void save_image(const std::string& dir,const std::string& name,
                       const std::vector<float>& a,int w,int h) {
    if(a.size()!=size_t(w)*h) throw std::runtime_error("image size mismatch");
    std::filesystem::create_directories(dir);
    std::ofstream pgm(dir+"/"+name+".pgm",std::ios::binary);
    std::ofstream svg(dir+"/"+name+".svg");
    if(!pgm || !svg) throw std::runtime_error("cannot create image output");
    pgm<<"P5\n"<<w<<" "<<h<<"\n255\n";
    svg<<"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""<<w*4
       <<"\" height=\""<<h*4<<"\" viewBox=\"0 0 "<<w<<" "<<h
       <<"\" shape-rendering=\"crispEdges\">\n";
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        int v=int(std::lround(std::clamp(a[y*w+x],0.0f,255.0f)));
        auto byte=static_cast<unsigned char>(v); pgm.write(reinterpret_cast<char*>(&byte),1);
        svg<<"<rect x=\""<<x<<"\" y=\""<<y<<"\" width=\"1\" height=\"1\" fill=\"rgb("
           <<v<<","<<v<<","<<v<<")\"/>\n";
    }
    svg<<"</svg>\n";
    if(!pgm || !svg) throw std::runtime_error("image write failed");
}
