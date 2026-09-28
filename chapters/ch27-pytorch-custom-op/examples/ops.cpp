#include <torch/library.h>

TORCH_LIBRARY(cuda_guide_ops, m) {
    m.def("scale(Tensor x, float factor) -> Tensor");
    m.def("scale_relu(Tensor x, float factor) -> Tensor");
}
