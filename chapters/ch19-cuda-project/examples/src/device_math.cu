#include <cuda_runtime.h>

namespace guide {
__device__ float affine_device(float x, float scale, float bias) {
    return x * scale + bias;
}
} // namespace guide
