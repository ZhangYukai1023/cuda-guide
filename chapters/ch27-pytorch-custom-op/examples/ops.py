"""Load the C++/CUDA registrations, then add fake and autograd registrations."""

import os
from pathlib import Path

import torch

_library = os.environ.get("CUDA_GUIDE_OPS_LIBRARY")
if not _library:
    _candidates = sorted(Path(__file__).resolve().parent.glob("_C*.so"))
    if not _candidates:
        raise RuntimeError("build the CUDA extension or set CUDA_GUIDE_OPS_LIBRARY")
    _library = str(_candidates[0])
torch.ops.load_library(_library)  # runs the static TORCH_LIBRARY registrations


@torch.library.register_fake("cuda_guide_ops::scale")
def _scale_fake(x, factor):
    torch._check(x.dtype == torch.float32)
    torch._check(x.device.type == "cuda")
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.register_fake("cuda_guide_ops::scale_relu")
def _scale_relu_fake(x, factor):
    torch._check(x.dtype == torch.float32)
    torch._check(x.device.type == "cuda")
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _scale_setup(ctx, inputs, output):
    _x, factor = inputs
    ctx.factor = factor


def _scale_backward(ctx, grad):
    return grad * ctx.factor, None


torch.library.register_autograd(
    "cuda_guide_ops::scale", _scale_backward, setup_context=_scale_setup
)


def _scale_relu_setup(ctx, inputs, output):
    x, factor = inputs
    ctx.save_for_backward(x)
    ctx.factor = factor


def _scale_relu_backward(ctx, grad):
    (x,) = ctx.saved_tensors
    return grad * (x * ctx.factor > 0).to(grad.dtype) * ctx.factor, None


torch.library.register_autograd(
    "cuda_guide_ops::scale_relu", _scale_relu_backward, setup_context=_scale_relu_setup
)


def scale(x, factor):
    return torch.ops.cuda_guide_ops.scale(x, factor)


def scale_relu(x, factor):
    return torch.ops.cuda_guide_ops.scale_relu(x, factor)
