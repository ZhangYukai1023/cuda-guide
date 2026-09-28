"""Correctness checks for the CUDA custom operators; exits nonzero on missing CUDA."""

import sys

import torch

import ops


def check_close(name, actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    print(f"{name}: shape={tuple(actual.shape)} dtype={actual.dtype} PASS")


def main():
    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda} available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required; operator was not tested")
    torch.manual_seed(20260928)
    for shape in [(0,), (7,), (3, 5), (1003,)]:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        check_close("scale", ops.scale(x, 1.25), x * 1.25)
        check_close("scale_relu", ops.scale_relu(x, 1.25), torch.relu(x * 1.25))

    base = torch.randn((5, 3), device="cuda", dtype=torch.float32)
    view = base.t()  # shape 3x5, non-contiguous
    assert not view.is_contiguous()
    check_close("noncontiguous_scale", ops.scale(view, -0.75), view * -0.75)
    check_close("noncontiguous_scale_relu", ops.scale_relu(view, -0.75), torch.relu(view * -0.75))

    x = torch.tensor([-1.7, -0.8, 0.3, 1.2], device="cuda", requires_grad=True)
    y = ops.scale_relu(x, 1.25)
    y.sum().backward()
    expected_grad = torch.where(x.detach() > 0, 1.25, 0.0)
    check_close("scale_relu_backward", x.grad, expected_grad)
    finite_difference = []
    for i in range(x.numel()):
        plus = x.detach().clone()
        minus = x.detach().clone()
        plus[i] += 1e-3
        minus[i] -= 1e-3
        finite_difference.append(
            (ops.scale_relu(plus, 1.25).sum().item() -
             ops.scale_relu(minus, 1.25).sum().item()) / 2e-3
        )
    torch.testing.assert_close(
        x.grad, torch.tensor(finite_difference, device="cuda"), rtol=2e-3, atol=2e-3
    )
    print("scale_relu_finite_difference: PASS")

    x2 = torch.tensor([1.0, -2.0, 0.5], device="cuda", requires_grad=True)
    ops.scale(x2, -0.75).sum().backward()
    check_close("scale_backward", x2.grad, torch.full_like(x2, -0.75))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.arange(17, device="cuda", dtype=torch.float32) - 8
        streamed = ops.scale_relu(source, 0.5)
        expected = torch.relu(source * 0.5)
    stream.synchronize()
    check_close("current_stream", streamed, expected)

    for invalid in [torch.ones(3), torch.ones(3, device="cuda", dtype=torch.float16)]:
        try:
            ops.scale(invalid, 2.0)
        except (RuntimeError, TypeError):
            pass
        else:
            raise AssertionError("unsupported input was accepted")

    torch.library.opcheck(torch.ops.cuda_guide_ops.scale_relu.default, (
        torch.randn(2, 3, device="cuda"), 1.25
    ))
    print("opcheck: PASS")
    print("chapter 27 custom operators: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"chapter 27 custom operators: FAIL: {exc}", file=sys.stderr)
        raise
