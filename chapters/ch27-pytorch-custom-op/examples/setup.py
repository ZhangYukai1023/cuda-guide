from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

setup(
    name="cuda_guide_ops",
    ext_modules=[
        CUDAExtension(
            name="_C",
            sources=[str(HERE / "ops.cpp"), str(HERE / "ops_cuda.cu")],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
