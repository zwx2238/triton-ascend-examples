# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Tiling
=======
Demonstrates improving NPU resource utilization by matching kernel count to physical cores.

Both kernels run on NPU with different parallelization strategies:
- GPU-style: Launches too many small kernels - High scheduling overhead, poor performance
- NPU-optimized: Launches fewer kernels (close to physical core count) - Better utilization

The key is to match the number of launched kernels to the NPU's physical cores.
"""

import torch
import torch_npu
import triton
import triton.language as tl

from utils import is_npu
from prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary


@triton.jit
def npu_vector_add_gpu_style_kernel(
    x,                          # [Tensor] input tensor
    y,                          # [Tensor] input tensor
    z,                          # [Tensor] output tensor
    vector_len: tl.constexpr,   # len of the vector
    BLOCK_SIZE: tl.constexpr    # size of each block (small, launches many kernels)
):
    """
    GPU-style implementation on NPU: Launches many small kernels.
    When BLOCK_SIZE is small, we launch many kernels, causing high scheduling overhead.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vector_len

    x1 = tl.load(x + offset, mask=len_mask, other=0)
    y1 = tl.load(y + offset, mask=len_mask, other=0)
    z1 = x1 + y1
    tl.store(z + offset, z1, mask=len_mask)


@triton.jit
def npu_vector_add_optimized_kernel(
    x,                          # [Tensor] input tensor
    y,                          # [Tensor] input tensor
    z,                          # [Tensor] output tensor
    vector_len: tl.constexpr,   # len of the vector
    BLOCK_SIZE: tl.constexpr,   # total size to process per kernel
    CHUNK_SIZE: tl.constexpr    # size of each chunk in the loop
):
    """
    NPU-optimized implementation: Launches fewer kernels with loops.
    By using larger BLOCK_SIZE and loops, we launch fewer kernels that match physical cores.
    """
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # Calculate number of loops needed
    num_chunks = tl.cdiv(BLOCK_SIZE, CHUNK_SIZE)

    # Process data in chunks using loop
    for i in range(num_chunks):
        chunk_offset = base_offset + i * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
        len_mask = chunk_offset < vector_len

        x1 = tl.load(x + chunk_offset, mask=len_mask, other=0)
        y1 = tl.load(y + chunk_offset, mask=len_mask, other=0)
        z1 = x1 + y1
        tl.store(z + chunk_offset, z1, mask=len_mask)


def run(kernel_name="optimized", result_paths=None):
    """
    Run tiling test on NPU.

    Args:
        kernel_name: Kernel implementation to use ("gpu_style" or "optimized")
        result_paths: Dictionary to store profiling result paths
    """
    device = "npu"
    vector_len = 65536  # Total vector length

    # Select kernel implementation and grid configuration
    if kernel_name == "gpu_style":
        # GPU-style: Small blocks, many kernels (e.g., 128 kernels)
        BLOCK_SIZE = 512
        CHUNK_SIZE = 512  # Not used in GPU-style
        BLOCK_DIM = vector_len // BLOCK_SIZE  # 128 kernels
        kernel = npu_vector_add_gpu_style_kernel
        kernel_label = f"GPU-style kernel (launches {BLOCK_DIM} kernels)"
    else:
        # NPU-optimized: Larger blocks with loops, fewer kernels (e.g., 16 kernels)
        BLOCK_SIZE = 4096   # Each kernel processes more data
        CHUNK_SIZE = 512    # Process in chunks within loop
        BLOCK_DIM = vector_len // BLOCK_SIZE  # 16 kernels (closer to physical cores)
        kernel = npu_vector_add_optimized_kernel
        kernel_label = f"NPU-optimized kernel (launches {BLOCK_DIM} kernels)"

    x = torch.randint(0, 100, (vector_len,), device=device, dtype=torch.int32)
    y = torch.randint(0, 100, (vector_len,), device=device, dtype=torch.int32)
    z = torch.zeros((vector_len,), device=device, dtype=torch.int32)

    # Warm up and correctness check
    if kernel_name == "gpu_style":
        kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)
    else:
        kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE, CHUNK_SIZE)
    torch.npu.synchronize()

    # Verify correctness
    expected = x + y
    torch.testing.assert_close(z, expected)
    print(f"==== {kernel_label} - correctness check passed")

    # Profile performance
    def kernel_wrapper():
        if kernel_name == "gpu_style":
            kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)
        else:
            kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE, CHUNK_SIZE)

    result_path = f"./result_profiling_tiling_{kernel_name}"
    print(f"==== Profiling {kernel_label}...")
    profiler_wrapper(kernel_wrapper, result_path=result_path)

    # Store result path for comparison
    if result_paths is not None:
        result_paths[kernel_name] = result_path


if __name__ == "__main__":
    if not is_npu():
        print("This example requires NPU device")
        exit(1)

    # Compare both implementations on NPU
    print("=" * 80)
    print("Running on NPU - Comparing GPU-style vs NPU-optimized tiling")
    print("=" * 80)

    profiling_results = {}

    # Run GPU-style kernel (too many kernels, poor performance)
    run(kernel_name="gpu_style", result_paths=profiling_results)

    print("\n")

    # Run NPU-optimized kernel (fewer kernels matching physical cores, better performance)
    run(kernel_name="optimized", result_paths=profiling_results)

    # Compare results
    print("\n" + "=" * 80)
    print("Performance Comparison: GPU-style vs NPU-optimized")
    print("=" * 80)
    print("Note: GPU-style launches too many small kernels (high scheduling overhead)")
    print("      NPU-optimized launches fewer kernels matching physical cores (better)")

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results, title="Tiling Performance Comparison")
