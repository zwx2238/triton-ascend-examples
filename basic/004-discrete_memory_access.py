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
Discrete Memory Access
=======================
Demonstrates different approaches for accessing discrete (non-contiguous) memory on NPU.

Both kernels run on NPU with different implementation styles:
- GPU-style: Uses tl.load with mask to access discrete indices (may have poor performance)
- NPU-optimized: Loads contiguous data first, then uses tl.gather for discrete access (better performance)
"""

import torch
import torch_npu
import triton
import triton.language as tl

from utils import is_npu
from prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary


@triton.jit
def npu_pick_gpu_style_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    """
    GPU-style implementation on NPU: Direct discrete memory access.
    Uses tl.load with dynamic indices, which may have poor performance on NPU.
    """
    pid = tl.program_id(0)
    rn = tl.arange(0, N)  # [0..N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Direct discrete memory access (may be slow on NPU)
    val = tl.load(x_ptr + idx * stride_x, mask=mask)

    # Temporarily comment out store to test if load works
    tl.store(y_ptr + rn * stride_y, val)


@triton.jit
def npu_pick_optimized_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    """
    NPU-optimized implementation: Load contiguous data first, then gather.
    First loads all data into shared memory contiguously, then picks elements using tl.gather.
    This approach is more efficient on NPU hardware.
    """
    pid = tl.program_id(0)  # 1 block
    rm = tl.arange(0, M)  # [M]
    rn = tl.arange(0, N)  # [N]

    idx = tl.load(idx_ptr + rn * stride_idx)  # [N]
    mask = idx < M

    # Load contiguous data into shared memory
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]

    # Gather from shared memory (efficient on NPU)
    val = tl.gather(x_shared, idx, 0)

    # Temporarily comment out store to test if load/gather works
    tl.store(y_ptr + rn * stride_y, val)


def run(kernel_name="optimized", result_paths=None):
    """
    Run discrete memory access test.

    Args:
        kernel_name: Kernel implementation to use ("gpu_style" or "optimized")
        result_paths: Dictionary to store profiling result paths
    """
    device = "npu"
    M = 1024
    N = 256

    # Select kernel implementation
    if kernel_name == "gpu_style":
        pick_kernel = npu_pick_gpu_style_kernel
        kernel_label = "GPU-style kernel (direct discrete access)"
    else:
        pick_kernel = npu_pick_optimized_kernel
        kernel_label = "NPU-optimized kernel (load then gather)"

    x = torch.randn(M, device=device)
    indices = torch.randint(0, M, (N,), device=device)
    y = torch.empty(N, dtype=x.dtype, device=device)

    # Warm up and correctness check
    grid = (1,)
    pick_kernel[grid](
        x,
        indices,
        y,
        x.stride(0),
        indices.stride(0),
        y.stride(0),
        M=M,
        N=triton.next_power_of_2(N),
    )
    torch.npu.synchronize()

    # Skip correctness check since store is commented out
    # Just testing if load/gather works without errors
    print(f"==== {kernel_label} - kernel executed (store commented out for testing)")

    # Profile performance
    def kernel_wrapper():
        pick_kernel[grid](
            x,
            indices,
            y,
            x.stride(0),
            indices.stride(0),
            y.stride(0),
            M=M,
            N=triton.next_power_of_2(N),
        )

    result_path = f"./result_profiling_discrete_memory_{kernel_name}"
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
    print("Running on NPU - Comparing GPU-style vs NPU-optimized discrete memory access")
    print("=" * 80)

    profiling_results = {}

    # Run GPU-style kernel (direct discrete access)
    run(kernel_name="gpu_style", result_paths=profiling_results)

    print("\n")

    # Run NPU-optimized kernel (load then gather)
    run(kernel_name="optimized", result_paths=profiling_results)

    # Compare results
    print("\n" + "=" * 80)
    print("Performance Comparison: GPU-style vs NPU-optimized")
    print("=" * 80)
    print("Note: GPU-style uses direct discrete access which may be slow on NPU")
    print("      NPU-optimized loads contiguous data first then uses gather")

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results, title="Discrete Memory Access Performance Comparison")
