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
Demonstrates improving NPU resource utilization through tiling strategy.

Both kernels run on NPU with different parallelization strategies:
- GPU-style: Launches many small kernels (B * K/BLOCK_K) - High scheduling overhead
- NPU-optimized: Launches fewer larger kernels (B/BLOCK_B) with loops - Better utilization

The NPU version matches kernel count to physical cores to maximize utilization.
"""

import torch
import torch_npu
import triton
import triton.language as tl

from utils import is_npu
from prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary


@triton.jit
def npu_gather_dim1_gpu_style_kernel(
        x_ptr,  # *x  [B, C]
        idx_ptr,  # *idx[B, K]
        out_ptr,  # *out[B, K]
        stride_xb, stride_xc,
        stride_ib, stride_ik,
        stride_ob, stride_ok,
        B, K,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """
    GPU-style implementation on NPU: Launches many small kernels.
    Grid size: (B, K/BLOCK_K) - One kernel per batch and K-block.
    May have high scheduling overhead on NPU.
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k_off < K

    idx = tl.load(idx_ptr + pid_b * stride_ib + k_off * stride_ik, mask=mask)

    x_val = tl.load(x_ptr + pid_b * stride_xb + idx * stride_xc, mask=mask)

    tl.store(out_ptr + pid_b * stride_ob + k_off * stride_ok, x_val, mask=mask)


@triton.jit
def npu_gather_dim1_optimized_kernel(
        x_ptr,  # *x  [B, C]
        idx_ptr,  # *idx[B, K]
        out_ptr,  # *out[B, K]
        stride_xb, stride_xc,
        stride_ib, stride_ik,
        stride_ob, stride_ok,
        B, K,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """
    NPU-optimized implementation: Launches fewer kernels with tiling.
    Grid size: (B/BLOCK_B,) - One kernel per B-block, loops over K dimension.
    Better matches NPU physical cores for improved utilization.
    """
    pid_b = tl.program_id(0)

    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_idx < B

    # Loop over K dimension (tiling)
    for k_start in range(0, K, BLOCK_K):
        ks = tl.arange(0, BLOCK_K)
        k_mask = ks < K - k_start

        idx_off = (b_idx[:, None] * stride_ib +
                   (k_start + ks)[None, :] * stride_ik)
        col_idx = tl.load(idx_ptr + idx_off, mask=b_mask[:, None] & k_mask)

        x_off = (b_idx[:, None] * stride_xb +
                 col_idx * stride_xc)
        x_val = tl.load(x_ptr + x_off, mask=b_mask[:, None] & k_mask)

        out_off = (b_idx[:, None] * stride_ob +
                   (k_start + ks)[None, :] * stride_ok)
        tl.store(out_ptr + out_off, x_val, mask=b_mask[:, None] & k_mask)


def run(kernel_name="optimized", result_paths=None):
    """
    Run tiling test.

    Args:
        kernel_name: Kernel implementation to use ("gpu_style" or "optimized")
        result_paths: Dictionary to store profiling result paths
    """
    device = "npu"
    B = 128  # batch dim
    C = 1024  # column dim
    K = 64  # index dim

    BLOCK_B = 4
    BLOCK_K = 128

    x = torch.randn(B, C, dtype=torch.float32, device=device)
    idx = torch.randint(0, C, (B, K), dtype=torch.int64, device=device)
    out = torch.empty(B, K, dtype=x.dtype, device=device)

    # Select kernel implementation
    if kernel_name == "gpu_style":
        gather_dim1_kernel = npu_gather_dim1_gpu_style_kernel
        grid = (B, triton.cdiv(K, BLOCK_K))
        kernel_label = f"GPU-style kernel (grid={B}x{triton.cdiv(K, BLOCK_K)}={B*triton.cdiv(K, BLOCK_K)} kernels)"
    else:
        gather_dim1_kernel = npu_gather_dim1_optimized_kernel
        grid = (triton.cdiv(B, BLOCK_B),)
        kernel_label = f"NPU-optimized kernel (grid={triton.cdiv(B, BLOCK_B)} kernels with tiling)"

    # Warm up and correctness check
    gather_dim1_kernel[grid](
        x, idx, out,
        x.stride(0), x.stride(1),
        idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1),
        B, K,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
    )
    torch.npu.synchronize()

    # Verify correctness
    expected = torch.gather(x, 1, idx)
    torch.testing.assert_close(out, expected)
    print(f"==== {kernel_label} - correctness check passed")

    # Profile performance
    def kernel_wrapper():
        gather_dim1_kernel[grid](
            x, idx, out,
            x.stride(0), x.stride(1),
            idx.stride(0), idx.stride(1),
            out.stride(0), out.stride(1),
            B, K,
            BLOCK_B=BLOCK_B,
            BLOCK_K=BLOCK_K,
        )

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

    # Run GPU-style kernel (many small kernels)
    run(kernel_name="gpu_style", result_paths=profiling_results)

    print("\n")

    # Run NPU-optimized kernel (fewer kernels with tiling)
    run(kernel_name="optimized", result_paths=profiling_results)

    # Compare results
    print("\n" + "=" * 80)
    print("Performance Comparison: GPU-style vs NPU-optimized")
    print("=" * 80)
    print("Note: GPU-style launches many small kernels (high scheduling overhead)")
    print("      NPU-optimized launches fewer kernels with loops (better utilization)")

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results, title="Tiling Performance Comparison")
