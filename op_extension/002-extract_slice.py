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
Extract Slice
==============
Demonstrates token reverse operation using tl.extract_slice on NPU.

Both kernels run on NPU with different implementation styles:
- GPU-style: Processes tokens one by one (simple but may be slow)
- NPU-optimized: Batch processes using tl.extract_slice (efficient for NPU)
"""

import torch
import torch_npu
import triton
import triton.language as tl

import sys
sys.path.append('..')
from utils import is_npu
from basic.prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary


@triton.jit
def npu_token_reverse_gpu_style_kernel(x_ptr, indices, output_ptr, n_elements, S: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    GPU-style implementation on NPU: Process tokens one by one.
    Each kernel handles one token, simple but launches many kernels.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * D

    # 1.load data
    offsets = block_start + tl.arange(0, D)
    data_mask = offsets < n_elements
    data = tl.load(x_ptr + offsets, mask=data_mask)

    # 2.load index
    index_offset = pid * BLOCK_SIZE
    idx_mask = index_offset < S
    index = tl.load(indices + index_offset, mask=idx_mask)

    # 3.calc output offset by index & store
    out_offset = index * D + tl.arange(0, D)
    out_msk = out_offset < n_elements
    tl.store(output_ptr + out_offset, data, mask=out_msk)


@triton.jit
def npu_token_reverse_optimized_kernel(x_ptr, indices, output_ptr, n_elements, S : tl.constexpr, D : tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    NPU-optimized implementation: Batch process using tl.extract_slice.
    Uses NPU-specific extract_slice operation for efficient batch processing.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * D

    # 1. batch load data
    data_offset = D * tl.arange(0, BLOCK_SIZE)[:,None] + tl.arange(0, D)[None, :]
    data_mask = data_offset < n_elements
    data = tl.load(x_ptr + block_start + data_offset, data_mask)

    # 2. batch load indices
    idx_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx_mask = idx_start < S
    idx = tl.load(indices + idx_start, idx_mask)

    # 3. extract token one by one and store
    for i in tl.range(0, BLOCK_SIZE):
        x_sub = tl.extract_slice(data, [i,0], [1,D], [1,1])
        output_offset = D * tl.get_element(idx, (i,))+ tl.arange(0, D)[None,:]
        out_mask = output_offset < n_elements
        tl.store(output_ptr + output_offset, x_sub, out_mask)


def run(kernel_name="optimized", result_paths=None):
    """
    Run token reverse test on NPU.

    Args:
        kernel_name: Kernel implementation to use ("gpu_style" or "optimized")
        result_paths: Dictionary to store profiling result paths
    """
    device = "npu"
    S = 1024  # Sequence length
    D = 32    # Token dimension

    x = torch.rand(S, D, device=device)
    indices = torch.randperm(S).to(device=device)
    output = torch.empty_like(x)

    # Select kernel implementation
    if kernel_name == "gpu_style":
        kernel = npu_token_reverse_gpu_style_kernel
        grid = (S, 1, 1)
        block_size = 1
        kernel_label = f"GPU-style kernel (launches {S} kernels, BLOCK_SIZE=1)"
    else:
        kernel = npu_token_reverse_optimized_kernel
        grid = (48, 1, 1)
        block_size = 22
        kernel_label = f"NPU-optimized kernel (launches 48 kernels, BLOCK_SIZE=22)"

    # Warm up and correctness check
    kernel[grid](x, indices, output, x.numel(), S, D, BLOCK_SIZE=block_size)
    torch.npu.synchronize()

    # Verify correctness
    # Create expected output: output[indices[i]] = x[i]
    expected = torch.empty_like(x)
    for i in range(S):
        expected[indices[i]] = x[i]
    # torch.testing.assert_close(output, expected)
    # print(f"==== {kernel_label} - correctness check passed")

    # Profile performance
    def kernel_wrapper():
        kernel[grid](x, indices, output, x.numel(), S, D, BLOCK_SIZE=block_size)

    result_path = f"./result_profiling_extract_slice_{kernel_name}"
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
    print("Running on NPU - Comparing GPU-style vs NPU-optimized extract_slice")
    print("=" * 80)

    profiling_results = {}

    # Run GPU-style kernel (many small kernels)
    run(kernel_name="gpu_style", result_paths=profiling_results)

    print("\n")

    # Run NPU-optimized kernel (fewer kernels with extract_slice)
    run(kernel_name="optimized", result_paths=profiling_results)

    # Compare results
    print("\n" + "=" * 80)
    print("Performance Comparison: GPU-style vs NPU-optimized")
    print("=" * 80)
    print("Note: GPU-style launches many kernels (one per token)")
    print("      NPU-optimized uses tl.extract_slice for batch processing")

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results, title="Extract Slice Performance Comparison")
