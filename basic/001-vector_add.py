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
Vector Add
=============
"""
import os

import torch
import torch_npu
import triton
import triton.language as tl

from prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary


@triton.jit
def npu_vector_add_kernel(
    x,                          # [Tensor] input tensor (1 x col)     
    y,                          # [Tensor] input tensor (1 x col)
    z,                          # [Tensor] output tensor (1 x col)
    vector_len: tl.constexpr,   # len of the vector
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vector_len
    x1 = tl.load(x + offset, mask=len_mask, other=0)
    y1 = tl.load(y + offset, mask=len_mask, other=0)
    z1 = x1 + y1
    tl.store(z + offset, z1, mask=len_mask)


def run(dtype_name, result_paths):
    """
    Run vector add kernel test for a specific data type.

    Args:
        dtype_name: PyTorch data type (e.g., torch.int32, torch.int64)
        result_paths: Dictionary to store profiling result paths
    """
    vector_len = 16384
    BLOCK_SIZE = 512
    BLOCK_DIM = 32
    device_name = "npu"

    # Get dtype name string for labeling
    dtype_str = str(dtype_name).split('.')[-1]

    x = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    y = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    z = torch.zeros((1, vector_len), device=device_name, dtype=dtype_name)

    # Test correctness first
    npu_vector_add_kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)
    torch.npu.synchronize()

    # Verify correctness
    expected = x + y
    torch.testing.assert_close(z, expected)
    print(f"==== {dtype_str} correctness check passed")

    # Profile performance
    def kernel_wrapper():
        npu_vector_add_kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)

    result_path = f"./result_profiling_{dtype_str}"
    print(f"==== Profiling {dtype_str} vector add kernel...")
    profiler_wrapper(kernel_wrapper, result_path=result_path)

    # Store result path for later comparison
    result_paths[dtype_str] = result_path

if __name__ == "__main__":
    # Dictionary to store profiling result paths
    profiling_results = {}

    # Run tests for different data types
    run(torch.int64, profiling_results)
    run(torch.int32, profiling_results)  # prefer using int32 dtype

    # Compare and report profiling results
    print("\n" + "=" * 80)
    print("Comparing profiling results across different data types...")
    print("=" * 80)

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results, title="Vector Add Kernel Performance Comparison")

