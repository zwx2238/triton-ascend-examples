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

from prof_util import profiler_wrapper


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


def run(dtype_name):
    vector_len = 16384
    BLOCK_SIZE = 512
    BLOCK_DIM = 32
    device_name = "npu"

    x = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    y = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    z = torch.zeros((1, vector_len), device=device_name, dtype=dtype_name)

    # Test correctness first
    npu_vector_add_kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)
    torch.npu.synchronize()

    # Verify correctness
    expected = x + y
    torch.testing.assert_close(z, expected)
    print(f"==== {dtype_name} correctness check passed")

    # Profile performance
    def kernel_wrapper():
        npu_vector_add_kernel[(BLOCK_DIM,)](x, y, z, vector_len, BLOCK_SIZE)

    print(f"==== Profiling {dtype_name} vector add kernel...")
    profiler_wrapper(kernel_wrapper, result_path=f"./result_profiling_{dtype_name}")

if __name__ == "__main__":
    run(torch.int64)
    run(torch.int32) # prefer using int32 dtype

