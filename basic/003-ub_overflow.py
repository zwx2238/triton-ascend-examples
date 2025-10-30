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
UB Overflow
=============
Demonstrates how to avoid UB (Unified Buffer) overflow on NPU devices.

Both kernels run on NPU, but with different implementation approaches:
- GPU-style: Loads large data in one shot (may cause UB overflow or poor performance on NPU)
- NPU-optimized: Uses loop-based approach to load data in chunks (avoids UB overflow)
"""

import torch
import torch_npu
import triton
import triton.language as tl

from utils import is_npu


@triton.jit
def npu_vector_add_gpu_style_kernel(
    x,                          # [Tensor] input tensor
    y,                          # [Tensor] input tensor
    z,                          # [Tensor] output tensor
    vector_len: tl.constexpr,   # len of the vector
    BLOCK_SIZE: tl.constexpr    # size of each block (large, may cause UB overflow on NPU)
):
    """
    GPU-style implementation on NPU: Load large data blocks in one shot.
    This approach may cause UB overflow on NPU when BLOCK_SIZE is very large.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vector_len

    # Load large data in one operation (may overflow UB on NPU)
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
    BLOCK_SIZE: tl.constexpr,   # total size to process
    CHUNK_SIZE: tl.constexpr    # size of each chunk (smaller, avoids UB overflow)
):
    """
    NPU-optimized implementation: Use loop to load data in chunks to avoid UB overflow.
    This approach conforms to NPU hardware constraints.
    """
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # Calculate number of loops needed
    num_chunks = tl.cdiv(BLOCK_SIZE, CHUNK_SIZE)

    # Process data in chunks using loop
    for i in range(num_chunks):
        chunk_offset = base_offset + i * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
        len_mask = chunk_offset < vector_len

        # Load smaller chunks to avoid UB overflow
        x1 = tl.load(x + chunk_offset, mask=len_mask, other=0)
        y1 = tl.load(y + chunk_offset, mask=len_mask, other=0)
        z1 = x1 + y1
        tl.store(z + chunk_offset, z1, mask=len_mask)


def run():
    """
    Run test to demonstrate UB overflow avoidance on NPU.
    Both kernels run on NPU with different implementation styles.
    """
    device = "npu"
    vector_len = 65536   # Larger vector
    BLOCK_SIZE = 16384   # Very large block size (will cause UB overflow on NPU)
    CHUNK_SIZE = 1024    # Smaller chunk size for NPU (avoids UB overflow)
    BLOCK_DIM = 4        # Number of blocks

    x = torch.randint(0, 100, (vector_len,), device=device, dtype=torch.int32)
    y = torch.randint(0, 100, (vector_len,), device=device, dtype=torch.int32)

    # Test GPU-style kernel on NPU (should have UB overflow issues)
    print("Testing GPU-style kernel on NPU (large single load)...")
    print(f"     Attempting to load BLOCK_SIZE={BLOCK_SIZE} elements at once")
    z_gpu_style = torch.zeros((vector_len,), device=device, dtype=torch.int32)
    try:
        npu_vector_add_gpu_style_kernel[(BLOCK_DIM,)](x, y, z_gpu_style, vector_len, BLOCK_SIZE)
        torch.npu.synchronize()

        # Verify correctness
        expected = x + y
        torch.testing.assert_close(z_gpu_style, expected)
        print("==== GPU-style kernel: correctness check passed")
        print("     WARNING: This may indicate BLOCK_SIZE is still too small for UB overflow")
    except Exception as e:
        print(f"==== GPU-style kernel FAILED (UB overflow detected): {type(e).__name__}")
        print(f"     {str(e)}")

    print("\n")

    # Test NPU-optimized kernel (avoids UB overflow using loop)
    print("Testing NPU-optimized kernel (chunked load with loop)...")
    print(f"     Loading data in chunks of CHUNK_SIZE={CHUNK_SIZE}")
    z_npu_opt = torch.zeros((vector_len,), device=device, dtype=torch.int32)
    npu_vector_add_optimized_kernel[(BLOCK_DIM,)](x, y, z_npu_opt, vector_len, BLOCK_SIZE, CHUNK_SIZE)
    torch.npu.synchronize()

    # Verify correctness
    expected = x + y
    torch.testing.assert_close(z_npu_opt, expected)
    print("==== NPU-optimized kernel: correctness check passed")
    print("     Successfully avoided UB overflow by processing data in chunks")

    print("\n" + "=" * 80)
    print("Summary:")
    print("- GPU-style kernel: Loads large blocks in one shot (BLOCK_SIZE={})".format(BLOCK_SIZE))
    print("                    FAILS on NPU due to UB overflow")
    print("- NPU-optimized kernel: Loads data in chunks (CHUNK_SIZE={})".format(CHUNK_SIZE))
    print("                        Avoids UB overflow by using loop-based approach")
    print("=" * 80)


if __name__ == "__main__":
    if not is_npu():
        print("This example requires NPU device")
        exit(1)
    run()
