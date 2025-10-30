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
Load Order
===========
Demonstrates how load order affects performance by managing data dependencies.

In loops with data dependencies, the order of load operations matters:
- BA order: Load B then load A - Load B waits for previous store B, blocking load A
- AB order: Load A then load B - Load A executes in parallel with previous store B (better)

Both implementations run on NPU, showing how instruction ordering impacts parallelism.
"""

import torch
import torch_npu
import triton
import triton.language as tl

from utils import is_npu
from prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary


@triton.jit
def BA_load_kernel(
    A,
    B,
    B_index,
    O,
    B_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Load B first, then load A.
    Load B waits for the previous loop's store B, blocking load A execution.
    This reduces parallelism and hurts performance.
    """
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        # calc index
        p_A = A + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index + i

        # load B (blocked by previous store B)
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        # load A (blocked by load B)
        b_A = tl.load(p_A)

        # calculation
        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        # store O
        tl.store(p_O, b_O)

        # store B (creates dependency for next iteration)
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)


@triton.jit
def AB_load_kernel(
    A,
    B,
    B_index,
    O,
    B_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Load A first, then load B.
    Load A has no dependency, so it executes in parallel with previous store B.
    This improves parallelism and performance.
    """
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        # calc index
        p_A = A + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index + i

        # load A (no dependency, can run in parallel with previous store B)
        b_A = tl.load(p_A)

        # load B (still waits for previous store B, but doesn't block A)
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        # calculation
        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        # store O
        tl.store(p_O, b_O)

        # store B (creates dependency for next iteration)
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)


def load_order_example(
    A,
    B,
    B_index,
    O,
    mode: str
):
    """Execute the kernel with specified load order."""
    HEAD_NUM, HEAD_DIM = A.shape
    B_DIM = 8
    N_DIM = HEAD_DIM // B_DIM

    grid = (N_DIM, )
    if mode == "BA":
        BA_load_kernel[grid](
            A,
            B,
            B_index,
            O,
            B_DIM,
            HEAD_NUM,
            HEAD_DIM,
        )
    else:
        AB_load_kernel[grid](
            A,
            B,
            B_index,
            O,
            B_DIM,
            HEAD_NUM,
            HEAD_DIM,
        )


def run(mode="AB", result_paths=None):
    """
    Run load order test.

    Args:
        mode: Load order mode ("BA" or "AB")
        result_paths: Dictionary to store profiling result paths
    """
    device = "npu"
    HEAD_NUM = 4
    HEAD_DIM = 32

    A = torch.arange(HEAD_DIM, dtype=torch.float32, device=device)
    A = A.repeat(HEAD_NUM, 1).contiguous()

    B = torch.arange(HEAD_NUM, dtype=torch.float32, device=device)
    B_index = torch.arange(HEAD_NUM, dtype=torch.int32, device=device)

    O = torch.empty_like(A)

    if mode == "BA":
        kernel_label = "BA load order (load B first, then A)"
    else:
        kernel_label = "AB load order (load A first, then B)"

    # Warm up and correctness check
    load_order_example(A, B, B_index, O, mode)
    torch.npu.synchronize()
    print(f"==== {kernel_label} - execution completed")

    # Profile performance
    def kernel_wrapper():
        load_order_example(A, B, B_index, O, mode)

    result_path = f"./result_profiling_load_order_{mode}"
    print(f"==== Profiling {kernel_label}...")
    profiler_wrapper(kernel_wrapper, result_path=result_path)

    # Store result path for comparison
    if result_paths is not None:
        result_paths[mode] = result_path


if __name__ == "__main__":
    if not is_npu():
        print("This example requires NPU device")
        exit(1)

    # Compare both load orders on NPU
    print("=" * 80)
    print("Running on NPU - Comparing BA vs AB load order")
    print("=" * 80)

    profiling_results = {}

    # Run BA load order (suboptimal)
    run(mode="BA", result_paths=profiling_results)

    print("\n")

    # Run AB load order (optimized)
    run(mode="AB", result_paths=profiling_results)

    # Compare results
    print("\n" + "=" * 80)
    print("Performance Comparison: BA load order vs AB load order")
    print("=" * 80)
    print("Note: BA order loads B first (blocked by previous store B), then loads A")
    print("      AB order loads A first (no dependency, runs in parallel), then loads B")
    print("      AB order should be faster due to better instruction-level parallelism")

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results, title="Load Order Performance Comparison")
