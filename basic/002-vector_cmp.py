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
Vector Cmp
1. Case study on performance optimization of Triton operators on NPU, compared GPU-style vs NPU-optimized.
2. Ascend does not support Cmp on i32/i64, which leads to vector compute fallback to scalar
3. Both kernels run on NPU with different implementation styles
"""
import os

import torch
import torch_npu
import triton
import triton.language as tl

from prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary



def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@triton.jit
def npu_vector_cmp_gpu_style_kernel(
    X,                 # [Tensor] input tensor (row x col)
    Out,               # [Tensor] output tensor (row x col)
    Mean,              # [Vector] mean tensor (row, ) of X
    Rstd,              # [Vector] std tensor (row, ) of X
    stride_x_row,      # [Scalar] stride of row of x
    stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
    M,                 # [Scalar] row number
    N,                 # [Scalar] col number
    eps,               # [Scalar] epsilon to aviod division by zeros
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    GPU-style layernorm implementation on NPU.
    Uses i32/i64 comparison which may fallback to scalar operations on NPU.

    Out = ((X - E[X]) / sqrt(V[X] + eps)) on dim -1

    Assumptions:
    1. BLOCK_N >= X.shape(-1), group_n = 0 only
    2. BLOCK_M = 1, group_m = range(0, row, 1)
    """
    group_m = tl.program_id(0)
    group_n = tl.program_id(1)
    row = group_m

    # calculate index & offset
    Mean = Mean + group_n * M
    Rstd = Rstd + group_n * M
    X = X + row * stride_x_row + group_n * N
    Out = Out + row * stride_out_row + group_n * N

    cols = tl.arange(0, BLOCK_N)  # cols is int64
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    # calculate mean & rstd
    mean = tl.sum(x, axis=0) / N
    tl.store(Mean + row, mean)
    xbar = tl.where(cols < N, x - mean, 0.0)  # i64 cmp may fallback to scalar on NPU
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # calculate Out
    mask = cols < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)


@triton.jit
def npu_vector_cmp_optimized_kernel(
    X,                 # [Tensor] input tensor (row x col)
    Out,               # [Tensor] output tensor (row x col)
    Mean,              # [Vector] mean tensor (row, ) of X
    Rstd,              # [Vector] std tensor (row, ) of X
    stride_x_row,      # [Scalar] stride of row of x
    stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
    M,                 # [Scalar] row number
    N,                 # [Scalar] col number
    eps,               # [Scalar] epsilon to aviod division by zeros
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    NPU-optimized layernorm implementation.
    Converts i64 to float32 before comparison to avoid scalar fallback on NPU.
    """
    group_m = tl.program_id(0)
    group_n = tl.program_id(1)
    row = group_m

    # calculate index & offset
    Mean = Mean + group_n * M
    Rstd = Rstd + group_n * M
    X = X + row * stride_x_row + group_n * N
    Out = Out + row * stride_out_row + group_n * N

    cols = tl.arange(0, BLOCK_N)  # cols is int64
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    # calculate mean & rstd
    mean = tl.sum(x, axis=0) / N
    tl.store(Mean + row, mean)
    # [Optimized for NPU]
    # Convert i64 to float32 before comparison to avoid scalar fallback
    cols_cmp = cols.to(tl.float32)
    xbar = tl.where(cols_cmp < N, x - mean, 0.0)
    # [End optimization]

    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # calculate Out
    mask = cols < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)


def run(device="cuda", kernel_name="optimized", result_paths=None):
    """
    Run random test along with native norm.

    Args:
        device: Device to run on ("cuda" or "npu")
        kernel_name: Kernel implementation to use ("gpu_style" or "optimized")
        result_paths: Dictionary to store profiling result paths
    """
    def run_ref(X):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        out =  (X.to(torch.float32) - mean) / (std + eps)
        return out.to(device_dtype)

    batch_size = 256
    feature_dim = 128
    eps = 1e-6

    # Select kernel implementation
    if kernel_name == "gpu_style":
        vector_cmp_kernel = npu_vector_cmp_gpu_style_kernel
        kernel_label = "GPU-style kernel (may fallback to scalar on NPU)"
    else:
        vector_cmp_kernel = npu_vector_cmp_optimized_kernel
        kernel_label = "NPU-optimized kernel"

    if device == "npu":
        sync_device = torch.npu
        device_dtype = torch.float32
    else:
        sync_device = torch.cuda
        device_dtype = torch.float32

    X = torch.rand(batch_size, feature_dim, device=device, dtype=device_dtype)
    Out = torch.empty_like(X)
    Mean = torch.empty(batch_size, device=device, dtype=device_dtype)
    Rstd = torch.empty(batch_size, device=device, dtype=device_dtype)

    BLK_M = 1
    BLK_N = triton.next_power_of_2(feature_dim)
    num_warps = min(max(BLK_N // 256, 1), 8)

    # Warm up and correctness check
    vector_cmp_kernel[(batch_size // BLK_M, 1)](
        X, Out, Mean, Rstd, X.stride(0), Out.stride(0), batch_size, feature_dim, eps, BLK_M, BLK_N, num_warps=num_warps)
    sync_device.synchronize()

    Out_ref = run_ref(X)
    torch.testing.assert_close(Out, Out_ref, rtol=1e-3, atol=1e-2)
    print(f"==== {device} {kernel_label} - correctness check passed")

    # Profile performance
    def kernel_wrapper():
        vector_cmp_kernel[(batch_size // BLK_M, 1)](
            X, Out, Mean, Rstd, X.stride(0), Out.stride(0), batch_size, feature_dim, eps, BLK_M, BLK_N, num_warps=num_warps)

    result_path = f"./result_profiling_vector_cmp_{device}_{kernel_name}"
    print(f"==== Profiling {device} {kernel_label}...")
    profiler_wrapper(kernel_wrapper, result_path=result_path)

    # Store result path for comparison
    if result_paths is not None:
        result_paths[f"{device}_{kernel_name}"] = result_path


if __name__ == "__main__":
    if is_npu():
        # On NPU, compare both GPU-style and NPU-optimized kernel implementations
        print("=" * 80)
        print("Running on NPU - Comparing GPU-style vs NPU-optimized kernels")
        print("=" * 80)

        profiling_results = {}

        # Run GPU-style kernel (may have performance issues due to i32/i64 cmp fallback)
        run("npu", kernel_name="gpu_style", result_paths=profiling_results)

        print("\n")

        # Run NPU-optimized kernel
        run("npu", kernel_name="optimized", result_paths=profiling_results)

        # Compare results
        print("\n" + "=" * 80)
        print("Performance Comparison: GPU-style vs NPU-optimized")
        print("=" * 80)
        print("Note: GPU-style kernel may fallback to scalar operations on NPU due to i32/i64 cmp")

        results = compare_profiling_results(profiling_results)
        print_profiling_summary(results, title="Vector Cmp Kernel Performance Comparison")
    else:
        # On CUDA, only run GPU-style kernel
        run("cuda", kernel_name="gpu_style")

