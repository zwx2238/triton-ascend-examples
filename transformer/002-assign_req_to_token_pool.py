"""
Assign Request to Token Pool
=============================
Demonstrates performance optimization of token pool assignment on NPU.

Both kernels run on NPU with different implementation styles:
- GPU-style: Uses int64 and standard offset increment (works but suboptimal)
- NPU-optimized: Uses int32 and calculated offsets (better performance)

Core logic: Assigns request data to corresponding token pool by calculating KV
start/end positions, computing output cache offset from previous request lengths,
and using block-based processing to load/store data.
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
def npu_assign_req_to_token_pool_gpu_style(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """
    GPU-style implementation on NPU: Uses int64 and standard offset increment.
    Works on NPU but uses int64 operations and incremental offsets.
    Source: python/sglang/srt/speculative/spec_utils.py
    """
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0) # int64
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0) # int64
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


@triton.jit
def npu_assign_req_to_token_pool_optimized(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """
    NPU-optimized implementation: Uses int32 and calculated offsets.

    Key optimizations for NPU:
    - Uses int32 as input dtype (improves vector subtraction and sum performance)
    - Calculates offsets as 'i * BLOCK_SIZE + offset' instead of 'offset += BLOCK_SIZE'
    - Better instruction-level parallelism and performance on NPU architecture
    """
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        l_offset = i * BLOCK_SIZE + load_offset
        s_offset = i * BLOCK_SIZE + save_offset
        mask = s_offset < kv_end
        data = tl.load(out_cache_ptr + l_offset, mask=mask)
        tl.store(token_pool + s_offset, data, mask=mask)


def run(kernel_name="optimized", result_paths=None):
    """
    Run token pool assignment test on NPU.

    Args:
        kernel_name: Kernel implementation to use ("gpu_style" or "optimized")
        result_paths: Dictionary to store profiling result paths
    """
    device = "npu"
    max_batch_size = 512
    max_context_len = 8192
    batch_size = 32
    offset_len = 2

    # Select kernel and dtype
    if kernel_name == "gpu_style":
        kernel = npu_assign_req_to_token_pool_gpu_style
        dtype_name = torch.int64
        kernel_label = "GPU-style kernel (int64, offset increment)"
    else:
        kernel = npu_assign_req_to_token_pool_optimized
        dtype_name = torch.int32
        kernel_label = "NPU-optimized kernel (int32, calculated offsets)"

    # Prepare test data
    req_to_token = torch.zeros(max_batch_size, max_context_len, device=device, dtype=dtype_name)
    req_pool_indices = torch.randint(0, max_batch_size, (batch_size,), device=device, dtype=dtype_name)
    start_offset = torch.randint(0, max_context_len - offset_len, (batch_size,), device=device, dtype=dtype_name)
    end_offset = start_offset + torch.randint(1, offset_len + 1, (batch_size,), device=device, dtype=dtype_name)
    out_cache_loc = torch.randint(0, 8 * max_context_len, ((end_offset - start_offset).sum(),), device=device, dtype=dtype_name)
    pool_len = max_context_len
    bs_upper = batch_size

    # Warm up and correctness check
    req_to_token_reference = req_to_token.clone()
    kernel[(batch_size,)](req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, pool_len, bs_upper)
    torch.npu.synchronize()

    # Verify correctness by running reference implementation
    if kernel_name == "optimized":
        npu_assign_req_to_token_pool_gpu_style[(batch_size,)](
            req_pool_indices, req_to_token_reference, start_offset, end_offset, out_cache_loc, pool_len, bs_upper
        )
        torch.npu.synchronize()
        torch.testing.assert_close(req_to_token, req_to_token_reference)
        print(f"==== {kernel_label} - correctness check passed")
    else:
        print(f"==== {kernel_label} - running as reference")

    # Profile performance
    def kernel_wrapper():
        kernel[(batch_size,)](req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, pool_len, bs_upper)

    result_path = f"./result_profiling_assign_req_to_token_pool_{kernel_name}"
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
    print("Running on NPU - Comparing GPU-style vs NPU-optimized assign_req_to_token_pool")
    print("=" * 80)

    profiling_results = {}

    # Run GPU-style kernel (int64, offset increment)
    run(kernel_name="gpu_style", result_paths=profiling_results)

    print("\n")

    # Run NPU-optimized kernel (int32, calculated offsets)
    run(kernel_name="optimized", result_paths=profiling_results)

    # Compare results
    print("\n" + "=" * 80)
    print("Performance Comparison: GPU-style vs NPU-optimized")
    print("=" * 80)
    print("Note: GPU-style uses int64 with incremental offsets")
    print("      NPU-optimized uses int32 with calculated offsets for better performance")

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results, title="Token Pool Assignment Performance Comparison")