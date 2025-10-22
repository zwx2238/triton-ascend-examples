"""
This Triton kernel is responsible for allocating physical memory slots (pages) in a paged KV-cache
for newly extended tokens in variable-length sequences.

The newly added tokens are divided into three parts:
Part 1: Fill the remaining slots in the last partially filled page from the previous step.
Part 2: Allocate full new pages if there are many new tokens.
Part 3: Handle the final partial page if needed.

The difference between the gpu and npu version is part 2 implementation.
The NPU version uses a loop-based approach in Part 2 to conform to hardware constraints,
while the GPU version uses full vectorized parallelism for maximum performance.
"""

import numpy as np
import torch

import triton
import triton.language as tl

from utils import is_npu, next_power_of_2

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    device = torch.device('npu')
else:
    device = torch.device('cuda')


@triton.jit
def gpu_alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        ret_values,
        bs_upper: tl.constexpr,
        page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
            pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Return value
    if pid == tl.num_programs(0) - 1:
        merged_value = (sum_num_new_pages.to(tl.int64)) << 32 | sum_extend_lens.to(
            tl.int64
        )
        tl.store(ret_values, merged_value)

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
            min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
            seq_len // page_size * page_size
            - (pre_len + page_size - 1) // page_size * page_size
    )

    offset_many_page = tl.arange(0, max_num_extend_tokens)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,
    )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


@triton.jit
def npu_alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        ret_values,
        bs_upper: tl.constexpr,
        page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
        BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
            pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Return value
    if pid == tl.num_programs(0) - 1:
        merged_value = (sum_num_new_pages.to(tl.int64)) << 32 | sum_extend_lens.to(
            tl.int64
        )
        tl.store(ret_values, merged_value)

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
            min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
            seq_len // page_size * page_size
            - (pre_len + page_size - 1) // page_size * page_size
    )

    num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    blk_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(num_loop):
        offset_many_page = blk_offset + i * BLOCK_SIZE
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset_many_page // page_size,
            mask=offset_many_page < num_part2,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset_many_page,
            page_start * page_size + offset_many_page % page_size,
            mask=offset_many_page < num_part2,
        )

    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


def _gen_test_date(batch_size: int,
                   max_context_len: int,
                   max_free_page: int,
                   is_prefill: bool,
                   page_size: int = 128,
                   speculative_num_steps: int = 1):
    free_pages = torch.arange(max_free_page, dtype=torch.int32, device=device)

    if is_prefill:
        prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)  # shape:(batch)

        seq_lens = torch.tensor([873, 1500], dtype=torch.int32, device=device)

        last_loc = torch.full(prefix_lens.shape, -1, dtype=torch.int32, device=device)

        extend_num_tokens = torch.sum(seq_lens).item()
    else:
        prefix_lens = torch.from_numpy(
            np.random.choice(range(max_context_len), size=batch_size, replace=False)
        ).to(torch.int32).to(device=device)

        seq_lens = prefix_lens + (speculative_num_steps + 1)  # shape:(batch)

        allocated_pages = torch.ceil(prefix_lens / page_size).to(torch.int32)
        allocated_pages = torch.sum(allocated_pages).item()
        free_pages = free_pages[allocated_pages + 1:]

        last_loc = torch.from_numpy(
            np.random.choice(range(allocated_pages), size=batch_size, replace=False)
        ).to(torch.int32).to(device)  # shape:(batch)
        last_loc = last_loc * page_size + prefix_lens % page_size

        extend_num_tokens = (speculative_num_steps + 1) * batch_size

    out_indices = torch.empty(
        (extend_num_tokens,), dtype=torch.int64, device=device
    )
    ret_values = torch.empty((), dtype=torch.int64, device=device)

    max_num_extend_tokens = next_power_of_2(extend_num_tokens)

    return prefix_lens, seq_lens, last_loc, free_pages, out_indices, ret_values, max_num_extend_tokens


def run(device_name="npu"):
    kwargs = {}
    if device_name == "npu":
        alloc_extend_kernel = npu_alloc_extend_kernel
        kwargs = {
            'BLOCK_SIZE': 1024
        }
    else:
        alloc_extend_kernel = gpu_alloc_extend_kernel

    batch_size = 2
    max_context_len = 2000
    max_free_page = 2048
    is_prefill = True

    (
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        ret_values,
        max_num_extend_tokens
    ) = _gen_test_date(batch_size, max_context_len, max_free_page, is_prefill)

    print(f"{seq_lens=}", flush=True)

    bs = prefix_lens.shape[0]
    alloc_extend_kernel[(bs,)](
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        ret_values,
        next_power_of_2(bs),
        page_size=128,
        max_num_extend_tokens=max_num_extend_tokens,
        **kwargs
    )

    print(f'{ret_values=}\n{out_indices=}', flush=True)


if __name__ == "__main__":
    run("npu" if _is_npu else "cuda")
