# 003-ub-overflow.py说明

本示例的目的是演示如何在NPU设备上避免UB溢出。

UB是Unified Buffer的缩写，指的是Vector Core在计算过程中可以同时使用的最大缓冲区大小（通常为192 KB）。
为了避免UB OverFlow异常，必须考虑Triton内核可能同时访问的内存大小。
重点关注以下UB消费者：
1. `tl.load()` | `tl.store()`
- 每次向量化加载/存储会将`BLOCK_SIZE`个元素带入UB。
- 乘以元素大小（例如，float32为4字节，float16为2字节）。
- 如果同时进行多个加载（例如，A、B、C张量），则将它们的占用量相加。
2. 中间计算缓冲区
- 创建的任何`tl.zeros`、`tl.empty`或`tl.full`都存在于UB中。
- 任何用于`reduce`、`softmax`等的临时数组也都计入UB。
- Triton不会将数据溢出到DRAM；所有内容都保留在UB中，直到你将其存回。
3. 掩码与索引
- `bool`掩码、`arange`索引或花式索引临时变量也会在UB中分配空间。
4. Pipeline Multi-Buffering
- 如果启用了Multi-Buffering，Triton会同时保留多个缓冲区；此时需要乘以2。

## 功能
本实例是Sglang中alloc_extend_kernel的简化版，功能是为Token分配KV Slot

## 差异点概述
GPU版本与NPU版本的主要差异在于从Global Memory上搬入大量数据的区别
GPU版本可以直接搬入，NPU版本受限于UB大小，在长序列的场景下需要循环搬入

## 差异点详解

Code diff of NPU and CUDA
```diff
@triton.jit
def npu_alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        free_page_ptr,
        out_indices,
        bs_upper: tl.constexpr,
        page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
+       BLOCK_SIZE: tl.constexpr = 1024,
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

    # Part 2: fill the new full pages
    num_part2 = (
            seq_len // page_size * page_size
            - (pre_len + page_size - 1) // page_size * page_size
    )

-   offset_many_page = tl.arange(0, max_num_extend_tokens) # UB usage: max_num_extend_tokens(ex. 8192) * size0f(int64)
-   page_start = tl.load(
-       free_page_ptr + new_page_start_loc + offset_many_page // page_size,  # UB usage: max_num_extend_tokens(ex. 8192) * size0f(int64)
-       mask=offset_many_page < num_part2,  # UB usage: max_num_extend_tokens(ex. 8192) * size0f(int64)
-   )
-   tl.store(
-       out_indices + output_start_loc + offset_many_page,  # UB usage: max_num_extend_tokens(ex. 8192) * size0f(int64)
-       page_start * page_size + offset_many_page % page_size,  # UB usage: max_num_extend_tokens(ex. 8192) * size0f(int64)
-       mask=offset_many_page < num_part2,
-   )
    # Totally, the UB usage in this case is (5 * max_num_extend_tokens(ex. 8192) * size0f(int64)) approximately.

+   num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
+   blk_offset = tl.arange(0, BLOCK_SIZE)  # UB usage: BLOCK_SIZE * size0f(int64)
+   for i in range(num_loop):
+       offset_many_page = blk_offset + i * BLOCK_SIZE  # UB usage: BLOCK_SIZE * size0f(int64)
+       page_start = tl.load(
+           free_page_ptr + new_page_start_loc + offset_many_page // page_size,  # UB usage: BLOCK_SIZE * size0f(int64)
+           mask=offset_many_page < num_part2,  # UB usage: BLOCK_SIZE * size0f(int64)
+       )
+       tl.store(
+           out_indices + output_start_loc + offset_many_page,  # UB usage: BLOCK_SIZE * size0f(int64)
+           page_start * page_size + offset_many_page % page_size,  # UB usage: BLOCK_SIZE * size0f(int64)
+           mask=offset_many_page < num_part2,
+       )
    # Totally, the UB usage in this case is (6 * BLOCK_SIZE * size0f(int64)) approximately.
    # In addition, if multibuffering is enabled, the total usage will be roughly double that when it is disabled.

```

