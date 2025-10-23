# 002-vector_cmp.py说明

## 功能

## 差异点概述

## 差异点详解

Code diff of NPU and CUDA
```diff
@triton.jit
def npu_vector_cmp_kernel(
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
    NPU example shows how to use vec cmp from upper gpu original triton
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
    
-   xbar = tl.where(cols < N, X - mean, 0.0)
+   # 注释
+   cols_cmp = cols.to(tl.float32)
+   xbar = tl.where(cols_cmp < N, x - mean, 0.0)

    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # calculate Out
    mask = cols < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)

```

