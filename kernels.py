import triton
import triton.language as tl

@triton.jit
def elementwise_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * 2, mask=mask)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(a_ptr + (offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak))
    b = tl.load(b_ptr + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(BLOCK_K):
        acc += a[:, k:k+1] * b[k:k+1, :]
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), acc)