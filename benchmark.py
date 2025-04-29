import time, torch
from kernels import elementwise_kernel, matmul_kernel
import triton

def bench_elementwise(n):
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    t0 = time.time()
    elementwise_kernel[grid](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    return n*2/((time.time()-t0)*1e9)

def bench_matmul(M, N, K, cfg):
    a = torch.randn((M, K), device='cuda')
    b = torch.randn((K, N), device='cuda')
    c = torch.empty((M, N), device='cuda')
    grid = (triton.cdiv(M, cfg['BLOCK_M']), triton.cdiv(N, cfg['BLOCK_N']))
    def run():
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=cfg['BLOCK_M'], BLOCK_N=cfg['BLOCK_N'], BLOCK_K=cfg['BLOCK_K']
        )
    t0 = time.time()
    run()
    torch.cuda.synchronize()
    return 2*M*N*K/((time.time()-t0)*1e9)