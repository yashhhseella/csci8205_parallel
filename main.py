import argparse
from gpt_interface import get_cost_estimation
import benchmark

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--op_type', choices=['elementwise','matmul'], required=True)
    p.add_argument('--shape', type=int, nargs='+', required=True)
    p.add_argument('--dtype', default='float32')
    p.add_argument('--suggestions', type=int, default=1)
    p.add_argument('--threshold', type=float, default=100.0)
    return p.parse_args()

def main():
    args = parse()
    ir = {
        'shape': args.shape,
        'dtype': args.dtype,
        'layout': 'strided',
        'op_type': args.op_type,
        'gpu_arch': 'A100'
    }
    for i in range(1, args.suggestions+1):
        info = get_cost_estimation(ir)
        cost = info.get('predicted_cost', args.threshold+1)
        cfg  = info.get('recommendation', {})
        ok   = cost <= args.threshold
        print(f"suggestion {i}: cost={cost}, cfg={cfg}, bench={ok}")
        if ok:
            if args.op_type=='elementwise':
                perf = benchmark.bench_elementwise(args.shape[0])
            else:
                M,N,K = args.shape
                perf = benchmark.bench_matmul(M, N, K, cfg)
            print(f"measured perf: {perf} TFLOPS")
        else:
            print("skipped bench")
        print()

if __name__=='__main__':
    main()