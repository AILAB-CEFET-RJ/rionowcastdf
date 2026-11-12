"""Real CUDA Warp kernel smoke test for the reference environment doctor.

Checks compilation, launch and synchronization on each selected GPU. This
script reads no dataset and produces no training checkpoints.
"""
import argparse
import json
import sys
import numpy as np
import warp as wp


@wp.kernel
def increment(a: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a[i] = a[i] + 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0')
    args = parser.parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    if len(set(gpu_ids)) != len(gpu_ids) or any(x < 0 for x in gpu_ids):
        raise ValueError('Invalid GPU indices')
    wp.init()
    details = []
    for gpu in gpu_ids:
        device = f'cuda:{gpu}'
        x = wp.array(np.array([1, 2, 3, 4], dtype=np.float32),
                     dtype=wp.float32, device=device)
        wp.launch(increment, dim=4, inputs=[x], device=device)
        wp.synchronize_device(device)
        result = x.numpy()
        if not np.array_equal(result, np.array([2, 3, 4, 5], dtype=np.float32)):
            raise RuntimeError(f'Incorrect Warp output on {device}: {result}')
        details.append({'device': device, 'kernel_passed': True})
    print(json.dumps({'passed': True, 'devices': details}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
