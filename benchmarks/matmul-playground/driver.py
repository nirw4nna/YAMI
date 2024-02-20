#!/usr/bin/env python3

import os

# Force Numpy to use only 1 thread, otherwise the results will not be fair
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'

from jinja2 import Environment, FileSystemLoader
from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt
import time


GCC = 'g++'
OPTIM_FLAGS = '-march=native -mtune=native -Ofast -ffp-contract=fast -funroll-loops'
FLAGS = f'-std=c++17 -Wall -Wextra -Wshadow -Wformat -Wnoexcept -Wcast-qual -Wunused -Wdouble-promotion \
-Wlogical-op -Wcast-align -fno-exceptions -fno-rtti {OPTIM_FLAGS}'

RNG = np.random.default_rng()
WARMUP = 3


def random_ndarray(n_rows: int, n_cols: int) -> np.ndarray:
    return (RNG.random((n_rows, n_cols), dtype=np.float32) + 100).astype(np.float32)


base_names = set()
all_files = set(os.listdir('kernels/'))

for file in all_files:
    if file.endswith('.h') and file.startswith('gemm_'):
        source_name = file.replace('.h', '.cpp')
        assert source_name in all_files
        base_names.add(file.replace('.h', ''))

# Compile a shared object .so for each file .c with the same base name
for obj in base_names:
    shared_obj = f'{obj}.so'
    binding = f'_{obj}.py'
    if os.path.exists(shared_obj):
        print(f'rm {shared_obj}')
        os.remove(shared_obj)
    
    if os.path.exists(binding):
        print(f'rm {binding}')
        os.remove(binding)

    cmd = f'{GCC} {FLAGS} -fPIC -shared kernels/{obj}.cpp -o {shared_obj}'
    print(cmd)
    os.system(cmd)

# Load the jinja template
env = Environment(loader=FileSystemLoader('./'))
bindings_template = env.get_template('_bindings.pytmpl')

for kernel in base_names:
    # Each template requires two things:
    #   - {{ shared_obj }} the name of the .so file (with the extension)
    #   - {{ kernel }} the name of the actual kernel to call
    data = {
        'shared_obj': f'{kernel}.so',
        'kernel': kernel
    }
    generated_code = bindings_template.render(data)
    module = f'_{kernel}'
    with open(f'{module}.py', 'wt') as f:
        f.write(generated_code)
    
    # Flush everything to disk
    os.sync()

    # Try to load the generated module
    globals()[kernel] = import_module(module)


kernel_sizes = {
    'np_matmul': [],
}
kernel_gflops = {
    'np_matmul': [],
}

print(f'Timing kernels {base_names} with Numpy as a reference. This can take some time...')
for k in range(10, 1000, 30):
    # Rows of C
    M = k
    # Cols of C
    N = k
    # Cols of A
    K = k

    # Leading dimension of A
    LDA = K
    # Leading dimension of B
    LDB = N
    # Leading dimension of C
    LDC = N
    a = random_ndarray(M, K)
    b = random_ndarray(K, N)
    
    # Add Numpy as a reference
    np_latency = float('+inf')

    for _ in range(WARMUP):
        start = time.perf_counter()
        target = np.matmul(a, b)
        cur_latency = time.perf_counter() - start
        np_latency = np_latency if cur_latency >= np_latency else cur_latency

    kernel_sizes['np_matmul'].append(k)
    kernel_gflops['np_matmul'].append((2 * M * N * K) / (1e9 * np_latency))

    for kernel in base_names:
        func = getattr(globals()[kernel], kernel)
        
        # Take the 'best' latency
        latency = float('+inf')

        for _ in range(WARMUP):
            c = np.zeros((M, N)).astype(np.float32)
            cur_latency = func(M, N, K, a, LDA, b, LDB, c, LDC)
            latency = latency if cur_latency >= latency else cur_latency
            assert np.allclose(target, c, rtol=1.e-5, atol=1.e-5, equal_nan=True)

        if kernel not in kernel_sizes:
            kernel_sizes[kernel] = []

        kernel_sizes[kernel].append(k)

        if kernel not in kernel_gflops:
            kernel_gflops[kernel] = []

        kernel_gflops[kernel].append((2 * M * N * K) / (1e9 * latency))

for kernel in kernel_sizes:
    sizes = kernel_sizes[kernel]
    gflops = kernel_gflops[kernel]
    plt.plot(sizes, gflops, '.-', label=kernel)

plt.xlabel('m = n = k')
plt.ylabel('GFLOPS')
plt.legend()
plt.show()
