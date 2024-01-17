import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

import sys
import os

sys.path.append(f'{os.path.join(os.path.dirname(__file__), "pyyami")}')
from pyyami import *

# CSV files are in the form:
# Kernel,Shape_a,Shape_b,latency
#
# The idea for this tool is to:
# - load a CSV and show the top 10 worst timings (both in absolute and relative terms)
# - plot if required (print to console by default)
# - run the same kernel via pyyami and compare it with numpy and pytorch
#
# We strive to be at most 2X slower than NP which is faster than Pytorch for small matrices,
# comparable for medium-sized matrices and a bit slower with huge matrices.
# Once we can actually pass this tests we can then add it to the GitHub CI pipeline to check whether
# we can accept a new commit on the main branch!

CSV_FILE = 'matmul.csv'
PLOT = False
SHOW_TOP = 10
COLD_START = 2
STEPS = 10


def _plot(delays, delay_per_element, delay_pyyami, delay_np, labels):
    _, (ax) = plt.subplots(2, 2, figsize=(15, 5), tight_layout=True)

    ax[0, 0].bar(labels, delays, label='GEMM Delay', color='green')
    ax[0, 0].set_xlabel('Shape')
    ax[0, 0].set_ylabel('Delay (ms)')
    ax[0, 0].legend()

    ax[0, 1].bar(labels, delay_per_element, label='GEMM Delay / Size', color='red')
    ax[0, 1].set_xlabel('Shape')
    ax[0, 1].set_ylabel('Delay (us)')
    ax[0, 1].legend()

    ax[1, 0].bar(labels, delay_pyyami, label='GEMM Delay PyYAMI', color='blue')
    ax[1, 0].set_xlabel('Shape')
    ax[1, 0].set_ylabel('Delay (us)')
    ax[1, 0].legend()

    ax[1, 1].bar(labels, delay_np, label='GEMM Delay NumPy', color='orange')
    ax[1, 1].set_xlabel('Shape')
    ax[1, 1].set_ylabel('Delay (us)')
    ax[1, 1].legend()

    ax[0, 0].tick_params(axis='x', rotation=50)
    ax[0, 1].tick_params(axis='x', rotation=50)
    ax[1, 0].tick_params(axis='x', rotation=50)
    ax[1, 1].tick_params(axis='x', rotation=50)

    plt.show()


def _shape_arr(shape: str) -> list[int]:
    # Take as input a shape in the form [AxBxCxD] and return the number of elements
    shape = shape.replace("[", "")
    shape = shape.replace("]", "")
    return [int(x) for x in shape.split('x')]


def _bench(func, *args) -> float:
    for _ in range(COLD_START):
        func(*args)

    start = time.perf_counter_ns()

    for _ in range(STEPS):
        func(*args)

    stop = time.perf_counter_ns()

    return round(((stop - start) / 1e6) / STEPS, ndigits=3)


def _print_timings(df, ctx, worst=True):
    df = df.sort_values(by='delay', ascending=False)
    df = df.drop_duplicates(subset=['shape_a', 'shape_b'], keep='first' if worst else 'last')

    df = df.head(SHOW_TOP)
    shapes_a = df['shape_a']
    shapes_b = df['shape_b']
    labels = ['\n@\n'.join([s1, s2]) for s1, s2 in zip(shapes_a, shapes_b)]
    delays = df['delay']
    delay_per_element = [round((d / (np.prod(_shape_arr(s1)) + np.prod(_shape_arr(s2)))) * 1e6, ndigits=3)
                         for d, s1, s2 in zip(df['delay'],
                                              df['shape_a'],
                                              df['shape_b'])
                         ]

    data = []
    delays_np = []
    delays_pyyami = []
    for d, d_rel, s1, s2 in zip(delays, delay_per_element, shapes_a, shapes_b):
        s1_shape = _shape_arr(s1)
        s2_shape = _shape_arr(s2)
        # Assume this is a GEMM
        a = torch.rand(s1_shape)
        b = torch.rand(s2_shape)
        a_np = a.numpy()
        b_np = b.numpy()

        d_np = _bench(np.matmul, a_np, b_np)
        d_yami = _bench(yami_matmul, ctx,
                        YamiTensor.from_np(ctx, "a", a_np),
                        YamiTensor.from_np(ctx, "b", b_np)
                        )
        delays_np.append(d_np)
        delays_pyyami.append(d_yami)
        ctx.clear()

        data.append([s1_shape, s2_shape, d, d_rel, d_np, d_yami])

    frame = pd.DataFrame(data, columns=['Input A', 'Input B', 'Delay (ms)', 'Delay / element (ns)',
                                        'Delay NumPy (ms)', 'Delay pyyami (ms)'])
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 1000)
    print(f'\n{"Worst" if worst else "Best"} {SHOW_TOP} timings:')
    print(frame)

    if PLOT:
        _plot(delays, delay_per_element, delays_pyyami, delays_np, labels)


if __name__ == '__main__':
    data_frame = pd.read_csv(CSV_FILE, header=None, names=['kernel', 'shape_a', 'shape_b', 'delay'])
    yami_ctx = YamiContext(1024*1024*1024, 12)
    _print_timings(data_frame, yami_ctx)
    _print_timings(data_frame, yami_ctx, worst=False)
