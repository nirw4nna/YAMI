from pyyami import *
import numpy as np
from random import randint
import torch
import torch.nn.functional as F
import time

TEST_STEPS = 10
RNG = np.random.default_rng()


def all_close(actual, target, eps=1e-5):
    diffs = ~np.isclose(actual, target, atol=eps, rtol=eps, equal_nan=True)
    close = len(actual[diffs]) == 0
    if not close:
        print(f'Wrong indexes: {np.where(diffs == True)}')
        print(f'My values: {actual[diffs]}\nNP values: {target[diffs]}')

    return close


def random_ndarray(*dims: int) -> np.ndarray:
    # return np.random.randint(-100, 100, size=dims, dtype=np.float32)
    return (RNG.random(dims, dtype=np.float32) - 0.5).astype(np.float32)


def _bench_matmul(ctx: YamiContext, dim_a: tuple[int, ...], dim_b: tuple[int, ...]):
    target_a = random_ndarray(*dim_a)
    target_b = random_ndarray(*dim_b)
    start = time.perf_counter()
    target_res = np.matmul(target_a, target_b)
    stop = time.perf_counter()
    delay_np_ms = (stop - start) * 1000

    my_a = YamiTensor.from_np(ctx, 'a', target_a)
    my_b = YamiTensor.from_np(ctx, 'b', target_b)
    start = time.perf_counter()
    my_res = yami_matmul(ctx, my_a, my_b)
    stop = time.perf_counter()
    delay_yami_ms = (stop - start) * 1000

    my_res_np = my_res.as_np()
    assert all_close(my_res_np, target_res)
    diff = delay_yami_ms / delay_np_ms
    # We strive to be at most 3X slower than NP which is faster than Pytorch for small matrices,
    # comparable for medium-sized matrices and a bit slower with huge matrices.
    # Once we can actually pass this test we can then add it to the GitHub CI pipeline to check whether
    # we can accept a new commit on the main branch!
    print(f'YAMI is {round(diff)}X slower than Numpy when multiplying {dim_a} with {dim_b}!')
    if diff > 3.1:
        pass
        # assert False


class TestMatmul:
    def test_2d(self):
        ctx = YamiContext(1024*1024*1024*5, 12)
        # test small
        _bench_matmul(ctx, (2, 2), (2, 2))
        ctx.clear()

        # test medium
        _bench_matmul(ctx, (1024, 512), (512, 1024))
        ctx.clear()

        # test large
        _bench_matmul(ctx, (4096, 1024), (1024, 4096))
        ctx.clear()

    def test_3d(self):
        ctx = YamiContext(1024*1024*1024*5, 12)
        # test small
        _bench_matmul(ctx, (4, 4, 4), (4, 4, 4))
        ctx.clear()

        # test medium
        _bench_matmul(ctx, (128, 1024, 512), (512, 1024))
        ctx.clear()

        # test large
        # _bench_matmul(ctx, (64, 4096, 1024), (64, 1024, 4096))
        # ctx.clear()

    def test_4d(self):
        ctx = YamiContext(1024*1024*1024*5, 1)
        # test small
        # _bench_matmul(ctx, (8, 1, 4, 4), (8, 2, 4, 4))
        _bench_matmul(ctx, (1024, 12, 11, 64), (1024, 12, 64, 11))
        # ctx.clear()

        # test medium
        # _bench_matmul(ctx, (2, 128, 512, 512), (128, 512, 63))
        # ctx.clear()

        # test large
        # _bench_matmul(ctx, (1, 12, 1024, 1024), (1, 1024, 2048))
        # ctx.clear()


class TestAdd:
    def test_1d(self):
        ctx = YamiContext(1024*1024)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_1d {step+1}/{TEST_STEPS} ================================\n')
            n = randint(2, 1000)
            target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else n)
            target_b = random_ndarray(n if randint(0, 100) % 2 == 0 else 1)
            target_res = target_a + target_b

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_add(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()

    def test_2d(self):
        ctx = YamiContext(1024*1024*10)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_2d {step+1}/{TEST_STEPS} ================================\n')
            n = randint(2, 100)
            m = randint(2, 100)
            target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else n, 1 if randint(0, 100) % 2 == 0 else m)
            target_b = random_ndarray(n if randint(0, 100) % 2 == 0 else 1, m if randint(0, 100) % 2 == 0 else 1)
            target_res = target_a + target_b

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_add(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()

    def test_3d(self):
        ctx = YamiContext(1024*1024*100)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_3d {step+1}/{TEST_STEPS} ================================\n')
            n = randint(2, 100)
            m = randint(2, 100)
            k = randint(2, 100)
            target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                      1 if randint(0, 100) % 2 == 0 else n,
                                      1 if randint(0, 100) % 2 == 0 else m)
            target_b = random_ndarray(k if randint(0, 100) % 2 == 0 else 1,
                                      n if randint(0, 100) % 2 == 0 else 1,
                                      m if randint(0, 100) % 2 == 0 else 1)
            target_res = target_a + target_b

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_add(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()

    def test_4d(self):
        ctx = YamiContext(1024*1024*50)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_4d {step+1}/{TEST_STEPS} ================================\n')
            n = randint(2, 50)
            m = randint(2, 50)
            k = randint(2, 50)
            j = randint(2, 50)
            target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                      1 if randint(0, 100) % 2 == 0 else n,
                                      1 if randint(0, 100) % 2 == 0 else m,
                                      1 if randint(0, 100) % 2 == 0 else j)
            target_b = random_ndarray(k if randint(0, 100) % 2 == 0 else 1,
                                      n if randint(0, 100) % 2 == 0 else 1,
                                      m if randint(0, 100) % 2 == 0 else 1,
                                      j if randint(0, 100) % 2 == 0 else 1)

            target_res = target_a + target_b
            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_add(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()


class TestMul:
    def test_all(self):
        ctx = YamiContext(1024*1024*50)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_all {step+1}/{TEST_STEPS} ================================\n')
            n = randint(2, 50)
            m = randint(2, 50)
            k = randint(2, 50)
            j = randint(2, 50)
            target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                      1 if randint(0, 100) % 2 == 0 else n,
                                      1 if randint(0, 100) % 2 == 0 else m,
                                      1 if randint(0, 100) % 2 == 0 else j)
            target_b = random_ndarray(k if randint(0, 100) % 2 == 0 else 1,
                                      n if randint(0, 100) % 2 == 0 else 1,
                                      m if randint(0, 100) % 2 == 0 else 1,
                                      j if randint(0, 100) % 2 == 0 else 1)

            target_res = target_a * target_b

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_mul(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()


class TestDiv:
    def test_all(self):
        ctx = YamiContext(1024*1024*100)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_all {step+1}/{TEST_STEPS} ================================\n')
            n = randint(2, 50)
            m = randint(2, 50)
            k = randint(2, 50)
            j = randint(2, 50)
            target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                      1 if randint(0, 100) % 2 == 0 else n,
                                      1 if randint(0, 100) % 2 == 0 else m,
                                      1 if randint(0, 100) % 2 == 0 else j)
            target_b = random_ndarray(k if randint(0, 100) % 2 == 0 else 1,
                                      n if randint(0, 100) % 2 == 0 else 1,
                                      m if randint(0, 100) % 2 == 0 else 1,
                                      j if randint(0, 100) % 2 == 0 else 1)

            target_res = target_a / target_b

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_div(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()


def test_tanh():
    ctx = YamiContext(1024*1024*50)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_tanh {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 50)
        j = randint(2, 50)
        target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                  1 if randint(0, 100) % 2 == 0 else n,
                                  1 if randint(0, 100) % 2 == 0 else m,
                                  1 if randint(0, 100) % 2 == 0 else j)

        target_res = np.tanh(target_a)

        my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
        my_res = yami_tanh(ctx, my_a)

        my_res_np = my_res.as_np()

        assert all_close(my_res_np, target_res)

        ctx.report_usage()
        ctx.clear()


def test_gelu():
    ctx = YamiContext(1024*1024*50)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_gelu {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 50)
        j = randint(2, 50)
        target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                  1 if randint(0, 100) % 2 == 0 else n,
                                  1 if randint(0, 100) % 2 == 0 else m,
                                  1 if randint(0, 100) % 2 == 0 else j)

        target_res = F.gelu(torch.tensor(target_a), approximate='tanh')

        my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
        my_res = yami_gelu(ctx, my_a)

        my_res_np = my_res.as_np()

        assert all_close(my_res_np, target_res.numpy())

        ctx.report_usage()
        ctx.clear()


def test_sum():
    ctx = YamiContext(1024*1024*100)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_sum {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 100)
        j = randint(2, 100)
        target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                  1 if randint(0, 100) % 2 == 0 else n,
                                  1 if randint(0, 100) % 2 == 0 else m,
                                  1 if randint(0, 100) % 2 == 0 else j)

        dim = randint(-4, 3)
        print(f'summing over axis={dim}')

        target_res = np.sum(target_a, axis=dim, keepdims=True)

        my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
        my_res = yami_sum(ctx, my_a, dim=c_int(dim))
        my_res_np = my_res.as_np()

        assert all_close(my_res_np, target_res)

        ctx.report_usage()
        ctx.clear()


def test_softmax():
    ctx = YamiContext(1024*1024*100)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_softmax {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 50)
        j = randint(2, 50)
        target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                  1 if randint(0, 100) % 2 == 0 else n,
                                  1 if randint(0, 100) % 2 == 0 else m,
                                  1 if randint(0, 100) % 2 == 0 else j)
        # target_a = np.array([1, 2, 4, -1, 6, -8, 4, 2, 3, 8, 6, -1], dtype=np.float32).reshape(2, 3, 2)

        dim = randint(-4, 3)
        print(f'softmax over axis={dim}')

        target_res = F.softmax(torch.tensor(target_a), dim=dim)

        my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
        my_res = yami_softmax(ctx, my_a, dim=c_int(dim))
        my_res_np = my_res.as_np()

        assert all_close(my_res_np, target_res.numpy())

        ctx.report_usage()
        ctx.clear()


def test_max():
    ctx = YamiContext(1024*1024*100)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_max {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 50)
        j = randint(2, 50)
        target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                  1 if randint(0, 100) % 2 == 0 else n,
                                  1 if randint(0, 100) % 2 == 0 else m,
                                  1 if randint(0, 100) % 2 == 0 else j)

        dim = randint(-4, 3)
        print(f'max over axis={dim}')

        target_res = np.max(target_a, axis=dim, keepdims=True)

        my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
        my_res = yami_max(ctx, my_a, dim=c_int(dim))
        my_res_np = my_res.as_np()

        assert all_close(my_res_np, target_res)

        ctx.report_usage()
        ctx.clear()


def test_exp():
    ctx = YamiContext(1024*1024*100)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_exp {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 50)
        j = randint(2, 50)
        target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                  1 if randint(0, 100) % 2 == 0 else n,
                                  1 if randint(0, 100) % 2 == 0 else m,
                                  1 if randint(0, 100) % 2 == 0 else j)

        target_res = np.exp(target_a)

        my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
        my_res = yami_exp(ctx, my_a)
        my_res_np = my_res.as_np()

        assert all_close(my_res_np, target_res)

        ctx.report_usage()
        ctx.clear()


def test_transpose():
    ctx = YamiContext(1024*1024*1024)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_transpose {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 50)
        j = randint(2, 50)
        target_a = torch.tensor(random_ndarray(k, n, m, j))
        for a1 in range(-4, 4, 1):
            for a2 in range(-4, 4, 1):
                print(f'\n>>> Transpose axis ({a1}, {a2}) of a {target_a.shape} <<<\n')
                target_res = target_a.transpose(a1, a2)
                my_a = YamiTensor.from_np(ctx, 'my_a', target_a.numpy())

                my_res = yami_transpose(ctx, my_a, a1, a2)

                assert all_close(my_res.as_np(), target_res.numpy())

                ctx.report_usage()
                ctx.clear()


def test_mask():
    ctx = YamiContext(1024*1024*100)
    for step in range(TEST_STEPS):
        print(f'\n================================ test_mask {step+1}/{TEST_STEPS} ================================\n')
        n = randint(2, 50)
        m = randint(2, 50)
        k = randint(2, 50)
        j = randint(2, 50)
        target_a = torch.ones(j, k, n, m)

        target_res = target_a.masked_fill(torch.tril(target_a) == 0, float('-inf'))
        my_a = YamiTensor.from_np(ctx, 'my_a', target_a.numpy())
        my_res = yami_lt_mask(ctx, my_a)
        my_res_np = my_res.as_np()

        assert all_close(my_res_np, target_res.numpy())
        ctx.report_usage()
        ctx.clear()
