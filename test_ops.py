from pyyami import *
import numpy as np
from random import randint


TEST_STEPS = 100


def all_close(actual, target, eps=1e-5):
    diffs = ~np.isclose(actual, target, atol=eps, rtol=eps)
    close = len(actual[diffs]) == 0
    if not close:
        print(f'My values: {actual[diffs]}\nNP values: {target[diffs]}')

    return close


def random_ndarray(*dims: int) -> np.ndarray:
    return np.random.randn(*dims).astype(np.float32)


class TestMatmul:
    def test_2d_square(self):
        ctx = YamiContext(1024*1024)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_2d_square {step+1}/{TEST_STEPS} ================================\n')
            n = randint(1, 100)
            target_a = random_ndarray(n, n)
            target_b = random_ndarray(n, n)
            target_res = np.matmul(target_a, target_b)

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_matmul(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()

    def test_2d_rect(self):
        ctx = YamiContext(1024*1024)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_2d_rect {step+1}/{TEST_STEPS} ================================\n')
            n = randint(1, 100)
            m = randint(1, 100)
            target_a = random_ndarray(n, m)
            target_b = random_ndarray(m, n)

            target_res = np.matmul(target_a, target_b)

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)

            my_res = yami_matmul(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()

    def test_3d(self):
        ctx = YamiContext(1024*1024*10)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_3d {step+1}/{TEST_STEPS} ================================\n')
            n = randint(1, 100)
            m = randint(1, 100)
            i = randint(2, 100)

            target_a = random_ndarray(i if randint(0, 100) % 2 == 0 else 1, n, m)
            target_b = random_ndarray(1 if randint(0, 100) % 2 == 0 else i, m, n)

            target_res = np.matmul(target_a, target_b)

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)

            my_res = yami_matmul(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()

    def test_4d(self):
        ctx = YamiContext(1024*1024*100)
        for step in range(TEST_STEPS):
            print(f'\n================================ test_4d {step+1}/{TEST_STEPS} ================================\n')
            n = randint(2, 50)
            m = randint(2, 50)
            i = randint(2, 50)
            j = randint(2, 50)
            target_a = random_ndarray(1 if randint(0, 100) % 2 == 0 else j, i if randint(0, 100) % 2 == 0 else 1,
                                      n, m)
            target_b = random_ndarray(j if randint(0, 100) % 2 == 0 else 1, 1 if randint(0, 100) % 2 == 0 else i,
                                      m, n)

            target_res = np.matmul(target_a, target_b)
            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', target_b)
            my_res = yami_matmul(ctx, my_a, my_b)
            my_res_np = my_res.as_np()

            assert all_close(my_res_np, target_res)

            ctx.report_usage()
            ctx.clear()


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
