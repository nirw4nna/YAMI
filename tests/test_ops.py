# Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the MIT license
# (https://opensource.org/license/mit).

from random import randint

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple
import sys
import os

sys.path.append(f'{os.path.join(os.path.dirname(__file__), "pyyami")}')
from pyyami import *

TEST_STEPS = 10
RNG = np.random.default_rng()


def _all_close(actual, target, eps=1e-5):
    diffs = ~np.isclose(actual, target, atol=eps, rtol=eps, equal_nan=True)
    close = len(actual[diffs]) == 0
    if not close:
        print(f'Wrong indexes: {np.where(diffs == True)}')
        print(f'My values: {actual[diffs]}\nNP values: {target[diffs]}')
    return close


def _random_ndarray(*dims: int) -> np.ndarray:
    return (RNG.random(dims, dtype=np.float32) + 100).astype(np.float32)


def _random_shape() -> tuple[int, int, int, int]:
    return randint(2, 64), randint(2, 64), randint(2, 64), randint(2, 64)


def _input_params(is_binary_op: bool = False, is_matmul: bool = False, same_shape: bool = False) \
        -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    m, n, k, j = _random_shape()
    if is_binary_op:
        if is_matmul:
            return (_random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                    1 if randint(0, 100) % 2 == 0 else n,
                                    m,
                                    j),
                    _random_ndarray(k if randint(0, 100) % 2 == 0 else 1,
                                    n if randint(0, 100) % 2 == 0 else 1,
                                    j,
                                    m))
        elif same_shape:
            return _random_ndarray(m, n, k, j), _random_ndarray(m, n, k, j)
        else:
            return (_random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                                    1 if randint(0, 100) % 2 == 0 else n,
                                    1 if randint(0, 100) % 2 == 0 else m,
                                    1 if randint(0, 100) % 2 == 0 else j),
                    _random_ndarray(k if randint(0, 100) % 2 == 0 else 1,
                                    n if randint(0, 100) % 2 == 0 else 1,
                                    m if randint(0, 100) % 2 == 0 else 1,
                                    j if randint(0, 100) % 2 == 0 else 1))

    else:
        return _random_ndarray(1 if randint(0, 100) % 2 == 0 else k,
                               1 if randint(0, 100) % 2 == 0 else n,
                               1 if randint(0, 100) % 2 == 0 else m,
                               1 if randint(0, 100) % 2 == 0 else j)


@pytest.fixture
def ctx():
    ctx = YamiContext(1024 * 1024 * 1024, 12)
    yield ctx


class TestOps:
    def test_matmul(self, ctx: YamiContext):
        # In YAMI matmul performs C = AB^T, this means target_b must be transposed
        for _ in range(TEST_STEPS):
            target_a, target_b = _input_params(is_binary_op=True, is_matmul=True)

            my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
            my_b = YamiTensor.from_np(ctx, 'my_b', np.ascontiguousarray(target_b.swapaxes(-1, -2)))

            my_res = yami_matmul(ctx, my_a, my_b)

            assert _all_close(my_res.as_np(), np.matmul(target_a, target_b))

            ctx.clear()

        # Test some big 2-D matrices
        for _ in range(TEST_STEPS):
            a = _random_ndarray(1, 1, randint(1, 1024), 1024)
            b = _random_ndarray(1, 1, 1024, 1024)
            c = np.matmul(a, b.swapaxes(-1, -2))
            my_a = YamiTensor.from_np(ctx, 'my_a', a)
            my_b = YamiTensor.from_np(ctx, 'my_b', b)

            my_res = yami_matmul(ctx, my_a, my_b)

            assert _all_close(my_res.as_np(), c)
            ctx.clear()

        # Test some big 1-D matrices
        for _ in range(TEST_STEPS):
            a = _random_ndarray(1, 1, 1, 1024)
            b = _random_ndarray(1, 1, randint(1, 1024), 1024)
            c = np.matmul(a, b.swapaxes(-1, -2))
            my_a = YamiTensor.from_np(ctx, 'my_a', a)
            my_b = YamiTensor.from_np(ctx, 'my_b', b)

            my_res = yami_matmul(ctx, my_a, my_b)

            assert _all_close(my_res.as_np(), c)
            ctx.clear()

    def test_binary(self, ctx: YamiContext):
        ops_to_test = {
            "yami_add": "add",
            "yami_sub": "subtract",
            "yami_mul": "multiply",
            "yami_div": "divide",
        }
        for yami_name, np_name in ops_to_test.items():
            print(f'Testing: {yami_name}')
            yami_op = getattr(pyyami, yami_name)
            np_op = getattr(np, np_name)
            for _ in range(TEST_STEPS):
                target_a, target_b = _input_params(is_binary_op=True,
                                                   is_matmul=yami_name == 'yami_matmul')

                my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
                my_b = YamiTensor.from_np(ctx, 'my_b', target_b)

                my_res = yami_op(ctx, my_a, my_b)

                assert _all_close(my_res.as_np(), np_op(target_a, target_b))

                ctx.clear()

    def test_unary(self, ctx: YamiContext):
        ops_to_test = {
            "yami_exp": "exp",
            "yami_sqrt": "sqrt",
            "yami_square": "square",
            "yami_gelu": "",
            "yami_swiglu": "",
            "yami_lt_mask": "",
        }
        for yami_name, np_name in ops_to_test.items():
            print(f'Testing {yami_name}')
            np_op = getattr(np, np_name, None)
            yami_op = getattr(pyyami, yami_name)
            for step in range(TEST_STEPS):
                target_a = _input_params()

                if yami_name == 'yami_gelu':
                    target_res = F.gelu(torch.tensor(target_a), approximate='tanh').numpy()
                elif yami_name == 'yami_swiglu':
                    target_res = F.silu(torch.tensor(target_a)).numpy()
                elif yami_name == 'yami_lt_mask':
                    a_pt = torch.tensor(target_a)
                    target_res = a_pt.masked_fill(torch.tril(a_pt) == 0, float('-inf')).numpy()
                else:
                    target_res = np_op(target_a)

                my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
                my_res = yami_op(ctx, my_a)

                assert _all_close(my_res.as_np(), target_res)

                ctx.clear()

    def test_unary_along_axis(self, ctx: YamiContext):
        ops_to_test = {
            "yami_sum": "sum",
            "yami_softmax": "",
            "yami_max": "max",
            "yami_mean": "mean",
            "yami_var": "var"
        }
        for yami_name, np_name in ops_to_test.items():
            print(f'Testing {yami_name}')
            np_op = getattr(np, np_name, None)
            yami_op = getattr(pyyami, yami_name)
            for _ in range(TEST_STEPS):
                target_a = _input_params()
                for axis in range(-YAMI_MAX_DIMS, YAMI_MAX_DIMS, 1):

                    if yami_name == 'yami_softmax':
                        target_res = F.softmax(torch.tensor(target_a), dim=axis).numpy()
                    else:
                        target_res = np_op(target_a, axis=axis, keepdims=True)

                    my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
                    my_res = yami_op(ctx, my_a, dim=axis)

                    assert _all_close(my_res.as_np(), target_res)

                    ctx.clear()

    def test_concat(self, ctx: YamiContext):
        # Binary along axis
        for _ in range(TEST_STEPS):
            target_a, target_b = _input_params(is_binary_op=True, same_shape=True)
            for axis in range(-YAMI_MAX_DIMS, YAMI_MAX_DIMS, 1):
                target_res = np.concatenate((target_a, target_b), axis=axis)
                my_a = YamiTensor.from_np(ctx, 'my_a', target_a)
                my_b = YamiTensor.from_np(ctx, 'my_b', target_b)

                my_res = yami_concat(ctx, my_a, my_b, dim=axis)

                assert _all_close(my_res.as_np(), target_res)

                ctx.clear()

    def test_reshape(self, ctx: YamiContext):
        target_a = _random_ndarray(20, 10, 5, 2)
        my_a = YamiTensor.from_np(ctx, "my_a", target_a)

        assert _all_close(yami_reshape(my_a, 20, 10, 10).as_np(), target_a.reshape((20, 10, 10)))
        assert _all_close(yami_reshape(my_a, 20, 100).as_np(), target_a.reshape((20, 100)))
        assert _all_close(yami_reshape(my_a, 5, 20, 10, 2).as_np(), target_a.reshape((5, 20, 10, 2)))

    def test_transpose(self, ctx: YamiContext):
        for _ in range(TEST_STEPS):
            target_a = _input_params()
            for a1 in range(-YAMI_MAX_DIMS, YAMI_MAX_DIMS, 1):
                for a2 in range(-YAMI_MAX_DIMS, YAMI_MAX_DIMS, 1):
                    target_res = torch.tensor(target_a).transpose(a1, a2).numpy()
                    my_a = YamiTensor.from_np(ctx, 'my_a', target_a)

                    my_res = yami_contiguous(ctx, yami_transpose(ctx, my_a, a1, a2))

                    assert _all_close(my_res.as_np(), target_res)
                    ctx.clear()

    def test_norm(self, ctx: YamiContext):
        op_to_test = ['yami_layer_norm', 'yami_rms_norm']
        for op_name in op_to_test:
            x = torch.randn((5, 10))
            w = torch.ones(10)
            my_x = YamiTensor.from_np(ctx, "my_x", x.numpy())
            my_w = YamiTensor.from_np(ctx, "my_w", w.numpy())
            yami_op = getattr(pyyami, op_name)
            if op_name == 'yami_layer_norm':
                b = torch.zeros(10)
                my_b = YamiTensor.from_np(ctx, "my_b", b.numpy())
                my_res = yami_op(ctx, my_w, my_b, my_x)
                my_res_np = my_res.as_np()
                for i in range(x.shape[0]):
                    assert (np.isclose(my_res_np[i, :].mean(), 0, atol=1e-5, rtol=1e-5) and
                            np.isclose(my_res_np[i, :].std(), 1, atol=1e-5, rtol=1e-5))
            elif op_name == 'yami_rms_norm':
                res = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)) * w
                my_res = yami_op(ctx, my_w, my_x)
                assert _all_close(my_res.as_np(), res)

            ctx.clear()

    def test_split(self, ctx: YamiContext):
        a = torch.randn((3, 9, 15, 2000))
        split_sizes = [1, 3, 5, 200]
        for i in range(YAMI_MAX_DIMS):
            dim = a.shape[i]
            split_size = split_sizes[i]
            n = int(dim / split_size)
            targets = a.split(split_size, dim=i)
            my_a = YamiTensor.from_np(ctx, "my_a", a.numpy())
            for j in range(n):
                my_res = yami_split(ctx, my_a, split_size, j, dim=i)
                assert _all_close(my_res.as_np(), targets[j].numpy())

            ctx.clear()

    def test_rope(self, ctx: YamiContext):
        seq_len = 12
        dim = 128
        end = 2048
        freqs = 1. / (10_000. ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs)
        freq_cos = torch.cos(freqs)
        freq_sin = torch.sin(freqs)
        
        def _rope(
                q: torch.Tensor,
                k: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            def _reshape_freqs(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                ndim = x.ndim
                assert 0 <= 1 < ndim
                assert f.shape == (x.shape[1], x.shape[-1])
                shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
                return f.view(shape)

            q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1)
            k_r, k_i = k.float().reshape(k.shape[:-1] + (-1, 2)).unbind(-1)

            fc = _reshape_freqs(freq_cos[:seq_len], q_r)
            fs = _reshape_freqs(freq_sin[:seq_len], q_r)

            q_out_r = q_r * fc - q_i * fs
            q_out_i = q_r * fs + q_i * fc
            k_out_r = k_r * fc - k_i * fs
            k_out_i = k_r * fs + k_i * fc

            q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(3)
            k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(3)

            return q_out.type_as(q), k_out.type_as(k)

        q_test = _random_ndarray(1, 12, 32, 128)
        k_test = _random_ndarray(1, 12, 32, 128)
        # RoPE input: q=[1, 12, 32, 128] k=[1, 12, 32, 128] freq_cos=[12, 64] freq_sin=[12, 64]
        q_target, k_target = _rope(torch.Tensor(q_test), torch.Tensor(k_test))

        q_yami = YamiTensor.from_np(ctx, 'q_yami', q_test)
        k_yami = YamiTensor.from_np(ctx, 'k_yami', k_test)
        q_res = yami_rope(ctx, q_yami, 128, False, 0)
        k_res = yami_rope(ctx, k_yami, 128, True, 0)

        assert _all_close(q_res.as_np(), q_target.numpy())
        assert _all_close(k_res.as_np(), k_target.numpy())
