import os
import numpy as np
import ctypes
from ctypes import (
    c_bool,
    c_char,
    c_char_p,
    c_int,
    c_int8,
    c_int32,
    c_int64,
    c_uint8,
    c_uint32,
    c_size_t,
    c_float,
    c_double,
    c_void_p,
    Structure,
    Array,
    POINTER
)

_lib_file = f'{os.path.dirname(__file__)}/yami.so'
if not os.path.exists(_lib_file):
    raise RuntimeError(f'Error loading YAMI shared object "${_lib_file}"')

_lib = ctypes.CDLL(_lib_file)

yami_context_p = c_void_p
c_float_p = POINTER(c_float)
c_byte_p = POINTER(c_char)

YAMI_MAX_DIMS = 4
YAMI_MAX_LABEL = 64


class yami_context_init_params(Structure):
    _fields_ = [
        ('n_workers', c_int),
        ('mem_size', c_size_t),
        ('scratch_mem_size', c_size_t),
    ]


class yami_tensor(Structure):
    _fields_ = [
        ('ne', c_size_t),
        ('dimensions', c_size_t * YAMI_MAX_DIMS),
        ('extended_dim', c_size_t * YAMI_MAX_DIMS),
        ('stride', c_size_t * YAMI_MAX_DIMS),
        ('label', c_char * YAMI_MAX_LABEL),
        ('data', c_float_p),
        ('n_dim', c_int),
    ]


yami_tensor_p = POINTER(yami_tensor)


# extern yami_context *yami_init(yami_context_init_params params) noexcept;
def yami_init(params: yami_context_init_params) -> yami_context_p:
    return _lib.yami_init(params)


_lib.yami_init.argtypes = [yami_context_init_params]
_lib.yami_init.restype = yami_context_p


# extern void yami_free(yami_context *ctx) noexcept;
def yami_free(ctx: yami_context_p):
    return _lib.yami_free(ctx)


_lib.yami_free.argtypes = [yami_context_p]
_lib.yami_free.restype = None


# extern void yami_clear_ctx(yami_context *ctx) noexcept;
def yami_clear_ctx(ctx: yami_context_p):
    return _lib.yami_clear_ctx(ctx)


_lib.yami_clear_ctx.argtypes = [yami_context_p]
_lib.yami_clear_ctx.restype = None


# extern void yami_mem_usage(const yami_context *ctx) noexcept;
def yami_mem_usage(ctx: yami_context_p):
    return _lib.yami_mem_usage(ctx)


_lib.yami_mem_usage.argtypes = [yami_context_p]
_lib.yami_mem_usage.restype = None


# Wrappers
class YamiContext:
    def __init__(self, size: int, n_workers: int = 1):
        if size <= 0:
            raise RuntimeError(f'Invalid context size {size}')

        self._ctx_p = yami_init(yami_context_init_params(n_workers, size, 0))
        self._as_parameter_ = self._ctx_p

    def __del__(self):
        yami_free(self._ctx_p)

    def report_usage(self):
        yami_mem_usage(self._ctx_p)

    def clear(self):
        yami_clear_ctx(self._ctx_p)

    def from_param(self):
        return self._ctx_p


class YamiTensor:
    def __init__(self, raw: yami_tensor_p):
        self._tensor_p = raw
        self._as_parameter_ = self._tensor_p

    @classmethod
    def from_np(cls, ctx: YamiContext, label: str, x: np.ndarray):
        if x.dtype != np.float32:
            raise RuntimeError('NumPy array type must be float32')

        dims = list(x.shape)
        n_dims = len(dims)
        if n_dims > YAMI_MAX_DIMS or n_dims <= 0:
            raise RuntimeError(f'Invalid number of dimensions {n_dims}')

        if n_dims == 1:
            res = YamiTensor(yami_tensor_1d(ctx, bytes(label, 'ascii'), dims[0]))
        elif n_dims == 2:
            res = YamiTensor(yami_tensor_2d(ctx, bytes(label, 'ascii'), dims[0], dims[1]))
        elif n_dims == 3:
            res = YamiTensor(yami_tensor_3d(ctx, bytes(label, 'ascii'), dims[0], dims[1], dims[2]))
        else:
            res = YamiTensor(yami_tensor_4d(ctx, bytes(label, 'ascii'), dims[0], dims[1], dims[2], dims[3]))

        ctypes.memmove(res._tensor_p.contents.data, x.ctypes.data, x.nbytes)
        return res

    def as_np(self):
        t = self._tensor_p.contents
        dim_arr = []
        for i in range(t.n_dim):
            dim_arr.append(t.dimensions[i])

        return np.ctypeslib.as_array(t.data, shape=tuple(dim_arr))

    def reshape(self, *dims: int):
        yami_reshape(self, *dims)

    def from_param(self):
        return self._tensor_p


def yami_tensor_1d(ctx: YamiContext, label: c_char_p, dim1: c_size_t) -> yami_tensor_p:
    return _lib.yami_tensor_1d(ctx, label, dim1)


_lib.yami_tensor_1d.argtypes = [yami_context_p, c_char_p, c_size_t]
_lib.yami_tensor_1d.restype = yami_tensor_p


def yami_tensor_2d(ctx: YamiContext, label: c_char_p, dim1: c_size_t, dim2: c_size_t) -> yami_tensor_p:
    return _lib.yami_tensor_2d(ctx, label, dim1, dim2)


_lib.yami_tensor_2d.argtypes = [yami_context_p, c_char_p, c_size_t, c_size_t]
_lib.yami_tensor_2d.restype = yami_tensor_p


def yami_tensor_3d(ctx: YamiContext, label: c_char_p, dim1: c_size_t, dim2: c_size_t, dim3: c_size_t) -> yami_tensor_p:
    return _lib.yami_tensor_3d(ctx, label, dim1, dim2, dim3)


_lib.yami_tensor_3d.argtypes = [yami_context_p, c_char_p, c_size_t, c_size_t, c_size_t]
_lib.yami_tensor_3d.restype = yami_tensor_p


def yami_tensor_4d(ctx: YamiContext, label: c_char_p, dim1: c_size_t, dim2: c_size_t,
                   dim3: c_size_t, dim4: c_size_t) -> yami_tensor_p:
    return _lib.yami_tensor_4d(ctx, label, dim1, dim2, dim3, dim4)


_lib.yami_tensor_4d.argtypes = [yami_context_p, c_char_p, c_size_t, c_size_t, c_size_t, c_size_t]
_lib.yami_tensor_4d.restype = yami_tensor_p


def yami_reshape(x: YamiTensor, *dims: c_size_t) -> YamiTensor:
    return _lib.yami_reshape(x, len(dims), *dims)


_lib.yami_reshape.argtypes = [yami_tensor_p, c_int]
_lib.yami_reshape.restype = yami_tensor_p


def yami_transpose(ctx: YamiContext, x: YamiTensor, dim1: int = -1, dim2: int = -2) -> YamiTensor:
    return YamiTensor(_lib.yami_transpose(ctx, x, dim1, dim2))


_lib.yami_transpose.argtypes = [yami_context_p, yami_tensor_p, c_int, c_int]
_lib.yami_transpose.restype = yami_tensor_p


def yami_contiguous(ctx: YamiContext, x: YamiTensor) -> YamiTensor:
    return YamiTensor(_lib.yami_contiguous(ctx, x))


_lib.yami_contiguous.argtypes = [yami_context_p, yami_tensor_p]
_lib.yami_contiguous.restype = yami_tensor_p


def yami_lt_mask(ctx: YamiContext, x: YamiTensor, mask: float = float('-inf')) -> YamiTensor:
    return YamiTensor(_lib.yami_lt_mask(ctx, x, mask))


_lib.yami_lt_mask.argtypes = [yami_context_p, yami_tensor_p, c_float]
_lib.yami_lt_mask.restype = yami_tensor_p


def yami_split(ctx: YamiContext, x: YamiTensor, n: int, offset: int, dim: int = -1) -> YamiTensor:
    return YamiTensor(_lib.yami_split(ctx, x, n, offset, dim))


_lib.yami_split.argtypes = [yami_context_p, yami_tensor_p, c_size_t, c_int, c_int]
_lib.yami_split.restype = yami_tensor_p


def yami_matmul(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor) -> YamiTensor:
    return YamiTensor(_lib.yami_matmul(ctx, xa, xb, None))


_lib.yami_matmul.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p, yami_tensor_p]
_lib.yami_matmul.restype = yami_tensor_p


def yami_add(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor, in_place: bool = False) -> YamiTensor:
    return YamiTensor(_lib.yami_add(ctx, xa, xb, in_place))


_lib.yami_add.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p, c_bool]
_lib.yami_add.restype = yami_tensor_p


def yami_mul(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor, in_place: bool = False) -> YamiTensor:
    return YamiTensor(_lib.yami_mul(ctx, xa, xb, in_place))


_lib.yami_mul.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p, c_bool]
_lib.yami_mul.restype = yami_tensor_p


def yami_div(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor, in_place: bool = False) -> YamiTensor:
    return YamiTensor(_lib.yami_div(ctx, xa, xb, in_place))


_lib.yami_div.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p, c_bool]
_lib.yami_div.restype = yami_tensor_p


# def yami_tanh(ctx: YamiContext, x: YamiTensor, in_place: c_bool = True) -> YamiTensor:
#     return YamiTensor(_lib.yami_tanh(ctx, x, in_place))
#
#
# _lib.yami_tanh.argtypes = [yami_context_p, yami_tensor_p, c_bool]
# _lib.yami_tanh.restype = yami_tensor_p


def yami_gelu(ctx: YamiContext, x: YamiTensor, in_place: c_bool = True) -> YamiTensor:
    return YamiTensor(_lib.yami_gelu(ctx, x, in_place))


_lib.yami_gelu.argtypes = [yami_context_p, yami_tensor_p, c_bool]
_lib.yami_gelu.restype = yami_tensor_p


def yami_softmax(ctx: YamiContext, x: YamiTensor, dim: c_int) -> YamiTensor:
    return YamiTensor(_lib.yami_softmax(ctx, x, dim))


_lib.yami_softmax.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_softmax.restype = yami_tensor_p


def yami_sum(ctx: YamiContext, x: YamiTensor, dim: c_int) -> YamiTensor:
    return YamiTensor(_lib.yami_sum(ctx, x, dim))


_lib.yami_sum.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_sum.restype = yami_tensor_p


def yami_max(ctx: YamiContext, x: YamiTensor, dim: c_int) -> YamiTensor:
    return YamiTensor(_lib.yami_max(ctx, x, dim))


_lib.yami_max.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_max.restype = yami_tensor_p


def yami_mean(ctx: YamiContext, x: YamiTensor, dim: int) -> YamiTensor:
    return YamiTensor(_lib.yami_mean(ctx, x, dim))


_lib.yami_mean.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_mean.restype = yami_tensor_p


def yami_var(ctx: YamiContext, x: YamiTensor, dim: int) -> YamiTensor:
    return YamiTensor(_lib.yami_var(ctx, x, dim))


_lib.yami_var.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_var.restype = yami_tensor_p


def yami_exp(ctx: YamiContext, x: YamiTensor, in_place: bool = True) -> YamiTensor:
    return YamiTensor(_lib.yami_exp(ctx, x, in_place))


_lib.yami_exp.argtypes = [yami_context_p, yami_tensor_p, c_bool]
_lib.yami_exp.restype = yami_tensor_p


def yami_layer_norm(ctx: YamiContext, w: YamiTensor, b: YamiTensor, x: YamiTensor,
                    in_place: bool = True, eps: float = float('1e-5')) -> YamiTensor:
    return YamiTensor(_lib.yami_layer_norm(ctx, w, b, x, in_place, eps))


_lib.yami_layer_norm.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p,
                                 yami_tensor_p, c_bool, c_float]
_lib.yami_layer_norm.restype = yami_tensor_p
