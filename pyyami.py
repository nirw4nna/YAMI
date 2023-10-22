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

_lib_file = f'{os.path.dirname(__file__)}/yami2.so'
if not os.path.exists(_lib_file):
    raise RuntimeError(f'Error loading YAMI shared object "${_lib_file}"')

_lib = ctypes.CDLL(_lib_file)

yami_context_p = c_void_p
c_float_p = POINTER(c_float)
c_byte_p = POINTER(c_char)

YAMI_MAX_DIMS = 4
YAMI_MAX_LABEL = 64


# struct yami_context_init_params {
#       size mem_size;
#       size scratch_mem_size;
#       void *mem_buffer;
#       void *scratch_mem_buffer;
# };
class yami_context_init_params(Structure):
    _fields_ = [
        ('mem_size', c_int64),
        ('scratch_mem_size', c_int64),
        ('mem_buffer', c_void_p),
        ('scratch_mem_buffer', c_void_p),
    ]


# struct yami_tensor {
#       int n_dim;
#       size ne;
#       size dimensions[yami_max_dims];
#       char label[yami_max_label];
#       f32 *data;
# };
class yami_tensor(Structure):
    _fields_ = [
        ('n_dim', c_int),
        ('ne', c_int64),
        ('dimensions', c_int64 * YAMI_MAX_DIMS),
        ('extended_dim', c_int64 * YAMI_MAX_DIMS),
        ('stride', c_int64 * YAMI_MAX_DIMS),
        ('label', c_char * YAMI_MAX_LABEL),
        ('data', c_float_p),
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
    def __init__(self, size: int):
        if size <= 0:
            raise RuntimeError(f'Invalid context size {size}')

        self._ctx_p = yami_init(yami_context_init_params(size, 0, None, None))
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


# extern yami_tensor *yami_tensor_1d(yami_context *ctx, const char *label,
#                                    size dim1) noexcept;
def yami_tensor_1d(ctx: YamiContext, label: c_char_p, dim1: c_int64) -> yami_tensor_p:
    return _lib.yami_tensor_1d(ctx, label, dim1)


_lib.yami_tensor_1d.argtypes = [yami_context_p, c_char_p, c_int64]
_lib.yami_tensor_1d.restype = yami_tensor_p


# extern yami_tensor *yami_tensor_2d(yami_context *ctx, std::string_view label,
#                                    size dim1, size dim2) noexcept;
def yami_tensor_2d(ctx: YamiContext, label: c_char_p, dim1: c_int64, dim2: c_int64) -> yami_tensor_p:
    return _lib.yami_tensor_2d(ctx, label, dim1, dim2)


_lib.yami_tensor_2d.argtypes = [yami_context_p, c_char_p, c_int64, c_int64]
_lib.yami_tensor_2d.restype = yami_tensor_p


# extern yami_tensor *yami_tensor_3d(yami_context *ctx, const char *label,
#                                    size dim1, size dim2,
#                                    size dim3) noexcept;
def yami_tensor_3d(ctx: YamiContext, label: c_char_p, dim1: c_int64, dim2: c_int64, dim3: c_int64) -> yami_tensor_p:
    return _lib.yami_tensor_3d(ctx, label, dim1, dim2, dim3)


_lib.yami_tensor_3d.argtypes = [yami_context_p, c_char_p, c_int64, c_int64, c_int64]
_lib.yami_tensor_3d.restype = yami_tensor_p


# extern yami_tensor *yami_tensor_4d(yami_context *ctx, const char *label,
#                                    size dim1, size dim2,
#                                    size dim3, size dim4) noexcept;
def yami_tensor_4d(ctx: YamiContext, label: c_char_p, dim1: c_int64, dim2: c_int64, dim3: c_int64, dim4: c_int64) -> yami_tensor_p:
    return _lib.yami_tensor_4d(ctx, label, dim1, dim2, dim3, dim4)


_lib.yami_tensor_4d.argtypes = [yami_context_p, c_char_p, c_int64, c_int64, c_int64, c_int64]
_lib.yami_tensor_4d.restype = yami_tensor_p


# extern void yami_reshape(yami_tensor *x, int n_dims...) noexcept;
def yami_reshape(x: YamiTensor, *dims: c_int64) -> YamiTensor:
    return _lib.yami_reshape(x, len(dims), *dims)


_lib.yami_reshape.argtypes = [yami_tensor_p, c_int]
_lib.yami_reshape.restype = yami_tensor_p


# extern yami_tensor *yami_transpose(yami_context *ctx,
#                                    yami_tensor *x,
#                                    int dim1 = -1,  int dim2 = -2,
#                                    bool in_place = false) noexcept;
def yami_transpose(ctx: YamiContext, x: YamiTensor, dim1: int = -1, dim2: int = -2) -> YamiTensor:
    return YamiTensor(_lib.yami_transpose(ctx, x, dim1, dim2))


_lib.yami_transpose.argtypes = [yami_context_p, yami_tensor_p, c_int, c_int]
_lib.yami_transpose.restype = yami_tensor_p


# extern yami_tensor *yami_lt_mask(yami_context *ctx,
#                                  const yami_tensor *x,
#                                  f32 mask) noexcept;
def yami_lt_mask(ctx: YamiContext, x: YamiTensor, mask: float = float('-inf')) -> YamiTensor:
    return YamiTensor(_lib.yami_lt_mask(ctx, x, mask))


_lib.yami_lt_mask.argtypes = [yami_context_p, yami_tensor_p, c_float]
_lib.yami_lt_mask.restype = yami_tensor_p


# extern yami_tensor *yami_matmul(yami_context *ctx,
#                                 const yami_tensor *xa,
#                                 const yami_tensor *xb) noexcept;
def yami_matmul(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor) -> YamiTensor:
    return YamiTensor(_lib.yami_matmul(ctx, xa, xb))


_lib.yami_matmul.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p]
_lib.yami_matmul.restype = yami_tensor_p


# extern yami_tensor *yami_add(yami_context *ctx,
#                              const yami_tensor *xa,
#                              const yami_tensor *xb) noexcept;
def yami_add(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor) -> YamiTensor:
    return YamiTensor(_lib.yami_add(ctx, xa, xb))


_lib.yami_add.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p]
_lib.yami_add.restype = yami_tensor_p


# extern yami_tensor *yami_mul(yami_context *ctx,
#                              const yami_tensor *xa,
#                              const yami_tensor *xb) noexcept;
def yami_mul(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor) -> YamiTensor:
    return YamiTensor(_lib.yami_mul(ctx, xa, xb))


_lib.yami_mul.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p]
_lib.yami_mul.restype = yami_tensor_p


# extern yami_tensor *yami_div(yami_context *ctx,
#                              const yami_tensor *xa,
#                              const yami_tensor *xb) noexcept;
def yami_div(ctx: YamiContext, xa: YamiTensor, xb: YamiTensor) -> YamiTensor:
    return YamiTensor(_lib.yami_div(ctx, xa, xb))


_lib.yami_div.argtypes = [yami_context_p, yami_tensor_p, yami_tensor_p]
_lib.yami_div.restype = yami_tensor_p


# extern yami_tensor *yami_tanh(yami_context *ctx,
#                               yami_tensor *x,
#                               bool in_place = true) noexcept;
def yami_tanh(ctx: YamiContext, x: YamiTensor, in_place: c_bool = True) -> YamiTensor:
    return YamiTensor(_lib.yami_tanh(ctx, x, in_place))


_lib.yami_tanh.argtypes = [yami_context_p, yami_tensor_p, c_bool]
_lib.yami_tanh.restype = yami_tensor_p


# extern yami_tensor *yami_gelu(yami_context *ctx,
#                               yami_tensor *x,
#                               bool in_place = true) noexcept;
def yami_gelu(ctx: YamiContext, x: YamiTensor, in_place: c_bool = True) -> YamiTensor:
    return YamiTensor(_lib.yami_gelu(ctx, x, in_place))


_lib.yami_gelu.argtypes = [yami_context_p, yami_tensor_p, c_bool]
_lib.yami_gelu.restype = yami_tensor_p


# extern yami_tensor *yami_softmax(yami_context *ctx,
#                                  yami_tensor *x, int dim) noexcept;
def yami_softmax(ctx: YamiContext, x: YamiTensor, dim: c_int) -> YamiTensor:
    return YamiTensor(_lib.yami_softmax(ctx, x, dim))


_lib.yami_softmax.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_softmax.restype = yami_tensor_p


# extern yami_tensor *yami_sum(yami_context *ctx,
#                              const yami_tensor *x, int dim) noexcept;
def yami_sum(ctx: YamiContext, x: YamiTensor, dim: c_int) -> YamiTensor:
    return YamiTensor(_lib.yami_sum(ctx, x, dim))


_lib.yami_sum.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_sum.restype = yami_tensor_p


# extern yami_tensor *yami_max(yami_context *ctx,
#                              const yami_tensor *x, int dim) noexcept;
def yami_max(ctx: YamiContext, x: YamiTensor, dim: c_int) -> YamiTensor:
    return YamiTensor(_lib.yami_max(ctx, x, dim))


_lib.yami_max.argtypes = [yami_context_p, yami_tensor_p, c_int]
_lib.yami_max.restype = yami_tensor_p


# extern yami_tensor *yami_exp(yami_context *ctx,
#                              const yami_tensor *x, int dim) noexcept;
def yami_exp(ctx: YamiContext, x: YamiTensor, in_place: c_bool = True) -> YamiTensor:
    return YamiTensor(_lib.yami_exp(ctx, x, in_place))


_lib.yami_exp.argtypes = [yami_context_p, yami_tensor_p, c_bool]
_lib.yami_exp.restype = yami_tensor_p
