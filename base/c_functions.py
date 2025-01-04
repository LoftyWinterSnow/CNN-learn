import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer

class CFunc:
    _double_p4 = ndpointer(dtype=np.double, ndim=4, flags='C')
    _double_p3 = ndpointer(dtype=np.double, ndim=3, flags='C')
    _double_p2 = ndpointer(dtype=np.double, ndim=2, flags='C')
    _double_p1 = ndpointer(dtype=np.double, ndim=1, flags='C')
    _int_p4 = ndpointer(dtype=np.int32, ndim=4, flags='C')
    _bool_p4 = ndpointer(dtype=np.bool, ndim=4, flags='C')
    lib = ct.CDLL('base/feature_map.dll', winmode=0)

    @classmethod
    def maxpool2d(cls, x, kernel_size, stride):
        _maxpool2d = cls.lib['maxpool2d']
        _maxpool2d.restype = None
        _maxpool2d.argtypes = [
            cls._double_p4, cls._double_p4, cls._bool_p4, ct.c_int, ct.c_int,
            ct.c_int, ct.c_int, ct.c_int, ct.c_int
        ]
        N, C, H, W = x.shape
        H1 = (H - kernel_size) // stride + 1
        W1 = (W - kernel_size) // stride + 1
        y = np.empty((N, C, H1, W1), dtype=np.double)
        mask = np.zeros((N, C, H, W)).astype(np.bool)
        _maxpool2d(x, y, mask, N, C, H, W, kernel_size, stride)
        return y, mask

    @classmethod
    def im2col(cls, x, kernel_size, stride):
        _im2col = cls.lib['im2col']
        _im2col.restype = None
        _im2col.argtypes = [
            cls._double_p3, cls._double_p2, ct.c_int, ct.c_int, ct.c_int,
            ct.c_int, ct.c_int
        ]
        C, H, W = x.shape
        W1 = kernel_size * kernel_size * C
        conv_w = (W - kernel_size) // stride + 1
        conv_h = (H - kernel_size) // stride + 1
        H1 = conv_h * conv_w
        y = np.empty((H1, W1), dtype=np.double)
        _im2col(x, y, C, H, W, kernel_size, stride)
        return y

    @classmethod
    def conv2d(cls, x, kernel, bias, stride):
        _conv2d = cls.lib['conv2d']
        _conv2d.restype = None
        _conv2d.argtypes = [
            cls._double_p4, cls._double_p4, cls._double_p3, cls._double_p4,
            cls._double_p1, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
            ct.c_int, ct.c_int
        ]
        N, C, H, W = x.shape
        kernel_size = kernel.shape[2]
        output_channel = kernel.shape[0]
        W1 = kernel_size * kernel_size * C
        conv_w = (W - kernel_size) // stride + 1
        conv_h = (H - kernel_size) // stride + 1
        H1 = conv_h * conv_w
        col_img = np.empty((N, H1, W1), dtype=np.double)
        y = np.empty((N, output_channel, conv_h, conv_w), dtype=np.double)
        _conv2d(x, y, col_img, kernel, bias, N, C, H, W, output_channel,
                kernel_size, stride)
        return y, col_img

    @classmethod
    def deconv2d(cls, pad_grad, input_shape, kernel, kernel_size, stride):
        _deconv2d = cls.lib['deconv2d']
        _deconv2d.restype = None
        _deconv2d.argtypes = [
            cls._double_p4, cls._double_p4, cls._double_p4, ct.c_int, ct.c_int,
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int
        ]
        N, C, H, W = pad_grad.shape
        y = np.empty(input_shape, dtype=np.double)
        _deconv2d(pad_grad, y, kernel, N, C, H, W, input_shape[1], kernel_size,
                  stride)
        return y


if __name__ == '__main__':
    img = np.random.randint(0, 5, (2, 1, 4, 4)).astype(np.float64)
    from cnn_layer import Conv2d, MaxPool2d
    m1 = MaxPool2d(2, 2)
    import code
    code.interact(local=locals())
