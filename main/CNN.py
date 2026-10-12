"""
CNN building blocks in the same style as Layer_Dense:
    forward(x) sets self.output
    backward(dvalues) sets self.dinputs (and dweight / dbiases for layers with params)
    update_params() applies Adam (trainable layers only)

Images are shaped (N, C, H, W).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from Layer import AdamOptimizer

F32 = np.float32


def make_adam(shape, lr):
    """AdamOptimizer from Layer.py, with float32 moment buffers."""
    opt = AdamOptimizer(shape, learning_rate=lr)
    opt.m = np.zeros(shape, dtype=F32)
    opt.v = np.zeros(shape, dtype=F32)
    return opt


class Layer_Conv2D:
    """
    Stride-1 convolution implemented with im2col so the heavy work is one big matmul.
    Weights are stored as a 2D matrix (C*k*k, F) so they plug into AdamOptimizer
    exactly like Layer_Dense weights.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, learning_rate=0.001):
        self.C = in_channels
        self.F = out_channels
        self.k = kernel_size
        self.p = padding

        fan_in = in_channels * kernel_size * kernel_size
        # He init (ReLU follows every conv)
        self.weights = (np.random.randn(fan_in, out_channels) * np.sqrt(2.0 / fan_in)).astype(F32)
        self.biases = np.zeros((1, out_channels), dtype=F32)

        self.opt_weights = make_adam(self.weights.shape, learning_rate)
        self.opt_biases = make_adam(self.biases.shape, learning_rate)

        # The first conv layer doesn't need to pass gradients back to the image
        self.need_input_grad = True

    def forward(self, x):
        N, C, H, W = x.shape
        k, p = self.k, self.p

        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        Ho, Wo = H + 2 * p - k + 1, W + 2 * p - k + 1

        # (N, C, Ho, Wo, k, k) view -> (N*Ho*Wo, C*k*k) matrix
        windows = sliding_window_view(xp, (k, k), axis=(2, 3))
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k)

        out = cols @ self.weights + self.biases  # (N*Ho*Wo, F)

        self.cols = cols
        self.x_shape = x.shape
        self.out_hw = (Ho, Wo)
        self.output = np.ascontiguousarray(
            out.reshape(N, Ho, Wo, self.F).transpose(0, 3, 1, 2)
        )

    def backward(self, dvalues):
        N, C, H, W = self.x_shape
        Ho, Wo = self.out_hw
        k, p = self.k, self.p

        d2 = dvalues.transpose(0, 2, 3, 1).reshape(N * Ho * Wo, self.F)

        self.dweight = self.cols.T @ d2
        self.dbiases = d2.sum(axis=0, keepdims=True)

        if not self.need_input_grad:
            self.dinputs = None
            return

        # Gradient w.r.t. the im2col matrix, then scatter-add back to the image (col2im)
        dcols = (d2 @ self.weights.T).reshape(N, Ho, Wo, C, k, k)
        dxp = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=dcols.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + Ho, j:j + Wo] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)

        self.dinputs = dxp[:, :, p:p + H, p:p + W] if p else dxp

    def update_params(self):
        self.weights = self.opt_weights.update(self.weights, self.dweight)
        self.biases = self.opt_biases.update(self.biases, self.dbiases)


class Layer_MaxPool2D:
    """2x2 max pooling, stride 2. H and W must be even."""

    def forward(self, x):
        N, C, H, W = x.shape
        xr = x.reshape(N, C, H // 2, 2, W // 2, 2)
        out = xr.max(axis=(3, 5))

        self.xr = xr
        self.pooled = out
        self.x_shape = x.shape
        self.output = out

    def backward(self, dvalues):
        # Route each gradient to the max position(s) in its 2x2 window.
        # If several positions tie, the gradient is split evenly between them.
        mask = self.xr == self.pooled[:, :, :, None, :, None]
        count = mask.sum(axis=(3, 5), keepdims=True).astype(F32)
        dx = mask * (dvalues[:, :, :, None, :, None] / count)
        self.dinputs = dx.reshape(self.x_shape).astype(F32, copy=False)


class Layer_Flatten:
    def forward(self, x):
        self.x_shape = x.shape
        self.output = x.reshape(x.shape[0], -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.x_shape)


class Layer_Dropout:
    """Inverted dropout: scales at train time so inference needs no change."""

    def __init__(self, rate=0.3):
        self.keep = 1.0 - rate

    def forward(self, x, training=True):
        if not training or self.keep >= 1.0:
            self.mask = None
            self.output = x
            return
        self.mask = (np.random.rand(*x.shape) < self.keep).astype(F32) / F32(self.keep)
        self.output = x * self.mask

    def backward(self, dvalues):
        self.dinputs = dvalues if self.mask is None else dvalues * self.mask