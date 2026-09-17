"""A small CNN implemented from scratch in NumPy: conv -> ReLU -> maxpool ->
dense -> ReLU -> dense -> sigmoid, trained with backprop + Adam.

Same rationale as Stage II's `gru_numpy.py`: this sandbox has no network
access, so TensorFlow/PyTorch (and any transfer-learning backbone) could not
be installed. Given the 100-image synthetic dataset is small, a single
small conv layer is an appropriately lightweight architecture rather than
inventing something deeper the data cannot support.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _im2col(x, kh, kw):
    """x: (N, H, W). Returns (N, out_h, out_w, kh*kw) patches, valid padding, stride 1."""
    n, h, w = x.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    shape = (n, out_h, out_w, kh, kw)
    strides = (x.strides[0], x.strides[1], x.strides[2], x.strides[1], x.strides[2])
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return patches.reshape(n, out_h, out_w, kh * kw)


class SimpleCNNBinaryClassifier:
    def __init__(self, img_size: int = 32, n_filters: int = 4, kernel: int = 3,
                 hidden_size: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.img_size = img_size
        self.n_filters = n_filters
        self.kernel = kernel
        self.hidden_size = hidden_size

        conv_out = img_size - kernel + 1
        pool_out = conv_out // 2
        self.pool_out = pool_out
        flat_size = pool_out * pool_out * n_filters

        def init(shape, fan_in):
            limit = np.sqrt(6.0 / (fan_in + np.prod(shape[1:]) if hasattr(shape, "__len__") else fan_in))
            return rng.uniform(-limit, limit, size=shape)

        self.conv_w = rng.uniform(-0.3, 0.3, size=(n_filters, kernel, kernel))
        self.conv_b = np.zeros(n_filters)
        self.W1 = init((flat_size, hidden_size), flat_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = init((hidden_size, 1), hidden_size)
        self.b2 = np.zeros(1)

        self.params = ["conv_w", "conv_b", "W1", "b1", "W2", "b2"]
        self._init_adam()

    def _init_adam(self):
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def forward(self, X):
        """X: (N, H, W) grayscale in [0,1]. Returns probs (N,), cache."""
        n = X.shape[0]
        patches = _im2col(X, self.kernel, self.kernel)  # (N, oh, ow, k*k)
        oh, ow = patches.shape[1], patches.shape[2]
        w_flat = self.conv_w.reshape(self.n_filters, -1)  # (F, k*k)
        conv_out = np.tensordot(patches, w_flat, axes=([3], [1]))  # (N, oh, ow, F)
        conv_out = conv_out + self.conv_b
        relu_out = np.maximum(conv_out, 0)

        # 2x2 max pool, stride 2.
        ph, pw = oh // 2, ow // 2
        cropped = relu_out[:, :ph * 2, :pw * 2, :]
        reshaped = cropped.reshape(n, ph, 2, pw, 2, self.n_filters)
        pooled = reshaped.max(axis=(2, 4))  # (N, ph, pw, F)
        # Bring the two size-2 sub-block axes adjacent before flattening to 4,
        # so the argmax index correctly decodes back to (subrow, subcol).
        transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)  # (N, ph, pw, subrow, subcol, F)
        flat4 = transposed.reshape(n, ph, pw, 4, self.n_filters)
        pool_argmax = flat4.argmax(axis=3)

        flat = pooled.reshape(n, -1)
        z1 = flat @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        logits = a1 @ self.W2 + self.b2
        probs = sigmoid(logits).ravel()

        cache = {
            "X": X, "patches": patches, "conv_out": conv_out, "relu_out": relu_out,
            "pool_argmax": pool_argmax, "flat": flat, "z1": z1, "a1": a1, "probs": probs,
            "oh": oh, "ow": ow, "ph": ph, "pw": pw,
        }
        return probs, cache

    def backward(self, cache, y, sample_weight=None):
        n = cache["X"].shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n)
        norm = sample_weight.sum() if sample_weight.sum() > 0 else n

        grads = {p: np.zeros_like(getattr(self, p)) for p in self.params}

        dlogits = (sample_weight * (cache["probs"] - y)).reshape(-1, 1) / norm
        grads["W2"] = cache["a1"].T @ dlogits
        grads["b2"] = dlogits.sum(axis=0)
        da1 = dlogits @ self.W2.T
        dz1 = da1 * (cache["z1"] > 0)
        grads["W1"] = cache["flat"].T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        dflat = dz1 @ self.W1.T

        ph, pw = cache["ph"], cache["pw"]
        dpooled = dflat.reshape(n, ph, pw, self.n_filters)

        # Route pooled gradient back to the argmax position in each 2x2 window.
        drelu_cropped = np.zeros((n, ph, 2, pw, 2, self.n_filters))
        argmax = cache["pool_argmax"]  # (n, ph, pw, F) values in [0,3]
        idx_n, idx_ph, idx_pw, idx_f = np.meshgrid(
            np.arange(n), np.arange(ph), np.arange(pw), np.arange(self.n_filters), indexing="ij"
        )
        row = argmax // 2
        col = argmax % 2
        drelu_cropped[idx_n, idx_ph, row, idx_pw, col, idx_f] = dpooled
        drelu_cropped = drelu_cropped.reshape(n, ph * 2, pw * 2, self.n_filters)

        oh, ow = cache["oh"], cache["ow"]
        drelu = np.zeros((n, oh, ow, self.n_filters))
        drelu[:, :ph * 2, :pw * 2, :] = drelu_cropped
        dconv = drelu * (cache["conv_out"] > 0)

        grads["conv_b"] = dconv.sum(axis=(0, 1, 2))
        patches = cache["patches"]  # (n, oh, ow, k*k)
        dconv_flat = dconv.reshape(n, oh * ow, self.n_filters)  # (n, oh*ow, F)
        patches_flat = patches.reshape(n, oh * ow, -1)  # (n, oh*ow, k*k)
        dw_flat = np.einsum("npf,npk->fk", dconv_flat, patches_flat)
        grads["conv_w"] = dw_flat.reshape(self.n_filters, self.kernel, self.kernel)

        for p in self.params:
            np.clip(grads[p], -5, 5, out=grads[p])
        return grads

    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def save(self, path):
        np.savez(path, **{p: getattr(self, p) for p in self.params},
                 img_size=self.img_size, n_filters=self.n_filters,
                 kernel=self.kernel, hidden_size=self.hidden_size)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["img_size"]), int(data["n_filters"]),
                     int(data["kernel"]), int(data["hidden_size"]))
        for p in model.params:
            setattr(model, p, data[p])
        return model


def bce_loss(probs, y, eps=1e-9):
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
