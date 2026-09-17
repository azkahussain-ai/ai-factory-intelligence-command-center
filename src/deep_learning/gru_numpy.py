"""A small GRU (Gated Recurrent Unit) implemented from scratch in NumPy.

Why from scratch: this sandbox has no network access, and neither
TensorFlow/Keras nor PyTorch could be installed (see docs/HANDOFF.md).
Rather than substitute a non-recurrent model and mislabel it "deep
learning", this implements the actual GRU recurrence + backprop-through-time
+ Adam, so the sequence model is real, not simulated.

Architecture: single-layer GRU -> linear head -> sigmoid, binary
cross-entropy loss, trained with mini-batch gradient descent (Adam).
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class GRUBinaryClassifier:
    def __init__(self, n_features: int, hidden_size: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.hidden_size = hidden_size

        def init(shape):
            limit = np.sqrt(6.0 / sum(shape))
            return rng.uniform(-limit, limit, size=shape)

        h = hidden_size
        d = n_features
        # Update gate z, reset gate r, candidate hidden n -- concat[x, h] -> h
        self.Wz = init((d + h, h)); self.bz = np.zeros(h)
        self.Wr = init((d + h, h)); self.br = np.zeros(h)
        self.Wn = init((d + h, h)); self.bn = np.zeros(h)
        # Output head
        self.Wo = init((h, 1)); self.bo = np.zeros(1)

        self.params = ["Wz", "bz", "Wr", "br", "Wn", "bn", "Wo", "bo"]
        self._init_adam()

    def _init_adam(self):
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def forward(self, X):
        """X: (batch, seq_len, n_features) -> probs (batch,), cache for backward."""
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.hidden_size))
        cache = {"x": X, "h": [h.copy()], "z": [], "r": [], "n": []}
        for t in range(seq_len):
            xt = X[:, t, :]
            concat = np.concatenate([xt, h], axis=1)
            z = sigmoid(concat @ self.Wz + self.bz)
            r = sigmoid(concat @ self.Wr + self.br)
            concat_r = np.concatenate([xt, r * h], axis=1)
            n = np.tanh(concat_r @ self.Wn + self.bn)
            h = (1 - z) * n + z * h
            cache["z"].append(z); cache["r"].append(r); cache["n"].append(n)
            cache["h"].append(h.copy())
        logits = h @ self.Wo + self.bo
        probs = sigmoid(logits).ravel()
        cache["probs"] = probs
        return probs, cache

    def backward(self, cache, y, sample_weight=None):
        """BPTT for (optionally weighted) binary cross-entropy loss.

        sample_weight: per-example weights (e.g. to upweight the rare
        positive class); defaults to uniform weight of 1 per example.
        Returns grads dict.
        """
        X = cache["x"]
        batch, seq_len, d = X.shape
        h_seq = cache["h"]  # h_seq[t] is hidden state after step t (h_seq[0] = initial zeros)
        probs = cache["probs"]
        if sample_weight is None:
            sample_weight = np.ones(batch)

        grads = {p: np.zeros_like(getattr(self, p)) for p in self.params}

        # Weighted mean gradient of BCE loss w.r.t. logits.
        norm = sample_weight.sum() if sample_weight.sum() > 0 else batch
        dlogits = (sample_weight * (probs - y)).reshape(-1, 1) / norm
        h_last = h_seq[-1]
        grads["Wo"] = h_last.T @ dlogits
        grads["bo"] = dlogits.sum(axis=0)
        dh_next = dlogits @ self.Wo.T  # (batch, hidden)

        for t in reversed(range(seq_len)):
            xt = X[:, t, :]
            h_prev = h_seq[t]
            z, r, n = cache["z"][t], cache["r"][t], cache["n"][t]

            dh = dh_next
            dn = dh * (1 - z) * (1 - n ** 2)
            dz = dh * (h_prev - n) * z * (1 - z)

            concat_r = np.concatenate([xt, r * h_prev], axis=1)
            grads["Wn"] += concat_r.T @ dn
            grads["bn"] += dn.sum(axis=0)
            dconcat_r = dn @ self.Wn.T
            dxt_from_n = dconcat_r[:, :d]
            drh = dconcat_r[:, d:]
            dr = drh * h_prev * r * (1 - r)

            concat = np.concatenate([xt, h_prev], axis=1)
            grads["Wz"] += concat.T @ dz
            grads["bz"] += dz.sum(axis=0)
            grads["Wr"] += concat.T @ dr
            grads["br"] += dr.sum(axis=0)

            dconcat_z = dz @ self.Wz.T
            dconcat_r2 = dr @ self.Wr.T
            dxt = dxt_from_n + dconcat_z[:, :d] + dconcat_r2[:, :d]
            dh_prev = (dh * z) + dconcat_z[:, d:] + dconcat_r2[:, d:] + drh * r

            dh_next = dh_prev
            # dxt currently unused beyond this layer (single-layer GRU, no input grad needed)

        # Gradient clipping for BPTT stability.
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
                 hidden_size=self.hidden_size, n_features=self.n_features)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["n_features"]), int(data["hidden_size"]))
        for p in model.params:
            setattr(model, p, data[p])
        return model


def bce_loss(probs, y, eps=1e-9):
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
