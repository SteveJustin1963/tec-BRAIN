"""
pc_numpy.py — Predictive Coding network in pure NumPy.

Trains on 2D Gaussian blob classification (two classes).
Prints accuracy each epoch and shows energy decreasing over relaxation.

Run: python3 pc_numpy.py
"""

import numpy as np

np.random.seed(42)


class PCNetwork:
    """
    Hierarchical Predictive Coding network.

    Architecture: input layer (clamped) → hidden layers → output layer.
    Each layer has representational neurons (x) and error neurons (ε).

    Weight update rule (local Hebbian):
        ΔW_l ∝ ε_l^T · f(x_{l+1})
    where ε_l = x_l - f(x_{l+1}) @ W_l^T  (prediction error at layer l)
    and f is ReLU applied to hidden layers.
    """

    def __init__(self, layer_sizes, alpha=0.05, beta=0.05, relax_steps=80):
        self.L = len(layer_sizes)
        self.alpha = alpha          # activity update rate (inference)
        self.beta = beta            # weight learning rate (Hebbian)
        self.relax_steps = relax_steps
        self.sizes = layer_sizes

        # W[l] has shape (sizes[l], sizes[l+1]):
        # top-down: prediction for layer l = relu(x[l+1]) @ W[l].T
        self.W = [
            np.random.randn(layer_sizes[l], layer_sizes[l + 1]) * 0.1
            for l in range(self.L - 1)
        ]

    def _relu(self, z):
        return np.maximum(0.0, z)

    def _init(self, batch_size):
        self.x = [np.zeros((batch_size, s)) for s in self.sizes]
        self.eps = [np.zeros((batch_size, s)) for s in self.sizes]

    def _compute_errors(self):
        for l in range(self.L - 1):
            pred = self._relu(self.x[l + 1]) @ self.W[l].T
            self.eps[l] = self.x[l] - pred
        self.eps[-1][:] = 0.0

    def _update_activities(self, clamp_in, clamp_out):
        for l in range(self.L):
            delta = -self.eps[l]
            if l > 0:
                delta = delta + self.eps[l - 1] @ self.W[l - 1]
            self.x[l] = self.x[l] + self.alpha * delta

        # enforce clamps
        self.x[0] = clamp_in.copy()
        if clamp_out is not None:
            self.x[-1] = clamp_out.copy()

        # nonlinearity on hidden layers only
        for l in range(1, self.L - 1):
            self.x[l] = self._relu(self.x[l])

    def energy(self):
        return sum(np.sum(e ** 2) for e in self.eps)

    def train(self, X, Y):
        """Relaxation + Hebbian weight update on a batch."""
        B = X.shape[0]
        self._init(B)
        self.x[0] = X.copy()
        self.x[-1] = Y.copy()

        for _ in range(self.relax_steps):
            self._compute_errors()
            self._update_activities(X, Y)

        self._compute_errors()

        # Hebbian update: ΔW_l ∝ ε_l^T · relu(x_{l+1})
        for l in range(self.L - 1):
            dW = self.eps[l].T @ self._relu(self.x[l + 1]) / B
            self.W[l] += self.beta * dW

        return self.energy()

    def predict(self, X):
        """Inference: clamp input, let output evolve freely."""
        B = X.shape[0]
        self._init(B)
        self.x[0] = X.copy()

        for _ in range(self.relax_steps):
            self._compute_errors()
            self._update_activities(X, None)

        return self.x[-1]


# ── Dataset: 2D Gaussian blobs (3 classes) ────────────────────────────────────

def make_blobs(n_per_class=120):
    centers = np.array([[3.0, 0.0], [-1.5, 2.6], [-1.5, -2.6]])
    X_list, Y_list = [], []
    for c, center in enumerate(centers):
        X_list.append(np.random.randn(n_per_class, 2) * 0.6 + center)
        label = np.zeros((n_per_class, 3))
        label[:, c] = 1.0
        Y_list.append(label)
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]


X_all, Y_all = make_blobs(n_per_class=120)
split = int(0.8 * len(X_all))
X_train, Y_train = X_all[:split], Y_all[:split]
X_test,  Y_test  = X_all[split:], Y_all[split:]

net = PCNetwork(
    layer_sizes=[2, 32, 16, 3],
    alpha=0.05,
    beta=0.05,
    relax_steps=60,
)

EPOCHS = 150
BATCH  = 32

print("3-class blob classification | network: 2→32→16→3")
print(f"{'Epoch':>6} | {'Train Acc':>9} | {'Test Acc':>8} | {'Energy':>10}")
print("-------+-----------+----------+----------")

for epoch in range(1, EPOCHS + 1):
    perm = np.random.permutation(len(X_train))
    total_E = 0.0

    for i in range(0, len(X_train), BATCH):
        idx = perm[i : i + BATCH]
        total_E += net.train(X_train[idx], Y_train[idx])

    if epoch % 15 == 0:
        tr_preds = net.predict(X_train).argmax(axis=1)
        te_preds = net.predict(X_test).argmax(axis=1)
        tr_acc = (tr_preds == Y_train.argmax(axis=1)).mean()
        te_acc = (te_preds == Y_test.argmax(axis=1)).mean()
        print(f"{epoch:6d} | {tr_acc:9.3f} | {te_acc:8.3f} | {total_E:10.2f}")

print("\nDone.")
