"""
pc_torch.py — PyTorch Predictive Coding vs backprop MLP.

Both networks solve 3-class Gaussian blob classification.
PC uses only local Hebbian updates (no backprop through activities).
MLP uses standard backprop (Adam) as a baseline.

Same network shape: 2 → 32 → 16 → 3

Run: python3 pc_torch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)
DEVICE = torch.device("cpu")


# ── Dataset: 3 well-separated Gaussian blobs ──────────────────────────────────

def make_blobs(n_per_class=150, noise=0.5):
    angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    radius = 3.0
    Xs, Ys = [], []
    for c, theta in enumerate(angles):
        cx = radius * math.cos(theta)
        cy = radius * math.sin(theta)
        x = torch.randn(n_per_class, 2) * noise + torch.tensor([cx, cy])
        y = torch.full((n_per_class,), c, dtype=torch.long)
        Xs.append(x); Ys.append(y)
    X = torch.cat(Xs); Y = torch.cat(Ys)
    perm = torch.randperm(len(X))
    return X[perm], Y[perm]


X_all, Y_all = make_blobs(150, noise=0.5)
split = int(0.8 * len(X_all))
X_tr, Y_tr = X_all[:split].to(DEVICE), Y_all[:split].to(DEVICE)
X_te, Y_te = X_all[split:].to(DEVICE), Y_all[split:].to(DEVICE)

N_CLASSES = 3


# ── Predictive Coding Network ──────────────────────────────────────────────────

class PCNet:
    """
    Manually implemented PC network — no autograd for weight updates.

    W[l] shape: (sizes[l], sizes[l+1])
    Prediction at layer l:  ŷ_l = relu(x[l+1]) @ W[l].T
    Error:                   ε_l = x[l] - ŷ_l
    Activity update:         Δx_l = α(-ε_l + ε_{l-1} @ W[l-1])
    Weight update (Hebbian): ΔW_l ∝ ε_l^T · relu(x[l+1])
    """

    def __init__(self, sizes, alpha=0.05, beta=0.04, relax_steps=50):
        self.sizes = sizes
        self.L = len(sizes)
        self.alpha = alpha
        self.beta = beta
        self.relax_steps = relax_steps
        self.W = [
            torch.randn(sizes[l], sizes[l + 1], device=DEVICE) * 0.1
            for l in range(self.L - 1)
        ]

    def _relu(self, z):
        return F.relu(z)

    def _init(self, batch):
        self.x   = [torch.zeros(batch, s, device=DEVICE) for s in self.sizes]
        self.eps = [torch.zeros(batch, s, device=DEVICE) for s in self.sizes]

    def _errors(self):
        for l in range(self.L - 1):
            pred = self._relu(self.x[l + 1]) @ self.W[l].T
            self.eps[l] = self.x[l] - pred
        self.eps[-1].zero_()

    def _step(self, clamp_in, clamp_out):
        for l in range(self.L):
            delta = -self.eps[l]
            if l > 0:
                delta = delta + self.eps[l - 1] @ self.W[l - 1]
            self.x[l] = self.x[l] + self.alpha * delta
        self.x[0] = clamp_in.clone()
        if clamp_out is not None:
            self.x[-1] = clamp_out.clone()
        for l in range(1, self.L - 1):
            self.x[l] = self._relu(self.x[l])

    @torch.no_grad()
    def train_batch(self, X, Y_oh):
        self._init(X.shape[0])
        self.x[0] = X.clone()
        self.x[-1] = Y_oh.clone()
        for _ in range(self.relax_steps):
            self._errors()
            self._step(X, Y_oh)
        self._errors()
        for l in range(self.L - 1):
            pre = self._relu(self.x[l + 1])
            dW  = self.eps[l].T @ pre / X.shape[0]
            self.W[l] = self.W[l] + self.beta * dW

    @torch.no_grad()
    def predict(self, X):
        self._init(X.shape[0])
        self.x[0] = X.clone()
        for _ in range(self.relax_steps):
            self._errors()
            self._step(X, None)
        return self.x[-1]

    def accuracy(self, X, Y):
        return (self.predict(X).argmax(1) == Y).float().mean().item()


# ── Backprop MLP baseline ─────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


mlp     = MLP(N_CLASSES).to(DEVICE)
mlp_opt = torch.optim.Adam(mlp.parameters(), lr=5e-3)

pc = PCNet([2, 32, 16, N_CLASSES], alpha=0.05, beta=0.04, relax_steps=50)

EPOCHS = 120
BATCH  = 48

print("3-class blob classification | network: 2→32→16→3")
print("PC uses local Hebbian rule only — no backpropagation.")
print()
print(f"{'Epoch':>6} | {'PC Train':>9} | {'PC Test':>8} | {'MLP Train':>10} | {'MLP Test':>9}")
print("-------+-----------+----------+------------+-----------")

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(len(X_tr))

    for i in range(0, len(X_tr), BATCH):
        idx = perm[i : i + BATCH]
        xb, yb = X_tr[idx], Y_tr[idx]

        # PC: local Hebbian
        pc.train_batch(xb, F.one_hot(yb, N_CLASSES).float())

        # MLP: backprop
        mlp_opt.zero_grad()
        F.cross_entropy(mlp(xb), yb).backward()
        mlp_opt.step()

    if epoch % 20 == 0:
        pc_tr  = pc.accuracy(X_tr, Y_tr)
        pc_te  = pc.accuracy(X_te, Y_te)
        with torch.no_grad():
            mlp_tr = (mlp(X_tr).argmax(1) == Y_tr).float().mean().item()
            mlp_te = (mlp(X_te).argmax(1) == Y_te).float().mean().item()
        print(f"{epoch:6d} | {pc_tr:9.3f} | {pc_te:8.3f} | {mlp_tr:10.3f} | {mlp_te:9.3f}")

print()
print("PC  = local Hebbian weight updates (no backpropagation through layers)")
print("MLP = Adam + standard backpropagation (baseline)")
