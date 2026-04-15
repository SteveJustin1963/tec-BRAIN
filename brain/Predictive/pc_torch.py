"""
pc_torch.py — PyTorch Predictive Coding vs backprop MLP.

Compares PC (local Hebbian only) vs MLP (Adam + backprop) across
four increasing dataset difficulties:

  1. Baseline:      3 classes, noise=0.5  (well-separated)
  2. High overlap:  3 classes, noise=1.5  (strongly overlapping)
  3. More classes:  6 classes, noise=0.5  (larger output space)
  4. Hard:          6 classes, noise=1.5  (both harder)

Network shape: 2 → 64 → 32 → N_CLASSES

Run: python3 pc_torch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE = torch.device("cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_blobs(n_classes=3, n_per_class=200, noise=0.5, seed=42):
    """Evenly-spaced Gaussian blobs on a circle of radius 3."""
    torch.manual_seed(seed)
    radius = 3.0
    Xs, Ys = [], []
    for c in range(n_classes):
        theta = 2 * math.pi * c / n_classes
        cx, cy = radius * math.cos(theta), radius * math.sin(theta)
        x = torch.randn(n_per_class, 2) * noise + torch.tensor([cx, cy])
        y = torch.full((n_per_class,), c, dtype=torch.long)
        Xs.append(x); Ys.append(y)
    X = torch.cat(Xs); Y = torch.cat(Ys)
    perm = torch.randperm(len(X))
    return X[perm], Y[perm]


# ── Predictive Coding Network ──────────────────────────────────────────────────

class PCNet:
    """
    PC network with purely local Hebbian weight updates.

    W[l] shape: (sizes[l], sizes[l+1])
    Prediction:   pred_l = relu(x[l+1]) @ W[l].T
    Error:        eps_l  = x[l] - pred_l
    Activity:     Δx_l   = α(-eps_l + eps_{l-1} @ W[l-1])
    Weight:       ΔW_l   ∝ eps_l.T @ relu(x[l+1])
    """

    def __init__(self, sizes, alpha=0.05, beta=0.02, relax_steps=50):
        self.sizes = sizes
        self.L = len(sizes)
        self.alpha = alpha
        self.beta = beta
        self.relax_steps = relax_steps
        self.W = [
            torch.randn(sizes[l], sizes[l + 1], device=DEVICE) * 0.1
            for l in range(self.L - 1)
        ]

    def _errors(self):
        for l in range(self.L - 1):
            self.eps[l] = self.x[l] - F.relu(self.x[l + 1]) @ self.W[l].T
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
            self.x[l] = F.relu(self.x[l])

    def _init(self, batch):
        self.x   = [torch.zeros(batch, s, device=DEVICE) for s in self.sizes]
        self.eps = [torch.zeros(batch, s, device=DEVICE) for s in self.sizes]

    @torch.no_grad()
    def train_batch(self, X, Y_oh):
        self._init(X.shape[0])
        self.x[0] = X.clone(); self.x[-1] = Y_oh.clone()
        for _ in range(self.relax_steps):
            self._errors(); self._step(X, Y_oh)
        self._errors()
        for l in range(self.L - 1):
            dW = self.eps[l].T @ F.relu(self.x[l + 1]) / X.shape[0]
            self.W[l] = self.W[l] + self.beta * dW

    @torch.no_grad()
    def predict(self, X):
        self._init(X.shape[0])
        self.x[0] = X.clone()
        for _ in range(self.relax_steps):
            self._errors(); self._step(X, None)
        return self.x[-1]

    def accuracy(self, X, Y):
        return (self.predict(X).argmax(1) == Y).float().mean().item()


# ── Backprop MLP baseline ──────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Experiment runner ──────────────────────────────────────────────────────────

EPOCHS     = 150
BATCH      = 64
THRESHOLD  = 0.95   # "reached" accuracy for summary


def run_experiment(label, n_classes, noise):
    torch.manual_seed(0)

    X_all, Y_all = make_blobs(n_classes=n_classes, n_per_class=200, noise=noise)
    split = int(0.8 * len(X_all))
    X_tr, Y_tr = X_all[:split].to(DEVICE), Y_all[:split].to(DEVICE)
    X_te, Y_te = X_all[split:].to(DEVICE), Y_all[split:].to(DEVICE)

    pc  = PCNet([2, 64, 32, n_classes], alpha=0.05, beta=0.02, relax_steps=50)
    mlp = MLP(n_classes).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=5e-3)

    pc_first = mlp_first = None   # epoch first crossing THRESHOLD on test set

    print(f"\n{'─'*65}")
    print(f"  {label}  |  {n_classes} classes, noise={noise}")
    print(f"{'─'*65}")
    print(f"{'Epoch':>6} | {'PC train':>9} {'PC test':>8} | {'MLP train':>10} {'MLP test':>9}")
    print(f"{'':─>6}-+-{'':─>18}-+-{'':─>20}")

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), BATCH):
            idx = perm[i : i + BATCH]
            xb, yb = X_tr[idx], Y_tr[idx]

            pc.train_batch(xb, F.one_hot(yb, n_classes).float())

            opt.zero_grad()
            F.cross_entropy(mlp(xb), yb).backward()
            opt.step()

        if epoch % 15 == 0 or epoch == 1:
            pc_tr  = pc.accuracy(X_tr, Y_tr)
            pc_te  = pc.accuracy(X_te, Y_te)
            with torch.no_grad():
                mlp_tr = (mlp(X_tr).argmax(1) == Y_tr).float().mean().item()
                mlp_te = (mlp(X_te).argmax(1) == Y_te).float().mean().item()
            print(f"{epoch:6d} | {pc_tr:9.3f} {pc_te:8.3f} | {mlp_tr:10.3f} {mlp_te:9.3f}")

        # Track first epoch crossing threshold
        with torch.no_grad():
            pc_te_now  = pc.accuracy(X_te, Y_te)
            mlp_te_now = (mlp(X_te).argmax(1) == Y_te).float().mean().item()
        if pc_first  is None and pc_te_now  >= THRESHOLD: pc_first  = epoch
        if mlp_first is None and mlp_te_now >= THRESHOLD: mlp_first = epoch

    # Final accuracies
    pc_final  = pc.accuracy(X_te, Y_te)
    with torch.no_grad():
        mlp_final = (mlp(X_te).argmax(1) == Y_te).float().mean().item()

    print(f"\n  First epoch ≥{THRESHOLD:.0%} test acc:")
    print(f"    PC  → {'epoch '+str(pc_first)  if pc_first  else 'never (<'+str(THRESHOLD)+')'}")
    print(f"    MLP → {'epoch '+str(mlp_first) if mlp_first else 'never (<'+str(THRESHOLD)+')'}")
    print(f"  Final test accuracy:  PC={pc_final:.3f}   MLP={mlp_final:.3f}")

    return dict(label=label, n_classes=n_classes, noise=noise,
                pc_first=pc_first, mlp_first=mlp_first,
                pc_final=pc_final, mlp_final=mlp_final)


# ── Run all experiments ────────────────────────────────────────────────────────

configs = [
    ("Baseline",     3, 0.5),
    ("High overlap", 3, 1.5),
    ("More classes", 6, 0.5),
    ("Hard",         6, 1.5),
]

print("PC (Hebbian) vs MLP (Adam+backprop) — scaling difficulty")
print("Network: 2 → 64 → 32 → N_classes")
print(f"Threshold for 'reached': {THRESHOLD:.0%} test accuracy")

results = [run_experiment(lbl, nc, ns) for lbl, nc, ns in configs]

# ── Summary table ──────────────────────────────────────────────────────────────

print("\n\n" + "═"*70)
print("SUMMARY — first epoch reaching ≥95% test accuracy (None = never)")
print("═"*70)
print(f"{'Experiment':<16} | {'Classes':>7} | {'Noise':>5} | {'PC first':>9} | {'MLP first':>9} | {'PC final':>8} | {'MLP final':>9}")
print(f"{'':─<16}-+-{'':─>7}-+-{'':─>5}-+-{'':─>9}-+-{'':─>9}-+-{'':─>8}-+-{'':─>9}")
for r in results:
    def cell(v): return str(v) if v else "—"
    print(f"{r['label']:<16} | {r['n_classes']:>7} | {r['noise']:>5.1f} | "
          f"{cell(r['pc_first']):>9} | {cell(r['mlp_first']):>9} | "
          f"{r['pc_final']:>8.3f} | {r['mlp_final']:>9.3f}")
print()
print("PC  = local Hebbian updates only — no backpropagation through layers")
print("MLP = Adam + standard backpropagation (baseline)")
