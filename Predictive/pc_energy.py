"""
pc_energy.py — Visualize energy minimization during PC relaxation.

Shows how total prediction error (energy E = Σ ||ε_l||²) decays
across relaxation steps, for inference vs. supervised modes.
Also demonstrates how alpha (activity update rate) affects convergence.

No matplotlib required — uses ASCII plots + numeric tables.

Run: python3 pc_energy.py
"""

import numpy as np

np.random.seed(7)


def relu(z):
    return np.maximum(0, z)


class PCNet:
    def __init__(self, sizes, alpha=0.08):
        self.L = len(sizes)
        self.sizes = sizes
        self.alpha = alpha
        # Pre-trained weights: use identity-like structure so network
        # has some signal to work with (not purely random)
        self.W = []
        for l in range(self.L - 1):
            W = np.random.randn(sizes[l], sizes[l + 1]) * 0.15
            self.W.append(W)

    def relax(self, X, Y=None, steps=200):
        B = X.shape[0]
        x = [np.random.randn(B, s) * 0.01 for s in self.sizes]
        x[0] = X.copy()
        if Y is not None:
            x[-1] = Y.copy()

        energies = []
        for step in range(steps):
            eps = []
            for l in range(self.L - 1):
                pred = relu(x[l + 1]) @ self.W[l].T
                eps.append(x[l] - pred)
            eps.append(np.zeros_like(x[-1]))

            E = sum(np.sum(e ** 2) for e in eps)
            energies.append(float(E))

            for l in range(self.L):
                delta = -eps[l]
                if l > 0:
                    delta = delta + eps[l - 1] @ self.W[l - 1]
                x[l] = x[l] + self.alpha * delta

            x[0] = X.copy()
            if Y is not None:
                x[-1] = Y.copy()

            for l in range(1, self.L - 1):
                x[l] = relu(x[l])

        return energies


def ascii_plot(curves, labels, width=64, height=18):
    """Minimal ASCII line plot — no dependencies."""
    all_vals = [v for c in curves for v in c]
    ymin, ymax = min(all_vals), max(all_vals)
    yrange = ymax - ymin + 1e-9
    steps = len(curves[0])
    chars = ["*", "o", "#", "+"]

    canvas = [[" "] * width for _ in range(height)]
    for ci, (curve, ch) in enumerate(zip(curves, chars)):
        for xi, val in enumerate(curve):
            col = int((xi / (steps - 1)) * (width - 1))
            row = height - 1 - int(((val - ymin) / yrange) * (height - 1))
            row = max(0, min(height - 1, row))
            canvas[row][col] = ch

    tick_step = height // 4
    print(f"  Energy")
    for r, row in enumerate(canvas):
        if r % tick_step == 0:
            tick_val = ymax - (r / (height - 1)) * yrange
            prefix = f"{tick_val:8.1f} |"
        else:
            prefix = "         |"
        print(prefix + "".join(row))
    print("         +" + "-" * width)
    print(f"         Step 1{' ' * (width - 12)}Step {steps}")
    print()
    for ch, label in zip(chars, labels):
        print(f"  {ch}  {label}")


# ── Main experiment ────────────────────────────────────────────────────────────

STEPS = 120
B     = 64

# Use a 4→24→12→4 network with random initial weights
net_demo = PCNet([4, 24, 12, 4], alpha=0.10)

X = np.random.randn(B, 4) * 1.5
Y = np.random.randn(B, 4) * 1.5

free_e    = net_demo.relax(X, Y=None, steps=STEPS)
clamped_e = net_demo.relax(X, Y=Y,   steps=STEPS)

print("=" * 70)
print("  Predictive Coding — Energy Minimization During Relaxation")
print("=" * 70)
print()
print("Network: 4→24→12→4  |  batch=64  |  α=0.10")
print()

ascii_plot(
    [free_e, clamped_e],
    [
        "Free inference (input clamped, output free)",
        "Supervised  (input AND target clamped)",
    ],
)

# ── Numeric table ─────────────────────────────────────────────────────────────
print()
print(f"{'Step':>6} | {'Free E':>10} | {'Clamped E':>10} | {'Ratio C/F':>10}")
print("-------+------------+------------+------------")
pct_drop_free    = 100 * (1 - free_e[-1] / free_e[0])
pct_drop_clamped = 100 * (1 - clamped_e[-1] / clamped_e[0])
for s in [0, 4, 9, 19, 49, STEPS - 1]:
    ratio = clamped_e[s] / (free_e[s] + 1e-9)
    print(f"{s+1:6d} | {free_e[s]:10.3f} | {clamped_e[s]:10.3f} | {ratio:10.3f}")

print()
print(f"Free   energy drop over {STEPS} steps: {pct_drop_free:.1f}%")
print(f"Clamped energy drop over {STEPS} steps: {pct_drop_clamped:.1f}%")
print()
print("Clamped/Free > 1 → target constraint adds extra error the network")
print("must resolve via activity updates — this gradient drives learning.")

# ── Alpha sweep: convergence speed ────────────────────────────────────────────

print()
print("=" * 50)
print("  Effect of α (activity update rate) on convergence")
print("  Metric: energy at step 100 as % of initial energy")
print("=" * 50)

X2 = np.random.randn(32, 4)
Y2 = np.random.randn(32, 4)

for alpha in [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]:
    net2 = PCNet([4, 24, 12, 4], alpha=alpha)
    e = net2.relax(X2, Y=Y2, steps=100)
    if e[0] > 0:
        final_pct = 100 * e[-1] / e[0]
    else:
        final_pct = 100.0
    bar_len = max(1, int((100 - final_pct) / 2))  # longer = more convergence
    bar = "#" * bar_len
    print(f"  α={alpha:.2f}: {final_pct:5.1f}% of E₀ remains  [{bar:<50}]")

print()
print("Lower residual % = faster / deeper convergence within 100 steps.")
print("Very high α can cause oscillations (energy doesn't monotonically decrease).")
