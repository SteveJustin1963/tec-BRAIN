# Predictive Coding — Code Manual

Source material: [Artem Kirsanov — "The Brain's Learning Algorithm Isn't Backpropagation"](https://www.youtube.com/watch?v=l-OLgbdZ3kk)

---

## Overview

This collection implements **Predictive Coding (PC)** — a biologically plausible
learning algorithm in which the brain learns by minimising prediction errors
rather than by propagating gradients backward through a network.

The core idea in one sentence:
> Every layer predicts the activity of the layer below it. The mismatch
> (prediction error) drives both belief updates and synaptic weight changes,
> entirely locally.

Four programs, each runnable independently:

| File | What it does | Dependencies |
|---|---|---|
| `pc_numpy.py` | PC network — classification, pure NumPy | numpy |
| `pc_torch.py` | PC vs backprop MLP — side-by-side comparison | torch |
| `pc_energy.py` | Energy minimisation visualised, ASCII plots | numpy |
| `pc_robot.py` | PC as robot perception + active inference control | numpy |

---

## Core Concepts

### The Energy Function

The network minimises a global energy (sum of squared prediction errors):

```
E = Σ_l  ||ε_l||²
```

where the prediction error at layer l is:

```
ε_l  =  x_l  −  f(x_{l+1}) @ W_l^T
```

- `x_l` — representational activity at layer l (a vector of neuron firing rates)
- `W_l` — feedback weight matrix from layer l+1 down to layer l, shape `(s_l, s_{l+1})`
- `f`   — nonlinearity (ReLU applied to hidden layers only)
- `ε_l` — how wrong the top-down prediction is; this is what error neurons carry

### Activity Update (Inference / Relaxation)

Activities evolve by gradient descent on E:

```
Δx_l  =  α · (−ε_l  +  ε_{l-1} @ W_{l-1})
```

- `−ε_l` pulls `x_l` toward what the layer above predicts it should be
- `ε_{l-1} @ W_{l-1}` is the bottom-up signal: errors from the layer below push `x_l` up
- This runs for `relax_steps` iterations until activities settle (equilibrium)

Input layer is **clamped** to data throughout. Output layer is **clamped** to
the target label during training, and **free** during inference.

### Weight Update (Learning — Local Hebbian Rule)

After activities settle, weights update using only local information:

```
ΔW_l  =  β · ε_l^T · f(x_{l+1})
```

- Pre-synaptic: `f(x_{l+1})` — what the higher layer is doing
- Post-synaptic: `ε_l` — how wrong the lower layer's prediction is
- No global gradient, no weight transport problem, no backward pass

This is **biologically plausible** because each synapse only needs to know
the activity of its two adjacent neurons.

### Two Phases

| Phase | Input clamped? | Output clamped? | Purpose |
|---|---|---|---|
| Training | Yes (to data) | Yes (to label) | Activities relax with target constraint → drives weight update |
| Inference | Yes (to data) | No (free) | Activities relax to find best explanation → read output layer |

---

## File Reference

---

### `pc_numpy.py` — Pure NumPy PC Network

**Run:** `python3 pc_numpy.py`

**Task:** 3-class Gaussian blob classification. Network: `2 → 32 → 16 → 3`.

**Class: `PCNetwork`**

```python
net = PCNetwork(layer_sizes, alpha, beta, relax_steps)
```

| Parameter | Default | Meaning |
|---|---|---|
| `layer_sizes` | `[2,32,16,3]` | Neuron counts per layer, input→output |
| `alpha` | `0.05` | Activity update rate (inference step size) |
| `beta` | `0.05` | Weight learning rate (Hebbian plasticity) |
| `relax_steps` | `80` | Iterations of activity update per batch |

**Key methods:**

```python
net.train(X, Y)     # One batch: relax activities (both ends clamped) → update W
                    # Returns total energy after relaxation

net.predict(X)      # Inference: clamp input, free output, relax → return x[-1]
```

**Internal flow of `train(X, Y)`:**

```
1. x[0] = X  (clamp input)
2. x[-1] = Y (clamp target)
3. for relax_steps:
       _compute_errors()      → ε_l = x_l − relu(x_{l+1}) @ W[l].T
       _update_activities()   → x_l += α(−ε_l + ε_{l-1} @ W[l-1])
4. _compute_errors()          → final residual errors
5. for each layer:
       W[l] += β · ε_l.T @ relu(x_{l+1]) / B    (Hebbian)
```

**Weights shape:** `W[l]` has shape `(sizes[l], sizes[l+1])`.
Top-down prediction: `relu(x[l+1]) @ W[l].T` gives shape `(B, sizes[l])`.

**Tuning tips:**

- If accuracy doesn't improve: increase `beta` (0.02–0.1 range is typical)
- If training is unstable/diverges: reduce `alpha` or `beta`
- If inference gives wrong class: increase `relax_steps`
- Energy should decrease epoch-over-epoch; if flat, `beta` is too small

**Expected output:**

```
3-class blob classification | network: 2→32→16→3
 Epoch | Train Acc | Test Acc |     Energy
-------+-----------+----------+----------
    15 |     1.000 |    1.000 |      33.49
    ...
```

---

### `pc_torch.py` — PyTorch PC vs Backprop MLP

**Run:** `python3 pc_torch.py`

**Task:** Same 3-class blob classification, same network shape `2 → 32 → 16 → 3`,
trained in parallel using two different algorithms.

**Class: `PCNet`** (not an `nn.Module` — weights managed manually)

```python
pc = PCNet(sizes, alpha, beta, relax_steps)
```

| Parameter | Default | Meaning |
|---|---|---|
| `sizes` | `[2,32,16,3]` | Layer sizes |
| `alpha` | `0.05` | Activity step size |
| `beta` | `0.04` | Hebbian weight learning rate |
| `relax_steps` | `50` | Relaxation iterations per batch |

**Key methods:**

```python
pc.train_batch(X, Y_onehot)   # X: (B,2) float, Y_onehot: (B,3) float one-hot
pc.predict(X)                  # Returns (B,3) output activities
pc.accuracy(X, Y)              # Y: (B,) long class indices → scalar float
```

**Class: `MLP`** (standard `nn.Module`, backprop baseline)

```python
mlp = MLP(n_classes)           # nn.Sequential: Linear→ReLU→Linear→ReLU→Linear
mlp_opt = torch.optim.Adam(mlp.parameters(), lr=5e-3)
```

**Why PCNet weights are NOT nn.Parameters:**
The Hebbian rule is applied manually via `.data` updates. Using `nn.Parameter`
would allow autograd to compute gradients, but the PC weight update intentionally
ignores the global loss gradient — it only uses local error × activity.

**Expected output:**

```
 Epoch |  PC Train |  PC Test |  MLP Train |  MLP Test
-------+-----------+----------+------------+-----------
    20 |     1.000 |    1.000 |      1.000 |     1.000
```

Both converge to 100% on this task, demonstrating that local Hebbian updates
are sufficient — backprop is not required.

---

### `pc_energy.py` — Energy Visualiser

**Run:** `python3 pc_energy.py`

**What it shows:** How total prediction error E = Σ||ε_l||² decays over
relaxation steps, for two modes:

- **Free inference** — input clamped, output free. Energy drops as the network
  finds the best internal explanation of the input.
- **Supervised** — input AND target clamped. Target adds an extra constraint,
  so initial energy is higher. The gap between the two curves is the "teaching
  signal" that drives weight updates during learning.

**Class: `PCNet`** (simplified, no weight update — only `relax()`)

```python
net = PCNet(sizes, alpha)
energies = net.relax(X, Y=None, steps=200)   # free inference
energies = net.relax(X, Y=Y,   steps=200)   # supervised
```

Returns a list of scalar energy values, one per relaxation step.

**Alpha sweep section:**
Shows how `alpha` (activity update rate) affects how quickly energy drops
within a fixed number of steps. Too low → slow convergence. Too high →
oscillations (energy not monotonically decreasing).

**Key insight from the ratio column:**

```
Clamped E / Free E > 1
```

The supervised mode always has higher energy because the target clamp
introduces an additional prediction error at the output layer. This extra
error is what propagates back through the network during weight updates —
it is the PC equivalent of the loss gradient in backprop.

---

### `pc_robot.py` — Robot Perception + Active Inference Control

**Run:** `python3 pc_robot.py`

**What it shows:** PC applied to a simulated 1-D robot on a track,
demonstrating perception (state estimation) and action (motor control)
as two faces of the same free-energy minimisation.

#### Class: `Robot1D` — Physics simulation

```python
robot = Robot1D(pos=0.0, sensor_noise=0.5, mass=1.0, drag=0.4)
robot.step(force, dt=0.1)   # apply force, update position and velocity
reading = robot.sense()      # return position + Gaussian noise
```

**Replace `sense()` and `step()` with real hardware drivers to use on an
actual robot.** Everything else stays the same.

Physics:
```
acc      = force / mass
velocity = velocity × (1 − drag) + acc × dt
position = position + velocity × dt
sensor   = position + N(0, noise²)
```

#### Class: `PCStateEstimator` — Perception via PC

```python
est = PCStateEstimator(init_pos, kappa_sensory, kappa_prior, steps)
belief = est.update(y_sensed, predicted_pos=None)
```

| Parameter | Default | Meaning |
|---|---|---|
| `kappa_sensory` | `2.0` | How much to trust sensor (precision of likelihood) |
| `kappa_prior` | `0.5` | How much to trust prior continuity |
| `steps` | `20` | Inference iterations per timestep |

**Generative model:**

```
belief μ is updated by gradient descent on variational free energy F:

    ε_sensory = y_sensed − μ          (sensor prediction error)
    ε_prior   = μ − μ₀                (deviation from prior)
    Δμ        = κ_s · ε_s − κ_p · ε_p
```

This is mathematically equivalent to a **Kalman filter** with fixed gains.
The ratio `kappa_sensory / kappa_prior` controls the tradeoff between
trusting sensors and trusting the model's continuity assumption.

**Effect on noise:**
With `kappa_sensory=2.0` and `kappa_prior=0.5`, the belief is a
precision-weighted average of the sensor reading and the prior.
In the default simulation this reduces sensor RMSE by ~18–22%.

#### Class: `ActiveInferenceController` — Motor commands via PC

```python
ctrl = ActiveInferenceController(goal, kappa_goal=3.0, max_force=5.0)
force = ctrl.command(believed_pos)
ctrl.set_goal(new_goal)          # switch target at runtime
```

**The equation:**

```
u = κ_goal · (goal − μ)
```

This looks like a proportional controller, but its derivation is different:
the robot acts to make its **sensory prediction come true**, not to correct
an error detected after the fact. In active inference, the goal is encoded
as a prior over expected proprioception; the motor command is what minimises
the prediction error between that prior and the current belief.

Biologically, this is the **gamma motor neuron / muscle spindle reflex arc**:
the brain sets a desired muscle length (efference copy), and the spinal cord
generates torque to make the actual length match.

#### Class: `PIDController` — Classical baseline

```python
pid = PIDController(goal, kp=3.0, ki=0.1, kd=0.5, max_force=5.0)
force = pid.command(measured_pos)   # uses raw noisy sensor directly
```

The PID takes the raw noisy sensor as input (no state estimator). Comparing
it with the PC robot shows the benefit of the Bayesian belief layer.

#### `run_simulation(goal, steps, sensor_noise)` — Main loop

Runs both robots for `steps` timesteps and prints a table + ASCII trajectory.

**Columns in the table:**

| Column | Meaning |
|---|---|
| `True (PC)` | Ground truth position of PC robot |
| `Belief` | PC state estimator's current belief μ |
| `Sensor` | Raw noisy sensor reading |
| `Force` | Motor command issued |
| `True (PID)` | Ground truth position of PID robot |
| `Err PC / PID` | Absolute distance from goal |

---

## How Everything Connects

```
pc_energy.py
  └── shows the relaxation dynamics that happen INSIDE every train() call

pc_numpy.py
  └── full PC network: relaxation + Hebbian weight update + classification

pc_torch.py
  └── same algorithm in PyTorch, compared directly with backprop MLP

pc_robot.py
  ├── PCStateEstimator  ←  uses the same prediction-error update as pc_numpy
  └── ActiveInferenceController  ←  applies prediction error as motor command
        (perception and action are two sides of the same coin)
```

The README's `ActiveInferenceAgent` (the full EFE-minimising agent) is the
next level up from `pc_robot.py` — it uses Expected Free Energy over future
timesteps to select among policies, rather than a single-step proportional
command.

---

## Quick-Start Hyperparameter Guide

| Parameter | Too low | Good range | Too high |
|---|---|---|---|
| `alpha` (activity rate) | Slow relaxation, inaccurate inference | 0.03–0.10 | Oscillations, energy increases |
| `beta` (weight rate) | No learning, flat accuracy | 0.02–0.08 | Weights explode, NaNs |
| `relax_steps` | Bad inference, poor accuracy | 50–150 | Slow training (diminishing returns) |
| `kappa_sensory` | Ignores sensor, belief stuck | 1.0–4.0 | Belief = raw noisy sensor |
| `kappa_prior` | Belief jumps with every reading | 0.2–1.0 | Belief never moves |
| `kappa_goal` | Slow robot, doesn't reach goal | 2.0–5.0 | Overshoots, oscillates |

---

## Adapting to a Real Robot

Swap out the two hardware-facing methods in `Robot1D`:

```python
# 1. Replace the sensor
def sense(self):
    return read_encoder()          # servo angle, lidar distance, IMU, etc.

# 2. Replace the actuator
def step(self, force, dt=0.1):
    set_motor_pwm(force)           # or send_torque(), set_velocity(), etc.
```

Everything else — `PCStateEstimator`, `ActiveInferenceController` — is pure
math and runs unchanged on any hardware.

**Extend the state vector for richer robots:**

```python
# 1-D position only (current)
estimator.mu  = 0.0

# position + velocity (add momentum prior)
estimator.mu  = np.array([pos, vel])
A_transition  = np.array([[1, dt],[0, 1-drag]])   # dynamics matrix
predicted_pos = A_transition @ estimator.mu        # pass to est.update()

# multi-joint arm: stack all joint angles into one vector
estimator.mu  = np.zeros(n_joints)
```

---

## Running All Programs

```bash
python3 pc_numpy.py    # ~10 s  — numpy only
python3 pc_energy.py   # ~2 s   — numpy only
python3 pc_robot.py    # ~2 s   — numpy only
python3 pc_torch.py    # ~60 s  — requires torch
```

Dependencies:

```bash
pip install numpy torch
```
