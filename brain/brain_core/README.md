# brain_core

A 400k-neuron spiking neural network brain with real-time SDL2 visualisation,
Hebbian learning, and ollama LLM integration.

Build:
```
gcc -O2 -o brain_core brain_core.c -lSDL2 -lpthread -lm
```

---

## Architecture

### Neurons — Leaky Integrate-and-Fire (LIF)
- **N = 399,424** neurons (632×632 grid)
- **FAN = 32** recurrent synaptic inputs per neuron
- Membrane potential `v[i]` leaks by 0.97× per step; fires when `v >= thr`
- Adaptive threshold `thr[i]` driven by local firing-rate homeostasis (target ~5%)
- Neighbourhood excitation: O(N) sliding-window sum over NEIGH_R=50 radius

### Readout layer
- **OUT = 256** readout units — dot-product of Wr[o] weights with spike vector
- Delta-rule learning: `ΔWr = lr * (target − out) * spk`
- Vocabulary: each readout slot learns a word label via Hebbian hash mapping

### Synaptic consolidation (every 1000 steps)
- Strong synapses `|W| > 0.35` grow toward ±1 (memory formation)
- Weak synapses `|W| < 0.35` decay toward 0 (forgetting / pruning)

---

## Temporal Self-Attention (ATTN-11 inspired)

Inspired by the **ATTN-11** project — a 1-layer transformer trained on a PDP-11/34
(1979) using Q8/Q15/Q16 fixed-point assembly. The key insight: attention is just
`Q·K^T / sqrt(d)` + softmax + weighted sum — simple enough to run on bare 16-bit
hardware with 1216 parameters.

We apply the same mechanism to brain_core but attending over **time** (recent
readout history) rather than token-sequence positions:

```
Q = Wq_r · out_current       # project current readout → ATTN_D=16
K[t] = Wk_r · out_hist[t]    # project each of ATTN_SEQ=8 past steps → ATTN_D
score[t] = Q·K[t] / sqrt(16) # scaled dot-product (mirrors SQRTSH=2 in ATTN-11)
A[t] = softmax(score)         # attention weights (ACTFN.MAC SFTMX equivalent)
attn_out = Σ A[t] · hist[t]  # weighted sum of history values
```

`ATTN_SEQ=8` and `ATTN_D=16` deliberately mirror ATTN-11's `SEQ.LN=8` and
`D.MODL=16`. The projections `Wq_r`, `Wk_r` are randomly initialised and fixed
(random projections preserve rough angular similarity, no backprop needed).

**Effect in brain_reply()**: output concepts are a blend of current and temporal
context — `0.6 * out[o] + 0.4 * attn_out[o]` — a residual connection analogous
to ATTN-11's `Y = O + X`. This gives the brain short-term contextual memory:
recent conversation topics influence which concepts surface in replies.

### Memory overhead
| Addition | Size |
|---|---|
| `out_hist[8][256]` ring buffer | 8 KB |
| `Wq_r[16][256]` + `Wk_r[16][256]` | 32 KB |
| `attn_out[256]` | 1 KB |
| **Total** | **~41 KB** |

---

## Learning pipeline

Per interaction (LLM mode):
1. `deep_encode(user_text, 2 cycles)` — 80-step stimulus + 20-step rest × 2
2. `push_readout_hist()` → `attn_readout()` — update temporal context
3. `hebbian_learn(llm_response)` — delta-rule update of Wr readout weights
4. `brain_associate(llm_response)` — LTP/LTD in recurrent W
5. `brain_replay(15 reps)` — consolidate spike pattern into W (hippocampal replay)

---

## Controls

| Key / Command | Action |
|---|---|
| Enter | Submit input |
| Ctrl-Q / Esc | Quit |
| `/l` | LLM mode — neurons learn from each reply |
| `/b` | Brain mode — neurons reply directly from readout |
| `/c [topic]` | Autonomous teaching loop (LLM ↔ brain) |
| `/stop` | End teaching loop |
| `/save` | Save brain state |
| `/stats` | Show interaction counts |

---

## State file (`brain_state.bin`)

Binary layout (append-compatible — old saves load cleanly, missing fields zero):

| Field | Notes |
|---|---|
| W[N][FAN] | Recurrent synaptic weights |
| Wr[OUT][N] | Readout weights |
| br[OUT] | Readout biases |
| n_train, n_interact | Event counters |
| vocab[OUT][32] | Word labels per readout slot |
| steps | Total simulation steps |
| vocab_count[OUT] | Activation counts |
| Wr_delta_acc | Accumulated learning magnitude |
| Wq_r[ATTN_D][OUT] | Temporal attention query projection |
| Wk_r[ATTN_D][OUT] | Temporal attention key projection |

---

## Origin

Built up from scratch over multiple sessions:
- GTX-750 CUDA baseline (200k neurons, CuPy) → ported to C + SDL2 for 400k neurons
- Added homeostasis, deep_encode, brain_replay, LTP/LTD association
- Temporal attention added from ATTN-11 PDP-11 transformer study

See `PROGRAMS.md` for the full Python variant history.
