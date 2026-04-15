# Brain Core — Program Guide

Ten programs, each a different experiment in spiking neural simulation and text interaction.
All share a common LIF (Leaky Integrate-and-Fire) neuron core but differ in scale, backend,
output decoding, and language model integration.

_Last updated: 2026-04-12 (brain_core.c rev 6)_

---

## Status Summary

| File | Backend | Neurons | LLM | Works Now |
|---|---|---|---|---|
| `brain_core.c` | C + SDL2 | 399,424 (~400k) | ollama (gemma3:1b) | YES |
| `brain_core_cpu.py` | NumPy | 100,000 | none | YES |
| `brain_core_cpu_smart.py` | NumPy | 100,000 | none | YES |
| `brain_core.py` | CuPy (GPU) | 200,000 | none | NO — cupy missing |
| `brain_core_cpu_gpt.py` | NumPy + PyTorch | 100,000 | DistilGPT2 | NO — PyTorch 2.2 |
| `brain_core_cpu_gpt_v2.py` | NumPy + PyTorch | 100,000 | DistilGPT2 | NO — PyTorch 2.2 |
| `brain_core_learning.py` | NumPy + PyTorch | 100,000 | GPT-2 | NO — PyTorch 2.2 |
| `brain_core_controlled.py` | NumPy + PyTorch | 100,000 | GPT-2 | NO — PyTorch 2.2 |
| `brain_core_reasoning.py` | NumPy + PyTorch | 100,000 | GPT-2 | NO — PyTorch 2.2 |
| `brain_core_combined.py` | NumPy + PyTorch | 100,000 | GPT-2 | NO — PyTorch 2.2 |

Fix for the PyTorch group: `pip install --upgrade torch`
Fix for the GPU version: `pip install cupy-cuda12x`

---

## Shared Neuron Model — How LIF Works

Every program is built on the same Leaky Integrate-and-Fire principle:

```
voltage(t) = voltage(t-1) * leak + synaptic_input + noise + external_stimulus
                                                                      |
                                              if voltage >= threshold ---> SPIKE, reset to 0
                                              else                    ---> decay continues
```

Parameters common to all:
- `leak = 0.95–0.97` — voltage decays each step if no input
- `threshold = 1.5` — spike when voltage crosses this
- `fan_in = 32` — each neuron connects to 32 random others
- `target_rate = 5%` — homeostasis keeps average firing near this

---

## 1. `brain_core.c` — Stats-Driven Brain with Ollama Chat and Memory Formation

**Why:** The primary program. Real-time stats dashboard showing how a spiking neural
network learns from LLM interactions — concept strength, memory consolidation, learning
progress, and network activity — while chatting with an LLM whose responses are
permanently encoded into the brain's weight structure.

**How:** Three threads run in parallel. The sim thread steps neurons continuously with a
1ms yield between steps so LLM/chat threads can acquire the brain mutex. The LLM thread
fires on user input, encodes text as stimulus, runs a forward pass, does Hebbian learning
on the response, then queries ollama over a raw TCP socket. The main thread renders a
**text stats panel at 4 fps** — gated to 250ms intervals, SDL_Delay(5) between frames.

**Scale:** 399,424 neurons (632×632), 12,781,568 synapses (N×FAN).

**Memory formation:** Two complementary mechanisms write learning into weights permanently:

1. **Synaptic consolidation** (background, every 1000 sim steps): scans all 12.78M
   recurrent weights. `|W|>0.35` → grows toward ±1 (LTP, stable attractor). `|W|<0.35`
   → decays slowly toward 0 (forgetting unused pathways). Bimodal distribution emerges:
   sparse committed memory highways + large plastic pool.

2. **Active learning** (per interaction): delta-rule Wr update + LTP/LTD recurrent
   association + hippocampal-style replay. See learning pipeline below.

**Stats panel (left pane, replaces neuron map):**
- **ACTIVITY** — total steps, rolling spike rate (%), interactions, training events, drive/noise
- **MEMORY CONSOLIDATION** — consolidated vs decaying synapse counts with formation bar
- **READOUT CONCEPTS** — vocab fill bar, top 18 concept slots sorted by Wr_norm (learned
  strength), each with a strength bar and activation count
- **LEARNING DYNAMICS** — Hebbian events, accumulated ΔWr, top concept, learning progress %

```
  STARTUP
     |
     +-- brain_new()  -- 399,424 neurons, random sparse weights
     +-- brain_load() -- restore brain_state.bin (W, Wr, br, vocab,
     |                   steps, vocab_count, Wr_delta_acc)
     |                   recomputes Wr_norm[OUT] from loaded Wr weights
     |
     +-- pthread_create(sim_thread)
     +-- SDL window (no texture — stats panel via embedded 8x8 bitmap font)
     |
  MAIN LOOP (4 fps render gate)
     |
     +-- SDL_PollEvent()
     |      +-- keypress → input buffer
     |      +-- RETURN   → submit()
     |      +-- CTRL+Q   → brain_save() + quit
     |
     +-- if elapsed >= 250ms → render_stats() + render_chat() + SDL_RenderPresent
     +-- else                → SDL_Delay(5)

  render_stats()
     |
     +-- trylock bmtx → snapshot all tracking fields (never blocks)
     +-- ACTIVITY:             steps, spike rate + bar, interactions, drive/noise
     +-- MEMORY CONSOLIDATION: strong/weak synapse counts, formation bar
     +-- READOUT CONCEPTS:     vocab fill bar, top 18 by Wr_norm with strength
     |                         bars (COL_BAR fixed position) and activation counts
     +-- LEARNING DYNAMICS:    Hebbian events, dWr total + per-event, top concept,
                                learning progress %

  SIM THREAD
     |
     +-- loop: lock → brain_step() → unlock → nanosleep(1ms)

  brain_step()
     |
     +-- snapshot _pre[] spikes
     +-- O(N) sliding-window neighbour sum (radius=50)
     +-- for each neuron i:
     |      syn = Σ W[i][j] * _pre[idx[i][j]]
     |      v[i] = v[i]*0.97 + syn + neigh + drive + noise + ext
     |      spike → reset v[i]=0; homeostasis → nudge thr[i]
     +-- steps++; spike_rate rolling avg
     +-- every 1000 steps: consolidation sweep
            |W|>0.35 → W += sign(W)*0.002*(1-|W|)   LTP → ±1
            |W|<0.35 → W *= 0.9995                   slow decay → 0

  ── LEARNING PIPELINE (per LLM interaction) ──────────────────────────────

  deep_encode(text, cycles=2)
     |
     +-- encode_text → _stim[] (character hash → neuron indices)
     +-- for each cycle:
     |      80 steps with stimulus injected   (pattern driven in)
     |      20 steps free decay               (pattern settles into attractor)
     +-- recompute out[] = tanh(br + Wr*spk)

  hebbian_learn(response)   [DELTA RULE — error-corrective]
     |
     +-- tokenise response → target[slot]=1 for each word's hashed slot
     +-- recompute out[] from current spk
     +-- for each slot o:
     |      err = target[o] - out[o]
     |      if |err| < 0.05: skip  (already learned — no redundant update)
     |      Wr[o][i] += 0.003 * err * spk[i]   (strong, error-scaled)
     |      br[o]    += 0.005 * err             (bias encodes base rate)
     |      Wr_norm[o] recomputed
     +-- n_train++

  brain_associate(response)   [LTP + LTD — specificity]
     |
     +-- deep_encode(response, 2)   (strong response pattern in b->spk)
     +-- for each synapse W[i][j]:
     |      pre=pre_spk[idx[i][j]],  post=spk[i]
     |      pre AND post:  W += 0.001   (LTP: Hebbian, 5x stronger than before)
     |      pre, NOT post: W -= 0.0002  (LTD: anti-Hebbian, prunes wrong paths)

  brain_replay(reps=15)   [HIPPOCAMPAL REPLAY — consolidation]
     |
     +-- save current spk into _replay_pat[]
     +-- for 15 free steps:
            brain_step(NULL)
            for each active post neuron: if pre in _replay_pat → W += 0.0008
     (writes response attractor into recurrent W without external stimulus)

  LLM THREAD (spawned per submit)
     |
     +-- deep_encode(user_text, 2) → pre_spk snapshot
     +-- find top-5 concepts by Wr_norm + weakest concept
     +-- build prompt:
     |      "Strongest learned: [top-5 words]"
     |      "Weakest (needs reinforcement): [word]"
     |      "Active concept: [current]"
     |      + 6-turn history + "User: ... Brain:"
     +-- ollama_query()
     +-- deep_encode(user_text, 2)   (restore user pattern)
     +-- hebbian_learn(response)
     +-- brain_associate(response)
     +-- brain_replay(15)
     +-- brain_save()

  /c LOOP THREAD (autonomous teaching)
     |
     +-- every 3rd round: topic = weakest concept (targeted remediation)
     +-- build teaching prompt:
     |      "Brain's strong concepts: [top-3]"
     |      "It needs to learn: [weak]"
     |      "It said: [topic] — teach in 1-2 sentences, connect to known"
     +-- ollama_query()
     +-- deep_encode(topic, 2) → pre_spk snapshot
     +-- hebbian_learn(llm_resp)
     +-- brain_associate(llm_resp)
     +-- brain_replay(15)
     +-- deep_encode(llm_resp, 1) → brain_reply() → next topic
     +-- brain_save()

  Drive levels
     idle:    g_drive=0.04  g_noise=0.02
     active:  g_drive=0.05  g_noise=0.03   (during deep_encode cycles)
```

**Brain struct — memory tracking fields:**

| Field | Type | Description |
|---|---|---|
| `steps` | `uint64_t` | Total sim steps since creation (persisted) |
| `spike_rate` | `float` | Rolling average firing rate 0..1 |
| `vocab_count[OUT]` | `int[256]` | Activation count per readout slot (persisted) |
| `Wr_norm[OUT]` | `float[256]` | L2 norm of each readout weight vector |
| `Wr_delta_acc` | `float` | Total accumulated |ΔWr| across all learning (persisted) |
| `consol_strong` | `uint32_t` | Synapse count with \|W\|>0.35 (updated every 1000 steps) |
| `consol_weak` | `uint32_t` | Synapse count with \|W\|<0.35 (updated every 1000 steps) |

**Key constants:**

| Constant | Value | Meaning |
|---|---|---|
| `N` | 399,424 | Total neurons (632²) |
| `FAN` | 32 | Recurrent fan-in per neuron (12.78M total synapses) |
| `OUT` | 256 | Readout concept slots |
| `NEIGH_R` | 50 | Neighbourhood radius |
| `RENDER_FPS` | 4 | Display frame rate |
| `g_drive` idle | 0.04 | Baseline drive (3-5% firing rate) |
| `g_noise` idle | 0.02 | Random noise at idle |

**Persistence:** `brain_state.bin` saves W, Wr, br, vocab, steps, vocab_count, Wr_delta_acc.
Wr_norm is recomputed on load (O(OUT×N) = 100M ops, ~0.5s startup cost after first save).

**Build:** `gcc -O2 -o brain_core brain_core.c -lSDL2 -lpthread -lm` (needs `libsdl2-dev`)
**Run:** `./brain_core` (needs `ollama serve` running with `gemma3:1b`)

**Slash commands:** `/l` LLM mode · `/b` Brain-only mode · `/c [topic]` autonomous teach loop · `/stop` · `/save` · `/stats`

**Rev 5 changes (2026-04-12):**
- Removed neuron pixel map; replaced with text stats dashboard
- Added synaptic consolidation (memory formation, bimodal weight distribution)
- Added `steps`, `spike_rate`, `vocab_count`, `Wr_norm`, `Wr_delta_acc`, `consol_strong/weak` to Brain struct
- Removed `snap_v[N]`, `snap_spk[N]`, `glow[N]` from App (saves ~3.5MB)
- Fixed `forward_pass` idle restore bug (was 0.01/0.005, now correctly 0.04/0.02)
- `brain_save/load` persists new fields; Wr_norm recomputed on load

**Rev 6 changes (2026-04-12) — real learning:**
- **`hebbian_learn` → delta rule**: `ΔWr = 0.003*(target-out)*spk` replaces pure
  Hebbian `ΔWr = 0.0005*spk`. Error-corrective: already-learned concepts (out≈target)
  get near-zero update; novel concepts get full-strength update. Rate 6× stronger.
- **`br[]` bias learning**: `br[o] += 0.005*err` per interaction. Bias encodes how
  often each concept fires — gives concepts a persistent base activation, lowering
  recall threshold for frequently-seen material.
- **`brain_associate` → LTP + LTD**: pre AND post active → W += 0.001 (5× stronger).
  Pre active, post silent → W -= 0.0002 (anti-Hebbian, LTD). Creates pathway
  specificity: user-concept → response-concept strengthened, wrong paths pruned.
- **`deep_encode(text, cycles)`** (new): replaces `forward_pass(text, 80)` during
  learning. Each cycle: 80 steps with stimulus + 20 steps free decay. 2 cycles =
  200 steps total. Rest gaps allow pattern to settle into attractor before next push.
- **`brain_replay(reps)`** (new): hippocampal-style replay. Saves current spk pattern,
  runs N free sim steps re-strengthening W from stored pattern. Converts short-term
  spike activity into long-term synaptic structure (15 reps per interaction).
- **Consolidation threshold**: 0.5 → 0.35 — catches learning-boosted weights much
  earlier in their trajectory toward ±1.
- **Prompt upgraded**: injects top-5 concepts by Wr_norm + weakest concept into every
  LLM prompt. Brain's actual learned state now genuinely shapes LLM responses.
- **`/c loop` smarter teaching**: finds weakest concept every round, builds teaching
  prompt that names strong concepts and targets weak ones. Every 3rd round forces
  topic = weakest concept (targeted remediation rather than drifting word salad).
- **Display fixes**: `ΔWr` → `dWr` (bitmap font), window title em-dash removed,
  consolidated/decaying lines shortened to fit panel, spike rate bar on own line,
  column headers aligned with `COL_BAR`/`COL_CNT` constants, non-ASCII vocab
  chars stripped from token labels.

---

## 2. `brain_core.py` — GPU Brain (CuPy)

**Why:** The original high-scale version. Moves all neuron arrays to GPU memory
so 200,000 neurons can run in real-time (~0.017s/step on a GTX-750).
Identical architecture to `brain_core_cpu.py` but uses `cp` (CuPy) instead of `np`.

**How:** Text input is hashed to neuron indices and injected as a current spike.
Output is decoded by a trained linear readout: spike pattern → ASCII character.

```
  STARTUP
     |
     +-- allocate on GPU (CuPy):
     |       voltage[200k], threshold[200k], leak[200k]
     |       inputs_idx[200k x 32], weights[200k x 32]   (sparse connectivity)
     |       readout_W[128 x 200k], readout_b[128]        (char decoder)
     +-- load brain_paths.npz if exists

  CHAT LOOP
     |
     v
  user types text
     |
     +-- _encode_text_to_input(text)
     |       |
     |       +-- for each character in text:
     |               hash char --> neuron index
     |               ext_input[index] += 1.0
     |
     +-- run settle_steps=80 GPU steps with ext_input injected
     |
     +-- generate reply (reply_len=8 chars):
     |       for each output character:
     |           run 10 free steps
     |           _decode_char_from_spikes():
     |               readout = readout_W @ spikes + readout_b   (GPU matmul)
     |               char = argmax(readout) --> ASCII
     |
     +-- _train_readout(spikes, target_char)  [Hebbian on readout layer]
     |       error = one_hot(target) - readout
     |       readout_W += lr * outer(error, spikes)
     |
     +-- print "Brain: <8 chars>"
     |
  on Ctrl-C: save brain_paths.npz --> loop end

  GPU step (brain_step):
     |
     +-- presyn = spike vector (bool -> float, on GPU)
     +-- incoming = presyn[inputs_idx]        (gather)
     +-- syn_current = sum(incoming * weights, axis=1)
     +-- neigh = cp.roll(presyn, ±radius) summed
     +-- total = syn_current + neigh + bias + global_drive + noise + ext_input
     +-- homeostasis: firing_avg update, threshold nudge
     +-- LIF: voltage = voltage * leak + total
     +-- spike = (voltage >= threshold)
     +-- voltage[spike] = 0
```

**Needs:** `pip install cupy-cuda12x`

---

## 3. `brain_core_cpu.py` — CPU Brain (NumPy)

**Why:** Identical to the GPU version but runs on CPU with NumPy.
Drops to 100,000 neurons to stay fast enough for interactive use (~4s/response).
The safe fallback when CuPy is not available.

**How:** Same LIF + sparse connectivity + linear readout as the GPU version,
just with `np` instead of `cp`. Output is still raw spike-to-ASCII.

```
  STARTUP
     |
     +-- allocate NumPy arrays:
     |       voltage[100k], threshold[100k], leak[100k]
     |       inputs_idx[100k x 32], weights[100k x 32]
     |       readout_W[128 x 100k], readout_b[128]
     +-- load brain_paths_cpu.npz if exists

  CHAT LOOP  (identical flow to brain_core.py, CPU only)
     |
     +-- encode: char hash --> ext_input vector
     +-- 80 settle steps (NumPy)
     +-- 8 output chars: 10 free steps each, argmax readout --> ASCII
     +-- Hebbian update on readout weights
     +-- print reply ("ooooooo" style — raw ASCII, usually nonsense)
     |
  on Ctrl-C: save brain_paths_cpu.npz

  CPU step:
     |
     +-- incoming = presyn[inputs_idx]               (fancy index)
     +-- syn_current = (incoming * weights).sum(1)
     +-- neigh = np.roll(presyn, offset) for ±radius
     +-- total = syn_current + neigh + bias + drive + noise + ext
     +-- homeostasis threshold adaptation
     +-- voltage = voltage * leak + total
     +-- spike = voltage >= threshold
     +-- voltage[spike] = 0
```

**Output example:** `Brain: oooooooo` — real-time but semantically empty.

---

## 4. `brain_core_cpu_smart.py` — CPU Brain with Word Embeddings

**Why:** Breaks out of raw ASCII output. Instead of spike → char,
it uses a learned embedding space so the brain outputs real English words.
No external LLM needed — vocabulary of 176 common words, embeddings in NumPy.

**How:** Input text is averaged into a 64-d embedding vector, projected into
100k neuron space via a learned encoder matrix. Spikes are projected back to 64-d
via a decoder matrix, then cosine-matched to the nearest word in the vocabulary.

```
  STARTUP
     |
     +-- SimpleWordEmbeddings(dim=64)
     |       176 words, random unit vectors (seed=42, fixed)
     |
     +-- SmartCPUBrain:
     |       neurons[100k] with attractor dynamics
     |       encoder_W [64 x 100k]   -- word embedding --> neuron space
     |       decoder_W [100k x 64]   -- neuron space --> word embedding
     |       decoder_b [64]
     +-- load brain_paths_smart.npz if exists

  CHAT LOOP
     |
     v
  user types text
     |
     +-- encode_words_to_input(text):
     |       words = text.split()
     |       word_vecs = [embedding[w] for w in words]
     |       avg_vec = mean(word_vecs)            -- 64-d average
     |       ext_input = tanh(avg_vec @ encoder_W) * 2.0   -- 100k vector
     |
     +-- run 50 settle steps with ext_input
     |
     +-- train_on_pair(text, text)   [self-supervised]
     |       target = embedding[first_word_of_input]
     |       pred = spikes @ decoder_W + decoder_b
     |       error = target - pred
     |       decoder_W += lr * outer(spikes, error)
     |       encoder_W += lr * 0.1 * outer(input_embedding, spikes)
     |
     +-- generate 3 reply words:
     |       for each word:
     |           run 20 free steps
     |           embedding_pred = spikes @ decoder_W + decoder_b
     |           word = argmax cosine_similarity(embedding_pred, all_embeddings)
     |
     +-- print "Brain: great stop right" (real words, random meaning at first)
     |
  on Ctrl-C: save brain_paths_smart.npz

  Attractor neuron step:
     |
     +-- voltage = voltage * leak + inputs
     +-- voltage += 0.1 * attractor    <-- slow positive feedback
     +-- spike = voltage >= threshold
     +-- voltage[spike] = 0
     +-- attractor = 0.9 * attractor + 0.1 * spike   <-- slow trace
```

**Why words instead of chars:** The decoder projects the entire spike pattern into
embedding space and finds the nearest of 176 real words by cosine similarity.
Results are real English but semantically random until the decoder is trained
through many interactions.

---

## 5. `brain_core_cpu_gpt.py` — Brain + DistilGPT2

**Why:** First attempt to wire a real language model to the brain state.
The spiking network runs first to extract a "brain state" signal (firing rate,
active neuron regions), which is then embedded into the GPT prompt as context.

**How:** Brain state is summarised as a short string (% firing, top active regions),
then prepended to the user message before calling DistilGPT2 for token generation.

```
  STARTUP
     |
     +-- HybridNeuronModel[100k] + sparse weights
     +-- GPT2LMHeadModel.from_pretrained("distilgpt2")   <-- REQUIRES PyTorch >= 2.4
     +-- GPT2Tokenizer

  CHAT LOOP
     |
     v
  user types text
     |
     +-- encode_text_to_input(text):
     |       each char --> neuron region via hash
     |       build ext_input vector
     |
     +-- run 40 settle steps
     |
     +-- get_brain_state():
     |       firing_rate = spikes.mean()
     |       active_regions = identify top firing neuron clusters
     |       return summary string
     |
     +-- generate_response_with_gpt(user_text, brain_state):
     |       prompt = f"Brain state: {brain_state}\nUser: {user_text}\nBrain:"
     |       tokens = tokenizer.encode(prompt)
     |       output = gpt_model.generate(tokens, max_new_tokens=50)
     |       response = tokenizer.decode(output)
     |
     +-- Hebbian update on recurrent weights
     +-- print response
     |
  on Ctrl-C: save brain_paths_gpt.npz
```

**Blocked by:** PyTorch 2.2 installed; needs >= 2.4. Fix: `pip install --upgrade torch`

---

## 6. `brain_core_cpu_gpt_v2.py` — Brain + GPT2 + Concept Space

**Why:** Improves on v1 by adding a structured `ConceptSpace` — neuron regions
are explicitly assigned to words (200 words, ~500 neurons each).
This makes the brain state readable: "firing regions: cat, food, happy" rather than
raw percentages.

**How:** Each word owns a neuron slice. When the user says "cat", those neurons
get extra stimulation. GPT sees the list of active concept regions as context.

```
  STARTUP
     |
     +-- HybridNeuronModel[100k]
     +-- ConceptSpace(100k neurons):
     |       200 concept words assigned to non-overlapping regions
     |       region_size = 100k / 200 = 500 neurons per word
     +-- DistilGPT2 model + tokenizer

  CHAT LOOP
     |
     v
  user types text
     |
     +-- encode_text(text):
     |       for each word in text:
     |           if word in concepts: activate_word(word)
     |               ext_input[region] += strength
     |           else: hash-based activation
     |
     +-- run 60 settle steps
     |
     +-- decode_brain_state():
     |       for each concept region:
     |           measure firing rate in that region
     |       return top-N active concepts as word list
     |
     +-- generate_response(user_text, active_concepts):
     |       prompt = f"Active concepts: {concepts}\nUser: {user_text}\nAssistant:"
     |       GPT2 generate --> decode
     |
     +-- update Hebbian weights
     +-- print response

  ConceptSpace.decode_active_regions():
     |
     +-- for each word/region:
     |       rate = spikes[region_start:region_end].mean()
     +-- sort by rate, return top words
```

---

## 7. `brain_core_learning.py` — Brain + GPT2 + Hebbian Training

**Why:** Adds a pre-training phase. Before the chat loop starts, the brain
trains on a built-in set of question/answer conversation pairs using Hebbian
weight updates. The goal is to bias the network toward sensible responses.

**How:** `HebbianLearning` computes weight updates from co-activation of
pre and post-synaptic neurons. After offline training, the same GPT-based
pipeline generates actual responses.

```
  STARTUP
     |
     +-- LearningBrain[100k]:
     |       layer_W[100k x 100k sparse]  -- recurrent
     |       readout_W[vocab x 100k]      -- output
     +-- HebbianLearning(lr=0.001)
     +-- GPT-2 (full, not distil)
     +-- load conversation_training.json (or use defaults)
     |
     +-- train_on_conversations():
             for each (user, assistant) pair:
                 encode_text_to_neurons(user)   --> ext_input
                 forward_pass(steps=50)         --> spike pattern A
                 encode_text_to_neurons(assistant) --> ext_input
                 forward_pass(steps=50)         --> spike pattern B
                 hebbian_update(W, A, B)
                     delta_W = lr * outer(post_activity, pre_activity)
                     W += delta_W
                     clip(W, -1, 1)

  CHAT LOOP
     |
     v
  user types text
     |
     +-- encode_text_to_neurons(text): char ordinal / word hash --> ext vector
     +-- forward_pass(steps=80)
     +-- generate_response(user_text):
     |       brain_context = firing_rate, top_regions
     |       prompt = build prompt with context + conversation history
     |       GPT-2 generate(max_new_tokens=100)
     |       clean/filter output
     |
     +-- learn_from_example(user, response)  [online Hebbian]
     +-- print response

  HebbianLearning.update_weights():
     |
     +-- pre  = pre-synaptic activity vector (float)
     +-- post = post-synaptic activity vector (float)
     +-- delta = lr * outer(post, pre)
     +-- W += delta
     +-- W = clip(W, -1, 1)
```

---

## 8. `brain_core_controlled.py` — Brain + GPT2 + Topic Lock

**Why:** GPT-2 tends to drift off-topic or hallucinate. This version adds a
`TopicController` that extracts the topic and keywords from user input, then
constructs a rigid prompt template that forces GPT to stay on subject.

**How:** Regex patterns detect topics (math, greeting, weather, etc.).
The prompt explicitly states "you MUST answer about: X" and provides
extracted keywords. A separate `HebbianLearning` module updates recurrent
weights in the background.

```
  STARTUP
     |
     +-- ControlledReasoningBrain[100k]:
     |       ReasoningLayers (see below)
     |       TopicController
     |       HebbianLearning
     +-- GPT-2

  TopicController patterns:
     math     --> r"\b(\d+|plus|minus|times|...)\b"
     greeting --> r"\b(hello|hi|hey|...)\b"
     weather  --> r"\b(weather|rain|...)\b"
     science  --> r"\b(atom|physics|...)\b"
     ... etc

  CHAT LOOP
     |
     v
  user types text
     |
     +-- TopicController.extract_topics(text) --> ["math", ...]
     +-- TopicController.extract_keywords(text) --> ["2", "plus", "2"]
     |
     +-- encode_text(text) --> activate concept regions
     +-- reason_about(user_text, steps=80):
     |       run 80 LIF steps with word-region stimulation
     |       collect layer activity at each step
     |       return reasoning trace
     |
     +-- generate_controlled_response(user_text):
     |       prompt = TopicController.build_controlled_prompt(
     |                   user_text, topics, keywords, history)
     |       --> "You are a focused AI. Topic: math. Keywords: 2, plus, 2.
     |            You MUST answer about math only.
     |            User: what is 2 plus 2?  Assistant:"
     |       GPT-2.generate() --> response
     |       filter: remove off-topic sentences
     |
     +-- Hebbian update
     +-- print response
```

---

## 9. `brain_core_reasoning.py` — Layered Reasoning Architecture

**Why:** Organises neurons into four functional layers inspired by cortical
hierarchy: Perception → Association → Decision → Output.
Each layer has dedicated neuron pools connected in a forward chain.
The idea is that input flows through layers and each stage abstracts further.

**How:** `ReasoningLayers` partitions 100k neurons into regions. Forward connections
pass spikes from perception → association → decision → output at each step.
GPT-2 uses the output layer activity as context.

```
  STARTUP
     |
     +-- ReasoningLayers(100k neurons):
     |       perception   [neurons  0   – 24,999]  25k
     |       association  [neurons 25k  – 49,999]  25k
     |       decision     [neurons 50k  – 74,999]  25k
     |       output       [neurons 75k  – 99,999]  25k
     |
     |       concept words mapped to perception layer regions
     |       forward connections: perception → association → decision → output
     +-- ReasoningBrain + GPT-2

  CHAT LOOP
     |
     v
  user types text
     |
     +-- encode_text(text):
     |       activate word regions in PERCEPTION layer only
     |
     +-- reason_about(user_text, steps=100):
     |       for each step:
     |           full LIF step (all neurons)
     |           perception   spikes propagate --> association  (W_p2a)
     |           association  spikes propagate --> decision     (W_a2d)
     |           decision     spikes propagate --> output       (W_d2o)
     |       record get_layer_activity() at each step:
     |           {perception: 0.04, association: 0.06, decision: 0.03, output: 0.05}
     |
     +-- generate_response(user_text):
     |       output_concepts = decode active words in output layer
     |       reasoning_summary = layer activity trace
     |       prompt = "Reasoning: {summary}\nConcepts: {concepts}\nUser: ...\nBrain:"
     |       GPT-2 generate
     |
     +-- print response

  _create_forward_connections(src, tgt, fan_in):
     |
     +-- for each target neuron:
             randomly sample fan_in neurons from source layer
             assign weights ~ N(0, 0.1)
```

---

## 10. `brain_core_combined.py` — Full Stack (Reservoir + Hebbian + Topic Control)

**Why:** Combines the best features of all the above into one program:
reservoir computing (random recurrent net), Hebbian learning on responses,
topic-controlled GPT-2 prompting, and concept word tracking.
Intended as the most complete Python-only version.

**How:** The recurrent network acts as a reservoir (Echo State Network style) —
its rich dynamics encode history. The readout layer is trained online.
GPT-2 handles final language generation with topic control.

```
  STARTUP
     |
     +-- CombinedBrain(100k neurons, 1024 output concepts):
     |       reservoir:  HybridNeuronModel[100k]
     |                   sparse weights[100k x 32]  (random, fixed)
     |       readout_W:  [1024 x 100k]              (trained online)
     |       HebbianLearning(lr=0.001)
     |       TopicController
     |       1024 concept words (generated vocabulary)
     +-- GPT-2 (full)
     +-- load brain_state if exists

  CHAT LOOP
     |
     v
  user types text
     |
     +-- forward_pass(text, steps=80):
     |       ext = _encode_text_to_reservoir_input(text)
     |               char ordinals normalised --> 100k sparse vector
     |       run 80 reservoir steps with ext injected
     |       reservoir_state = mean spike pattern over last 20 steps
     |       readout = readout_W @ reservoir_state   --> 1024 concept scores
     |       active_concepts = top-N from readout
     |
     +-- learn_from_example(user_text, response):
     |       target = _encode_text_to_output_target(response)
     |                   encode target as 1024-d binary concept vector
     |       HebbianLearning.update(readout_W, reservoir_state, target)
     |
     +-- generate_response(user_text):
     |       topics = TopicController.extract_topics(user_text)
     |       keywords = TopicController.extract_keywords(user_text)
     |       active_concepts_str = join top concept words
     |       prompt = TopicController.build_controlled_prompt(
     |                   user_text, topics, keywords, history)
     |               + f"\nBrain concepts: {active_concepts_str}"
     |       GPT-2.generate(prompt, max_new_tokens=80)
     |       clean response
     |
     +-- print response
     +-- online Hebbian update with (user_text, response) pair

  Reservoir (Echo State) principle:
     |
     +-- weights are FIXED after init (not trained)
     +-- rich recurrent dynamics encode temporal history
     +-- only readout_W is trained (much cheaper than full backprop)
     +-- reservoir_state = average recent spike activity
     +-- this is a classic Echo State Network applied to spiking neurons
```

---

## Dependency Tree

```
  brain_core.c
      └── SDL2, pthread, math, ollama (TCP socket)

  brain_core.py
      └── cupy  [MISSING]

  brain_core_cpu.py
      └── numpy  [OK]

  brain_core_cpu_smart.py
      └── numpy  [OK]

  brain_core_cpu_gpt.py
  brain_core_cpu_gpt_v2.py
  brain_core_learning.py
  brain_core_controlled.py
  brain_core_reasoning.py
  brain_core_combined.py
      └── numpy  [OK]
      └── transformers  [OK]
      └── torch >= 2.4  [NEED UPGRADE — currently 2.2.2]
```

---

## Evolution of the Series

```
  brain_core.c          <-- visual SDL2 display, ollama LLM, C speed
       |
  brain_core.py         <-- port to Python/GPU (CuPy), scale to 200k neurons
       |
  brain_core_cpu.py     <-- CPU fallback (NumPy), 100k neurons
       |                    output: raw spike → ASCII char
       |
  brain_core_cpu_smart.py  <-- replace char decoder with word embeddings
       |                       output: real English words, no LLM needed
       |
  brain_core_cpu_gpt.py    <-- attach DistilGPT2, brain state → prompt context
       |
  brain_core_cpu_gpt_v2.py <-- add ConceptSpace: named neuron regions per word
       |
  brain_core_learning.py   <-- add Hebbian pre-training on conversation pairs
       |
  brain_core_controlled.py <-- add TopicController to stop GPT drift
       |
  brain_core_reasoning.py  <-- add 4-layer cortical hierarchy (P→A→D→O)
       |
  brain_core_combined.py   <-- combine: reservoir + Hebbian + topic control
```

Each step added one mechanism. None of the Python versions are "finished" —
`brain_core_cpu_smart.py` is the most useful one that works today without
extra dependencies.

---

## Capability Ranking — Least to Most

Ranked on four axes: output quality, learning ability, interpretability, and
architectural sophistication. "Works now" is noted separately — a broken program
can still rank high if its design is sound.

```
  RANK   PROGRAM                    OUTPUT QUALITY   LEARNING   ARCHITECTURE
  ────────────────────────────────────────────────────────────────────────────
  1  (weakest)
         brain_core_cpu.py          spike → ASCII     readout    flat LIF
                                    "ooooooo"         only       no structure

  2      brain_core.py (GPU)        spike → ASCII     readout    flat LIF
                                    same as above,    only       (larger scale)
                                    just faster

  3      brain_core_cpu_gpt.py      GPT language      readout    flat LIF
                                    but brain state   + Hebbian  brain state as
                                    = raw % firing,              vague % number
                                    context is weak

  4      brain_core_cpu_smart.py    real words        encoder/   attractor LIF
                                    "great stop right" decoder   no LLM needed
                                    semantically      trained    word embeddings
                                    random but legit  online

  5      brain_core_cpu_gpt_v2.py   GPT language      readout    ConceptSpace:
                                    concept-aware     + Hebbian  named neuron
                                    prompt context               regions

  6      brain_core_learning.py     GPT language      Hebbian    pre-trained on
                                    pre-trained bias  + offline  conversation
                                    toward sensible   training   pairs before
                                    responses                    chat starts

  7      brain_core_controlled.py   GPT language      Hebbian    TopicController
                                    stays on topic    + topic    regex-based
                                    filtered output   filter     topic lock

  8      brain_core.c               ollama language   Hebbian    SDL2 visual
                                    (gemma3:1b)       online     10k neurons
                                    best raw output              real-time 60fps
                                    threaded I/O

  9      brain_core_combined.py     GPT language      Hebbian    reservoir net
                                    topic-controlled  + reservoir + Hebbian
                                    concept-aware     + online   + topic control
                                    richest context   learning   all combined

  10 (strongest design)
         brain_core_reasoning.py    GPT language      Hebbian    4-layer cortex
                                    reasoning trace   + layered  P→A→D→O
                                    fed into prompt   forward    closest to
                                    most structured   pass       neuroscience
  ────────────────────────────────────────────────────────────────────────────
  * Works today: ranks 1, 2, 4, 8.  Ranks 3,5,6,7,9,10 need PyTorch upgrade.
```

---

## Logical Progression — How Each Builds on the Last

The series is not random. Each program identifies one specific failure in its
predecessor and adds exactly one mechanism to fix it.

```
  PROBLEM                          SOLUTION                  PROGRAM
  ─────────────────────────────────────────────────────────────────────────────

  "We need a neural sim at all"
        |
        v
  Build LIF network + text I/O                              brain_core_cpu.py
  Output: spike index → chr() → ASCII "ooooooo"
        |
        | PROBLEM: too slow on CPU, want more neurons
        v
  Port to GPU with CuPy                                     brain_core.py
  200k neurons, same decode, same gibberish but faster
        |
        | PROBLEM: output is meaningless characters
        v
  Replace char decode with word embeddings                  brain_core_cpu_smart.py
  176-word vocab, cosine similarity decoder
  Output: real words, semantically random but legible
        |
        | PROBLEM: brain has no language generation ability
        v
  Attach DistilGPT2, feed brain firing rate as context      brain_core_cpu_gpt.py
  Brain state = "firing 4.8%" — context is near-useless
        |
        | PROBLEM: firing % tells GPT nothing meaningful
        v
  Add ConceptSpace: neurons assigned to word regions        brain_core_cpu_gpt_v2.py
  Brain state = "active: cat, food, happy" — real signal
        |
        | PROBLEM: brain starts cold, no prior knowledge
        v
  Add offline Hebbian training on conversation pairs        brain_core_learning.py
  Brain biased toward sensible input→output associations
        |
        | PROBLEM: GPT drifts off-topic, hallucinates
        v
  Add TopicController: regex → forced prompt template       brain_core_controlled.py
  GPT must stay on extracted topic and keywords
        |
        | PROBLEM: all neurons are equal, no structure
        v
  Add 4-layer cortical hierarchy: P→A→D→O                  brain_core_reasoning.py
  Input flows through dedicated processing stages
  Reasoning trace from all layers fed to GPT
        |
        | PROBLEM: features are spread across separate programs
        v
  Combine: reservoir + Hebbian + TopicControl               brain_core_combined.py
  Single program with all mechanisms active together
        |
        | SEPARATELY: want real-time visual + best LLM
        v
  C implementation with SDL2 + ollama                       brain_core.c
  Best output quality (gemma3 >> GPT-2)
  Real-time visualiser, persistent state, threaded
```

---

## Which Is the Best?

**For output quality right now:** `brain_core.c`
- Uses gemma3:1b via ollama — far better language than GPT-2
- Real-time SDL2 visual of spiking activity
- Threaded: sim never stops, LLM runs in background
- Persistent state across sessions
- Works today, no Python dependency issues

**Best design / highest ceiling:** `brain_core_reasoning.py` (once PyTorch is fixed)
- 4-layer cortical hierarchy is the most neuroscience-grounded architecture
- Each layer adds abstraction: perception → association → decision → output
- The reasoning trace gives the LLM the richest possible brain context
- The right foundation to build on for long-term development

**Best for working with right now (no fixes needed):** `brain_core_cpu_smart.py`
- Pure NumPy, no broken dependencies
- Produces real English words
- Online learning via encoder/decoder gradient updates
- Simple enough to extend without breaking

---

## How to Improve Each Program

### `brain_core_cpu.py` — Flat LIF, ASCII output

The fundamental decoder is the bottleneck. The whole spike pattern is collapsed
to one character per step. It cannot represent words, only letters, and the
mapping is arbitrary.

```
  IMPROVEMENTS (in order of impact):

  1. Replace char readout with word embedding decoder
     -- copy the decoder from brain_core_cpu_smart.py
     -- 64-d embedding space, cosine similarity, 176 words
     -- immediate: output becomes real words

  2. Add refractory period
     -- after spike, neuron cannot fire for N steps
     -- prevents runaway columns, more realistic dynamics
     -- add: refractory_counter[n] = REFRAC_PERIOD on spike
     --      skip update if refractory_counter[n] > 0

  3. Add inhibitory neuron population (20% of neurons)
     -- inhibitory neurons have negative output weights
     -- prevents synchrony collapse (all-spike or no-spike)
     -- assign: inh_mask = random 20k of 100k neurons
     --          weights[inh_mask] *= -1

  4. Add spike-timing-dependent plasticity (STDP)
     -- if pre fires just before post: strengthen synapse
     -- if post fires just before pre: weaken synapse
     -- delta_W = A+ * exp(-dt/tau+) for pre→post
     -- delta_W = -A- * exp(-dt/tau-) for post→pre
```

---

### `brain_core.py` — GPU LIF, ASCII output (needs cupy)

Same architectural weakness as cpu.py but at scale. The GPU is underused —
the simulation loop is the bottleneck, not the decode.

```
  IMPROVEMENTS:

  1. Fix cupy install: pip install cupy-cuda12x
     -- then all the same improvements as brain_core_cpu.py apply

  2. Scale to 500k–1M neurons (GTX-750 has 1GB VRAM)
     -- current 200k uses ~300MB at float32
     -- test: cp.zeros(1_000_000, dtype=cp.float32) fits?
     -- larger = richer dynamics, better separation of patterns

  3. Move connectivity matrix to sparse format
     -- current: inputs_idx [200k x 32] = 25MB
     -- cp.sparse.csr_matrix for weight multiplication
     -- faster matmul, more fan_in possible

  4. Add GPU-side STDP kernel
     -- write custom CuPy kernel for STDP weight update
     -- avoids Python loop overhead
     -- cp.RawKernel for the weight delta computation

  5. Swap ASCII decoder for GPU word embedding lookup
     -- decoder_W on GPU, matmul stays on device
     -- no CPU round-trip for decode
```

---

### `brain_core_cpu_smart.py` — Word Embeddings (works now, best starting point)

The encoder and decoder matrices start random and improve slowly. The main
weakness is the tiny vocabulary (176 words) and random embedding initialisation.

```
  IMPROVEMENTS:

  1. Replace random embeddings with pre-trained word vectors
     -- GloVe 50d or 100d (free, no PyTorch needed)
     -- download: glove.6B.50d.txt (~170MB)
     -- words = {}; open('glove.6B.50d.txt'):
     --     words[word] = np.array(vals, dtype=float32)
     -- genuine semantic similarity: "cat" near "dog", "hot" near "warm"
     -- response words become contextually related to input

  2. Expand vocabulary
     -- 176 → 10,000 words (still manageable with GloVe)
     -- decoder cosine search over 10k vectors is fast with np.dot

  3. Add temperature to word sampling
     -- current: argmax (always picks #1 match, deterministic)
     -- instead: softmax(similarities / T) then np.random.choice
     -- T=0.5 conservative, T=1.5 creative
     -- prevents the same 3 words appearing every reply

  4. Multi-word context (don't average input embeddings)
     -- current: avg_embedding = mean(word_vecs)
     -- averaging destroys word order and loses minority words
     -- instead: concatenate last N word embeddings → 64*N dim
     --          project with a wider encoder_W [64*N x 100k]

  5. Add word-to-word association training
     -- after each exchange, run: train_on_pair(word_i, word_j)
     --   for every pair of words seen in the last 5 turns
     -- builds co-occurrence structure in decoder_W over time

  6. Save/restore conversation history
     -- currently only weights saved, not context
     -- save last 20 turns to JSON alongside brain_paths_smart.npz
     -- on load, re-inject last turn as warm-start stimulus
```

---

### `brain_core_cpu_gpt.py` — LIF + DistilGPT2 (needs PyTorch upgrade)

The brain state signal fed to GPT is a single percentage. GPT ignores it
because "firing 4.8%" contains no semantic information.

```
  IMPROVEMENTS:

  1. Upgrade PyTorch: pip install --upgrade torch
     (required before anything else)

  2. Replace firing% with concept words (borrow from v2)
     -- assign neuron regions to words (500 neurons per word)
     -- brain_state = "active concepts: cat, food, question"
     -- GPT can actually use this

  3. Add conversation memory to the prompt
     -- keep rolling buffer of last 6 turns
     -- prepend to every GPT call
     -- GPT stops treating each message as isolated

  4. Use gemma3:1b via ollama instead of DistilGPT2
     -- DistilGPT2 was trained to predict next token, not answer
     -- gemma3 is instruction-tuned, far better for chat
     -- replace GPT call with ollama HTTP POST (copy from brain_core.c)
     -- remove torch dependency entirely

  5. Cache GPT tokenizer encoding
     -- currently re-encodes common words every call
     -- pre-compute embeddings for vocab and store
```

---

### `brain_core_cpu_gpt_v2.py` — LIF + ConceptSpace + DistilGPT2

ConceptSpace is the right idea. Weakness: 200 concepts is too few, and the
GPT-2 model is not instruction-tuned so it rambles.

```
  IMPROVEMENTS:

  1. Expand ConceptSpace from 200 → 1000+ concepts
     -- 100k neurons / 1000 words = 100 neurons per concept
     -- still detectable, much richer semantic coverage

  2. Add concept co-activation learning
     -- when concept A and B are both active: strengthen A↔B link
     -- separate concept_association_W [1000 x 1000] matrix
     -- after N steps: if region_A.mean() > 0.1 and region_B.mean() > 0.1:
     --     concept_W[A,B] += 0.01
     -- decode: active concepts pull in associated concepts

  3. Replace DistilGPT2 with ollama call
     -- same benefit as gpt.py #4 above
     -- concept list becomes a very clean prompt prefix

  4. Add concept decay
     -- concept activations should fade if not reinforced
     -- prevents old words lingering in "active concepts" list
     -- concept_activation[i] *= 0.95 each step

  5. Weight concept activation by recency
     -- words mentioned 1 step ago: weight 1.0
     -- words mentioned 10 steps ago: weight 0.5
     -- exponential decay: weight = exp(-age / tau)
```

---

### `brain_core_learning.py` — LIF + GPT-2 + Hebbian pre-training

Pre-training is the right instinct. The weakness is that the training corpus
is tiny (hardcoded Q&A pairs) and GPT-2 still hallucinates.

```
  IMPROVEMENTS:

  1. Expand training corpus
     -- load conversation_training.json from disk
     -- add 500–1000 real Q&A pairs (extract from Wikipedia QA datasets)
     -- more training = better readout bias

  2. Add online reinforcement signal
     -- after each response, prompt user: "good/bad? (y/n)"
     -- if good: hebbian_update(W, user_pattern, response_pattern, +lr)
     -- if bad:  hebbian_update(W, user_pattern, response_pattern, -lr)
     -- brain learns which patterns produce approved responses

  3. Separate fast and slow learning rates
     -- slow_lr = 0.0001 for recurrent weights (don't forget old learning)
     -- fast_lr = 0.01  for readout weights (adapt quickly to new patterns)
     -- prevents catastrophic forgetting of earlier training

  4. Add weight decay to prevent saturation
     -- Hebbian learning with no decay → all weights → ±1 → no more change
     -- W *= (1 - decay)  each step, decay = 0.0001
     -- keeps weights in sensitive range

  5. Replace GPT-2 with ollama (same as others)
```

---

### `brain_core_controlled.py` — LIF + GPT-2 + TopicController

Topic control works but is brittle — it's just regex. A question outside the
defined patterns gets "unknown" topic and the lock fails.

```
  IMPROVEMENTS:

  1. Replace regex topics with embedding similarity
     -- topic_exemplars = {"math": ["calculate","number","equation",...]}
     -- embed user input, compute cosine similarity to each topic's centroid
     -- robust to novel phrasing: "what's 2 and 2?" still → math topic

  2. Add topic persistence across turns
     -- current: topic extracted fresh each turn from one message
     -- add: topic_history = rolling window of last 3 detected topics
     -- if current turn is "unknown": inherit previous topic
     -- conversation stays coherent over multiple exchanges

  3. Soften the topic lock
     -- current controlled prompt is very rigid → GPT output sounds robotic
     -- instead of "you MUST answer about X", use:
     --   "The conversation is about X. Relevant keywords: Y. User: ..."
     -- GPT stays on topic but sounds more natural

  4. Add multi-topic handling
     -- "what's the weather like for a physics experiment outside?"
     -- current: picks one topic, discards the other
     -- extract top-2 topics, include both in prompt

  5. Cache topic detection
     -- topic regex runs every call
     -- cache result per (input_hash → topic) to avoid recomputation
```

---

### `brain_core.c` — C + SDL2 + ollama (best right now)

The strongest working program. Weaknesses are small neuron count (10k vs 100k
in Python), fixed gemma3:1b model, and no concept-level brain state.

```
  IMPROVEMENTS:

  1. Increase neuron count: 10k → 50k
     -- change: #define N 50000 and #define GRID 224 (224^2 = 50176)
     -- CELL size: 700/224 = 3 pixels per neuron (still visible)
     -- adjust WIN_W to accommodate
     -- richer dynamics, better pattern separation

  2. Add concept regions to the C brain
     -- define 64 named concept regions (156 neurons each for 10k)
     -- struct ConceptRegion { char name[32]; int start; int end; };
     -- in llm_thread: measure region firing rates, format as word list
     -- send to ollama: "active: cat, question, food" not just "4.8%"

  3. Add colour-coded neuron map by region
     -- current: all neurons same colour (blue/white)
     -- assign each concept region a distinct hue
     -- render: SDL_SetRenderDrawColor by region index
     -- visual immediately shows which concepts are active

  4. Switch model at runtime
     -- add 'M' keypress → cycle through available ollama models
     -- ollama list → parse model names → rotate MODEL string
     -- test gemma3:270m (faster) vs gemma3:1b (smarter)

  5. Add STDP weight update in C
     -- current Hebbian: if both fired → strengthen (no timing)
     -- STDP: track last spike time t_last[N]
     -- delta_W = A * exp(-(t_post - t_pre) / tau) if pre before post
     -- more biologically accurate, better pattern separation

  6. Add a "sleep" mode
     -- if no user input for 60s: reduce global_drive, lower noise
     -- neurons enter slow-wave oscillation (like sleep consolidation)
     -- on wake: hebbian replay of recent exchanges
     -- borrow idea from memory consolidation neuroscience

  7. Stream ollama tokens to screen as they arrive
     -- current: wait for full response then display
     -- ollama streaming API returns tokens as newline-delimited JSON
     -- parse stream in llm_thread, call chat_add() per token
     -- feels much more alive
```

---

### `brain_core_combined.py` — Reservoir + Hebbian + Topic Control

The most feature-complete Python program. Main weakness: the reservoir weights
are fixed after init (Echo State Network design), so the recurrent net never
learns — only the readout layer trains. Also blocked by PyTorch.

```
  IMPROVEMENTS:

  1. Fix: pip install --upgrade torch  (unblocks it)

  2. Allow slow recurrent weight learning
     -- current: reservoir W is fixed (ESN design)
     -- add: W += slow_lr * hebbian_delta (very small lr = 0.00001)
     -- reservoir gradually adapts its dynamics to seen patterns
     -- keeps ESN stability while allowing slow plasticity

  3. Replace 1024-concept readout with GloVe embedding space
     -- current 1024 concepts are generated words (may be nonsense)
     -- readout_W: [50d x 100k], train to predict GloVe embedding of response
     -- decode: find nearest GloVe word to readout output
     -- same improvement as brain_core_cpu_smart.py #1

  4. Add working memory buffer
     -- reservoir state is one snapshot (no temporal history in readout)
     -- maintain ring buffer of last T=10 reservoir states
     -- readout sees concatenated [state_t, state_t-1, ..., state_t-9]
     -- brain can "remember" what was said 10 steps ago

  5. Replace GPT-2 with ollama (remove torch dependency)
     -- reservoir + Hebbian + TopicControl all work in pure NumPy
     -- only the response generator needs an LLM
     -- switch to ollama HTTP: requests.post("http://localhost:11434/api/generate")
     -- now works without any PyTorch at all

  6. Expose reservoir state as a live plot
     -- matplotlib.animation to show reservoir firing rate over time
     -- or pipe to a separate terminal with curses ASCII sparklines
     -- gives same insight as the SDL2 visual in brain_core.c
```

---

### `brain_core_reasoning.py` — 4-Layer Cortex (highest ceiling)

The best architecture. The 4-layer hierarchy (Perception → Association →
Decision → Output) is the strongest design in the series. Blocked by PyTorch
and limited by GPT-2 quality.

```
  IMPROVEMENTS:

  1. Fix: pip install --upgrade torch

  2. Add lateral inhibition within each layer
     -- current: layers are independent LIF populations
     -- add: winner-take-all inhibition within each layer
     -- top-K neurons per layer survive, others suppressed
     -- creates sparse, distinct representations per layer
     -- implementation: each step, keep top 5% spikes per layer,
     --                 zero the rest

  3. Add feedback connections (top-down)
     -- current: P→A→D→O (feedforward only)
     -- add: O→D, D→A, A→P (weak feedback, lr=0.01)
     -- top-down attention: decision layer biases what perception notices
     -- closer to actual cortical architecture

  4. Add a dedicated memory layer
     -- insert between Association and Decision:
     --   P → A → Memory → D → O
     -- Memory layer has very slow leak (leak=0.999) → activity persists
     -- effectively a working memory: holds context across many steps

  5. Replace GPT-2 with ollama
     -- reasoning trace is already the richest context of any program
     -- "layer activity: P=4%, A=6%, D=3%, O=5%, concepts: cat, food"
     -- gemma3 will use this far better than GPT-2 can
     -- same HTTP call as in brain_core.c

  6. Add inter-layer STDP
     -- strengthen P→A connections when P fires before A (within 5 steps)
     -- the forward connections learn to route meaningful patterns
     -- over time: specific inputs reliably activate specific decision patterns

  7. Visualise layer activity
     -- 4 horizontal bars below the chat: one per layer
     -- firing rate 0–100% shown as fill level
     -- shows information flow from input to decision in real time

  8. Target architecture for this project:
     -- merge brain_core.c visual engine with brain_core_reasoning.py layers
     -- C simulation loop for speed (brain_core.c brain_step)
     -- Python reasoning layer for flexibility (via subprocess or cffi)
     -- ollama for language (already in brain_core.c)
     -- this gives: speed + structure + quality + visualisation
```

---

## Recommended Development Path

If you want to keep improving this project, the clearest path forward is:

```
  NOW (no fixes needed)
     |
     +-- Extend brain_core_cpu_smart.py:
     |       add GloVe embeddings (step 1 above)
     |       expand to 10k word vocab
     |       add temperature sampling
     |
  AFTER: pip install --upgrade torch
     |
     +-- Get brain_core_reasoning.py running
     |       add lateral inhibition per layer
     |       add feedback connections
     |       replace GPT-2 with ollama HTTP call
     |
  FINAL TARGET
     |
     +-- Merge brain_core.c + brain_core_reasoning.py:
             C core: speed, SDL2 visual, ollama, persistence
             4-layer Python reasoning: P→A→D→O hierarchy
             GloVe concept space: semantic brain state
             STDP weight updates: genuine online learning
             Result: a real-time, visually observable,
                     semantically grounded, self-modifying
                     spiking neural organism

---

## `brain_core.c` — Deep Dive

### What it is in one sentence
A self-contained C program that runs 10,000 spiking neurons continuously, draws them
as a live pixel map, and lets you chat with an ollama LLM whose replies are shaped by
the brain's current firing state.

---

### The Data Structures

Two structs hold everything:

**`Brain`** — the neural network itself:
```c
float    v[10000]          // membrane voltage of each neuron
float    thr[10000]        // adaptive spike threshold (per neuron)
float    avg[10000]        // long-run firing rate estimate
uint8_t  spk[10000]        // 1 if neuron fired this step, else 0

uint16_t idx[10000][32]    // which 32 neurons feed into each neuron
float    W[10000][32]      // strength of each of those 32 connections

float    Wr[256][10000]    // readout weights: maps spikes -> 256 concepts
float    br[256]           // readout bias
float    out[256]          // last readout output
```

**`App`** — everything the UI and threads need:
```c
Brain*          brain          // pointer to the brain
pthread_mutex_t bmtx           // lock: only one thread touches brain at a time
pthread_mutex_t cmtx           // lock: only one thread touches chat at a time
uint8_t         snap_spk[N]    // copy of spike state for rendering
float           glow[N]        // visual decay trail per neuron
char            chat[80][256]  // ring buffer of chat messages
char            hu[6][256]     // last 6 user messages (conversation history)
char            ha[6][256]     // last 6 brain replies
char            inp[510]       // current typed input
```

---

### The Three Threads

```
  main thread                 sim_thread                  llm_thread
  (SDL2 event loop)           (runs always)               (spawned per submit)
        |                           |                             |
  poll SDL events             lock brain mutex            lock brain mutex
  handle keypresses           brain_step(NULL)            forward_pass(input, 80 steps)
  render_neurons()            unlock                      find top concept
  render_chat()               sleep 1ms                   unlock
  SDL_RenderPresent()         repeat                      build ollama prompt
        |                                                 ollama_query()
  on RETURN key:                                          lock brain mutex
      submit()                                            hebbian_learn(response)
          |                                               unlock
          spawn llm_thread ------>                        chat_add("Brain: ...")
          pthread_detach()                                brain_save()
```

The two mutexes prevent the sim and LLM threads from touching the brain at the same
time. The render thread never locks the brain — it reads from `snap_spk`, a copy taken
briefly under lock inside `render_neurons()`.

---

### `brain_step()` — One Simulation Tick

Runs ~1000 times per second in `sim_thread`. What happens to every neuron each call:

```
  For each neuron i (0-9999):

  1. NEIGHBOURHOOD SUM  (sliding window, O(N) not O(N^2))
     Look at the 16 neurons before and after i in the flat array.
     Sum their spike values. Multiply by 0.02.
     This spreads activity to nearby neurons — creates wave effects.

  2. SYNAPTIC INPUT
     Look up the 32 pre-synaptic neurons in idx[i][0..31].
     Multiply each pre-spike (0 or 1) by weight W[i][j].
     Sum them:  syn = sum( spk[idx[i][j]] * W[i][j] )

  3. TOTAL DRIVE
     tot = syn + neighbourhood + 0.05 (global bias) + 0.03*noise + external

  4. LEAKY INTEGRATION
     v[i] = v[i] * 0.97 + tot
     (voltage decays to 63% per step with no input)

  5. SPIKE CHECK
     if v[i] >= thr[i]:  spk[i] = 1,  v[i] = 0   (fire and reset)
     else:               spk[i] = 0

  6. HOMEOSTASIS
     avg[i] = 0.98 * avg[i] + 0.02 * spk[i]    (slow firing rate average)
     d = (avg[i] - 0.05) * 0.5                  (target = 5% firing rate)
     thr[i] = clamp(thr[i] + d, 0.5, 2.5)
     -- fires too much: threshold rises, harder to fire
     -- fires too little: threshold drops, easier to fire
```

---

### `encode_text()` — Turning Words into Stimulus

```c
for each character in the input string:
    h = char_value * 2654435761  XOR  (position * 12345)   // hash
    stim[h % 10000] += 0.5                                 // poke that neuron
```

A simple hash spreads characters across neuron indices. "hello" pokes 5 different
neurons with +0.5 current. Different words hit different neurons. Not semantic —
"cat" and "dog" land in random unrelated places — but deterministic: the same word
always activates the same neurons.

---

### `forward_pass()` — Running the Brain on Input

```c
encode_text(text, stim)         // build stimulus vector
for 80 steps:
    brain_step(brain, stim)     // inject stimulus each step, run dynamics
// then compute readout:
for each output unit o (0-255):
    out[o] = tanh( br[o] + sum( Wr[o][i] * spk[i] ) )
```

The readout is a linear layer (256 outputs, 10k inputs) applied to the spike pattern
after settling. `out[]` is 256 floats representing how strongly the current brain state
matches each of 256 learned concepts.

---

### `hebbian_learn()` — Updating Weights After a Response

```c
// hash each word of the ollama response into one of 256 output slots
for each word in response:
    h = hash(word)
    target[h % 256] = 1.0

// for each target output unit that was active:
for each o where target[o] == 1:
    for each neuron i:
        Wr[o][i] += 0.0005 * spk[i]   // strengthen if neuron fired
        clip Wr[o][i] to [-1, 1]
```

Hebbian learning on the readout layer only. The recurrent weights `W` are never updated;
only `Wr` changes. Over many interactions, the readout learns which spike patterns
correspond to which response words. "Neurons that fire together wire together."

---

### `ollama_query()` — Talking to the LLM

No libcurl, no Python, no HTTP library. Opens a raw TCP socket to `127.0.0.1:11434`
and hand-crafts the HTTP request:

```
POST /api/generate HTTP/1.0
Host: localhost:11434
Content-Type: application/json
Content-Length: <n>

{"model":"gemma3:1b","prompt":"<escaped prompt>","stream":false}
```

Reads the full response into a 128KB buffer, finds `\r\n\r\n` to locate the HTTP body,
then parses JSON by hand with `json_str()` — a simple `strstr` search for
`"response":"..."`. No external JSON parser.

The prompt sent to gemma3 looks like:
```
You are a hybrid spiking neural network brain. Respond concisely (1-3 sentences).
Active concept 147 (strength 0.83).

User: hello
Brain: Hi there, I'm active and running.
User: what are you?
Brain:
```

---

### The Visual — `render_neurons()`

10,000 neurons laid out as a 100x100 grid. Each neuron = one 7x7 pixel cell
(700x700 total). Colour is computed from two values per neuron:

```
g = glow[i]            // 1.0 when just spiked, decays *0.82 per frame (~60fps)
v = voltage[i] / 1.5   // normalised membrane voltage 0-1

R = g*255 + (1-g)*v*80
G = g*220 + (1-g)*(20 + v*130)
B = g*200 + (1-g)*(60 + v*100)
```

```
  Just spiked        -->  bright white/yellow  (g=1: R=255 G=220 B=200)
  High voltage        -->  warm blue-green      (v high, g decaying)
  Silent/resting      -->  dark blue-grey       (v=0, g=0)
```

Glow decays at 0.82x per frame, so a spike leaves a fading trail for ~8 frames.
Wave propagation is clearly visible as moving bright patches across the grid.

---

### Bitmap Font — `F8[95][8]`

No SDL_ttf, no font file. All 95 printable ASCII characters (space through tilde)
are a compile-time constant array. Each character is 8 rows of 8 bits. To draw:

```c
for row in 0..7:
    for col in 0..7:
        if F8[char-32][row] & (0x80 >> col):
            draw 2x2 pixel rectangle at (x + col*2, y + row*2)
```

Produces 16x16 pixel glyphs on screen. `TS=2` means each font pixel = a 2x2 SDL rect.

---

### Persistence — `brain_state.bin`

On quit, `/save`, or after each LLM response, writes a binary dump:

```
W[10000][32]    floats   recurrent weights    1.28 MB
Wr[256][10000]  floats   readout weights     10.24 MB
br[256]         floats   readout bias            1 KB
n_train         int      training count
n_interact      int      interaction count
```

Neuron voltages and thresholds are NOT saved — only the learned weights.
Brain always starts from a resting state but with all learned associations intact.

---

### Commands

| Key / Command | Action |
|---|---|
| Type text | Appends to input buffer |
| `Backspace` | Deletes last character |
| `Enter` | Submits — triggers LLM thread |
| `Ctrl+Q` or `Escape` | Saves brain, quits |
| `/save` | Manual save to `brain_state.bin` |
| `/stats` | Prints `train=N interact=N neurons=10000` |

---

### Section Summary

The `brain_core.c` deep dive covers:

- The two structs (`Brain` and `App`) with field-by-field annotation
- The three-thread architecture and how the mutexes coordinate them
- `brain_step()` — the 6-stage per-neuron update (neighbourhood, synaptic, drive, LIF, spike, homeostasis)
- `encode_text()` — the hash that maps characters to neuron indices
- `forward_pass()` — stimulus injection + readout computation
- `hebbian_learn()` — how the readout weights update from responses
- `ollama_query()` — the hand-rolled HTTP/TCP socket call and JSON parsing
- `render_neurons()` — the colour formula and glow decay maths
- The embedded bitmap font and how it renders glyphs
- `brain_state.bin` — what is and isn't saved (weights yes, voltages no)
- Commands table

---

## `brain_core.c` — Version 2 Changes

### What was added

**Input box scrolling**
The input box previously clipped long text at the right edge, making it impossible
to see what you were typing after a certain length. Fixed by calculating how many
characters fit in the panel width and showing only the tail of the input string,
so the cursor is always visible regardless of input length.

```c
/* before: always showed from start — text disappeared off right edge */
snprintf(disp, sizeof(disp), "> %s", app->inp);

/* after: scroll to tail so cursor stays visible */
int max_chars = (CP_W - 12) / GW - 2;
int start = (app->inl > max_chars) ? (app->inl - max_chars) : 0;
snprintf(disp, sizeof(disp), "> %s", app->inp + start);
```

---

**Ctrl-B / Ctrl-L mode switch**

A `mode` field was added to `App`. Two modes:

| Mode | Key | Behaviour |
|---|---|---|
| `MODE_LLM` | `Ctrl-L` | ollama replies, neurons learn from the response (default) |
| `MODE_BRAIN` | `Ctrl-B` | neurons reply directly from learned vocab, no LLM |

The header bar colour changes to indicate active mode:
- Yellow — `[LLM+LEARN]`
- Green — `[BRAIN]`

`submit()` now selects the thread function based on mode:
```c
void *fn = (app->mode == MODE_BRAIN) ? brain_thread : llm_thread;
pthread_create(&t, NULL, fn, a);
```

---

**Neuron vocabulary (`vocab[OUT][32]`)**

Added to the `Brain` struct: a 256-slot string table where each slot stores the
best word that hashes to that readout unit. Built up automatically during LLM mode.

```c
char vocab[OUT][32];   /* word label for each readout slot */
```

`hebbian_learn()` now stores words as it processes each LLM response:
```c
int slot = (int)(h % OUT);
target[slot] = 1.f;
if (b->vocab[slot][0] == '\0')
    strncpy(b->vocab[slot], tok, 31);   /* label this slot with the word */
```

After many LLM interactions the vocab table fills in, giving the neurons a real
word-level representation of what they have learned. Saved to `brain_state.bin`
so vocabulary persists across sessions.

---

**`brain_associate()` — recurrent weight learning**

Previously only the readout weights `Wr` were updated (Hebbian on the output layer).
The recurrent weights `W` were fixed after initialisation.

`brain_associate()` now also updates `W` — the connections between neurons — to
physically wire the user-input spike pattern to the LLM-response spike pattern:

```
  User types "hello"
       |
  forward_pass(user_text, 80 steps) --> spike pattern A in b->spk[]
  memcpy(b->pre_spk, b->spk, N)     --> snapshot A saved
       |
  ollama_query() --> LLM response
       |
  brain_associate(response):
      forward_pass(response, 40 steps)  --> spike pattern B in b->spk[]
      for each neuron i that fired in B (post):
          for each of its 32 inputs j:
              if pre_spk[idx[i][j]] fired in A:
                  W[i][j] += 0.0002     --> strengthen A->B connection
```

Over many interactions the recurrent network physically rewires itself to route
"what I heard" toward "what was replied." This is genuine associative memory in
the weight matrix, not just in the readout.

---

**`brain_reply()` — decoding neurons to text**

When in `MODE_BRAIN`, the neurons generate a reply without the LLM:

```
  forward_pass(user_text, 80 steps)
       |
  compute readout: out[o] = tanh(br[o] + sum(Wr[o][i] * spk[i]))
       |
  find top-6 readout slots where:
      - out[o] is highest
      - vocab[o] is not empty (slot has been labelled)
       |
  concatenate vocab words -> reply string
```

Early in the brain's life, most vocab slots are empty so replies will be short
("..."). After extensive LLM-mode training the brain accumulates vocabulary and
its replies become longer and more varied. The words come entirely from what the
LLM has said in previous sessions.

---

**Neuron drive/noise levels — fire on demand, not randomly**

Previously the neurons had hardcoded constants (`0.05` global drive, `0.03` noise)
that kept them firing continuously even with no input, producing meaningless
background activity.

Two global floats now control drive and noise, switched contextually:

```c
static float g_drive = 0.01f;   /* idle: neurons mostly quiet    */
static float g_noise = 0.005f;  /* idle: minimal random firing   */

/* inside forward_pass(): */
g_drive = 0.05f; g_noise = 0.03f;   /* boost: respond to stimulus */
for (int s = 0; s < steps; s++) brain_step(b, _stim);
g_drive = 0.01f; g_noise = 0.005f;  /* restore quiet idle state   */
```

Effect:
- **Idle** (sim_thread free-running): near-silent, only existing attractors
  sustain weak activity. The grid looks mostly dark.
- **Processing** (forward_pass during submit): neurons fire strongly in response
  to the stimulus. Spike waves are clearly input-driven, not background noise.
- **After processing**: activity fades back to near-silence within a few hundred
  steps as homeostasis re-stabilises.

---

**Save every interaction**

Both `llm_thread` and `brain_thread` now call `brain_save()` after every exchange.
The save format was extended to include `vocab[]`:

```
W[10000][32]      floats   recurrent weights         1.28 MB
Wr[256][10000]    floats   readout weights           10.24 MB
br[256]           floats   readout bias               1 KB
n_train           int      training count
n_interact        int      interaction count
vocab[256][32]    chars    word labels per slot       8 KB   <-- new
```

Old `brain_state.bin` files load correctly — the `fread` for vocab silently fails
on short files and leaves the vocab zeroed, which is the correct starting state.

---

### Updated Commands

| Key / Command | Action |
|---|---|
| Type text | Appends to input buffer (now scrolls) |
| `Backspace` | Deletes last character |
| `Enter` | Submits in current mode |
| `Ctrl-L` | Switch to LLM mode (ollama replies, neurons learn) |
| `Ctrl-B` | Switch to Brain mode (neurons reply directly) |
| `Ctrl-Q` or `Escape` | Saves brain, quits |
| `/save` | Manual save to `brain_state.bin` |
| `/stats` | Prints training and interaction counts |

---

### How the two modes work together

The intended workflow:

```
  Session 1 (Ctrl-L, LLM mode)
       |
       +-- Chat extensively with the LLM
       +-- Each reply: hebbian_learn() fills vocab[], updates Wr
       +-- Each reply: brain_associate() wires W from input->response patterns
       +-- Saves after every message
       |
  Session 2 (same, more learning)
       |
  Session N — brain has heard hundreds of responses
       |
  Switch to Ctrl-B (Brain mode)
       |
       +-- Type a question
       +-- Neurons run forward_pass on your input
       +-- Readout finds top active vocab slots
       +-- Reply comes entirely from learned neuron state
       |
  Compare: does the brain's reply relate to what the LLM would have said?
  If not: back to Ctrl-L for more training.
```

The brain's replies will never match the LLM's fluency — they are top-K word
lookups, not language generation. But they will reflect the actual associative
structure learned from real LLM responses over real interactions.

---

## `brain_core.c` — Version 3 Changes (1 Million Neurons)

### What changed

**Scale-up: 10,000 → 1,000,000 neurons**

| Parameter | Before | After |
|---|---|---|
| `N` neurons | 10,000 | **1,000,000** |
| `GRID` | 100×100 | **1000×1000** |
| Neuron display | 700×700px (7px/neuron) | **1000×1000 texture scaled to 690×690px** |
| `idx` index type | `uint16_t` (max 65,535) | **`uint32_t`** |
| Neighbourhood radius | 16 | **50** |
| sim_thread sleep | 1 ms | **none** |
| render mutex | blocking lock | **trylock** |
| Brain struct RAM | ~12 MB | **~1.28 GB** |
| Save file size | ~11 MB | **~1.15 GB** |

---

### Why `uint16_t` had to change

The original index array was `uint16_t idx[N][FAN]`. `uint16_t` can only hold
values up to 65,535. With N=1,000,000 neurons, any index above 65,535 would
silently wrap around and point to the wrong neuron, corrupting all connectivity.

Changed to `uint32_t` which handles up to ~4 billion:

```c
/* before */
uint16_t idx[N][FAN];
b->idx[i][j] = (uint16_t)(rand() % N);   /* wrong: wraps at 65535 */

/* after */
uint32_t idx[N][FAN];
b->idx[i][j] = (uint32_t)((unsigned)rand() % (unsigned)N);
```

---

### Display: texture scaling instead of pixel-per-neuron

At 10k neurons (100×100 grid), each neuron was drawn as a 7×7 pixel cell
(700px total). At 1M neurons (1000×1000 grid), 7px per neuron would need a
7000×7000 window.

Instead, the texture is always created at GRID×GRID (1000×1000 pixels, one pixel
per neuron) and SDL scales it down to fit the 690×690 display panel:

```c
/* texture: 1000×1000 — one pixel per neuron */
SDL_CreateTexture(rdr, SDL_PIXELFORMAT_ARGB8888,
                  SDL_TEXTUREACCESS_STREAMING, GRID, GRID);

/* display: scaled to fit panel */
SDL_Rect dst = {NP_X, NP_Y, NP_W, NP_W};   /* NP_W = 690 */
SDL_RenderCopy(rdr, ntex, NULL, &dst);       /* SDL scales 1000->690 */
```

SDL uses bilinear filtering for the downscale, so nearby active neuron clusters
appear as bright blobs rather than individual pixels. At this scale you see
population-level activity (waves, columns, patches) not individual spikes.

---

### sim_thread: no sleep

At 10k neurons, `brain_step` takes ~0.1ms so a 1ms sleep kept it at ~900
steps/sec — intentional throttling for a responsive UI.

At 1M neurons, `brain_step` takes ~100–500ms (32M random memory accesses, one
per synapse per step). The step itself is the bottleneck. No sleep needed:

```c
while(app->run){
    pthread_mutex_lock(&app->bmtx);
    brain_step(app->brain, NULL);
    pthread_mutex_unlock(&app->bmtx);
    /* no sleep — step is slow enough at 1M neurons */
}
```

Expected step rate: 1–5 steps/sec on a single CPU core.

---

### render_neurons: trylock instead of blocking lock

At 10k neurons, `brain_step` was fast (~0.1ms) so the render thread rarely had
to wait for the mutex. At 1M neurons, holding the mutex during a 100–500ms step
would freeze the display for that entire duration — the window would appear to
stutter or lock up.

Fix: `pthread_mutex_trylock`. If the sim thread is mid-step, the render thread
skips the snapshot update and redraws with the previous frame's data. The
display runs at full 60fps regardless of simulation speed:

```c
if(pthread_mutex_trylock(&app->bmtx) == 0){
    memcpy(snap_spk, brain->spk, N);   /* 1MB copy */
    memcpy(snap_v,   brain->v,   N*4); /* 4MB copy */
    snap_train = brain->n_train;
    pthread_mutex_unlock(&app->bmtx);
}
/* else: reuse last snapshot — 1 step stale, visually fine */
```

---

### Memory layout at 1M neurons

```
Brain struct breakdown:
  v[1M]           float    4 MB   membrane voltages
  thr[1M]         float    4 MB   adaptive thresholds
  avg[1M]         float    4 MB   firing rate averages
  spk[1M]         uint8    1 MB   spike flags (current step)
  pre_spk[1M]     uint8    1 MB   spike snapshot (before LLM call)
  idx[1M][32]     uint32 128 MB   pre-synaptic indices
  W[1M][32]       float  128 MB   recurrent weights
  Wr[256][1M]     float  1024 MB  readout weights   <-- dominates
  vocab[256][32]  char      8 KB  word labels
  ───────────────────────────────────────────────
  Total Brain:           ~1.28 GB

App struct additions:
  snap_spk[1M]    uint8    1 MB   render snapshot
  snap_v[1M]      float    4 MB   render snapshot
  glow[1M]        float    4 MB   visual decay
  ───────────────────────────────────────────────
  Total App:             ~9 MB
```

Linux allocates pages lazily — at startup `VmRSS` (physical RAM used) is only
~244 MB. It grows toward 1.28 GB as the simulation touches each memory page
during the first few hundred steps.

---

### Save file at 1M neurons

```
W[1M][32]       float   128 MB
Wr[256][1M]     float  1024 MB
br[256]         float     1 KB
n_train         int       4 B
n_interact      int       4 B
vocab[256][32]  char      8 KB
─────────────────────────────
Total:                ~1.15 GB
```

Saving takes 2–5 seconds on a modern SSD. This happens automatically after
every LLM interaction and every `/c` loop exchange.

---

### Expected performance

```
  Operation               Time (single CPU core, estimate)
  ─────────────────────────────────────────────────────────
  brain_step (idle)       100–500 ms    1–5 steps/sec
  forward_pass (80 steps) 8–40 seconds  runs during LLM call
  brain_associate (40 st) 4–20 seconds
  brain_save              2–5 seconds   SSD write of 1.15 GB
  render frame            ~16 ms        60 fps, never blocked
  ─────────────────────────────────────────────────────────
```

The simulation is slow on CPU at this scale. The `/c` learning loop exchanges
will each take 10–60 seconds end-to-end (ollama + forward passes + save).
This is expected — 1M neurons on a single core is a real workload.

To go faster at this scale: multithreaded `brain_step` splitting the N loop
across cores, or move to GPU (the Python `brain_core.py` with CuPy handles
200k neurons at 0.017s/step — 1M on GPU would be ~0.08s/step).
