import os
import time
import json
import cupy as cp
import numpy as np
from transformers import GPT2Tokenizer

PATH_FILE = "brain_paths.npz"


# -------------------------------------------------------
#  Neuron model
# -------------------------------------------------------
class HybridNeuronModel:
    """
    Simple hybrid spiking neuron:
    - Leaky integrate-and-fire voltage
    - Threshold
    - State machine (for later neuron modes)
    - Spike vector (bool) from previous step
    """

    def __init__(self, n_neurons):
        self.n = n_neurons

        # Floating point neuron dynamics
        self.voltage   = cp.zeros(n_neurons, dtype=cp.float32)
        self.threshold = cp.ones(n_neurons, dtype=cp.float32) * 1.5
        self.leak      = cp.ones(n_neurons, dtype=cp.float32) * 0.97

        # Spiking + state-machine
        self.state               = cp.zeros(n_neurons, dtype=cp.uint8)
        self.spike               = cp.zeros(n_neurons, dtype=cp.bool_)
        self.refractory_counter  = cp.zeros(n_neurons, dtype=cp.uint8)

    def update(self, inputs):
        """
        inputs: GPU vector of synaptic + external drive
        """

        # Leaky integration
        self.voltage = self.voltage * self.leak + inputs

        # Spike rule
        self.spike = (self.voltage >= self.threshold)

        # Reset voltage on spike
        self.voltage = cp.where(self.spike, 0.0, self.voltage)

        # Simple state machine:
        fired = self.spike.astype(cp.uint8)
        self.state = cp.where(fired == 1, 1, self.state)

        # State-dependent thresholds
        # (right now just 2 modes: default vs spiked-before)
        self.threshold = cp.where(self.state == 1, 1.5, 1.0)

        return self.spike


# -------------------------------------------------------
#  GPU Brain core + learning readout + text I/O
#  NOW WITH: GPT-2 tokenizer, 2D cortex, attractor ensembles, eligibility traces, reward plasticity
# -------------------------------------------------------
class GPUNeuronBrain:
    """
    GPU 'brain core' with:
      - 2D cortical sheet (500×500 = 250k neurons)
      - GPT-2 tokenizer for vocabulary
      - Dedicated input/output patches for better topography
      - Stronger attractor ensembles (recurrent excitation within patterns)
      - Eligibility traces for temporal credit assignment
      - Reward-based plasticity (dopamine-like modulation)
      - Homeostatic firing-rate control
    """

    def __init__(self, grid_size=500, fan_in=32, radius=16):
        self.grid_size = grid_size
        self.n_neurons = grid_size * grid_size  # 250,000
        self.fan_in = fan_in
        self.radius = radius

        # Neuron model (on GPU)
        self.neurons = HybridNeuronModel(self.n_neurons)

        # Random sparse connectivity
        self.inputs_idx = cp.random.randint(
            0, self.n_neurons, size=(self.n_neurons, fan_in), dtype=cp.int32
        )
        self.weights = (0.1 * cp.random.randn(self.n_neurons, fan_in)).astype(cp.float32)

        # === ELIGIBILITY TRACES for temporal credit assignment ===
        self.eligibility = cp.zeros((self.n_neurons, fan_in), dtype=cp.float32)
        self.eligibility_decay = 0.95  # traces decay over time
        self.eligibility_boost = 0.1   # how much to boost on coincident pre/post activity

        # === REWARD-BASED PLASTICITY ===
        self.reward_signal = 0.0  # global reward (-1 to +1)
        self.reward_lr = 0.001    # REDUCED from 0.005 to prevent runaway
        self.reward_decay = 0.95  # faster decay

        # === WEIGHT REGULARIZATION ===
        self.weight_decay = 0.9995  # decay weights slightly each step to prevent explosion

        # Per-neuron bias (can stimulate regions later)
        self.bias = cp.zeros(self.n_neurons, dtype=cp.float32)

        # Global low-level tonic drive
        self.global_drive = 0.05

        # Noise
        self.noise_sigma = 0.03

        # Neighborhood excitation multiplier (now in 2D)
        self.neigh_strength = 0.01  # REDUCED from 0.02 to limit excitation

        # === GLOBAL INHIBITION (Fix #1: Prevent runaway excitation) ===
        self.global_inhibition = 0.15  # stronger inhibition

        # Homeostatic firing-rate smoothing (Fix #1: Stronger control)
        self.firing_avg = cp.zeros(self.n_neurons, dtype=cp.float32)
        self.target_rate = 0.05  # aim for ~5% of neurons firing each step
        self.homeostatic_strength = 1.0  # INCREASED for stronger control

        # === 2D CORTICAL LAYOUT with I/O patches ===
        # Define dedicated input and output patches
        self.input_patch_size = 50   # 50×50 = 2500 neurons for input
        self.output_patch_size = 50  # 50×50 = 2500 neurons for output

        # Input patch: top-left corner (0:50, 0:50)
        self.input_patch_start = (0, 0)
        # Output patch: bottom-right corner (450:500, 450:500)
        self.output_patch_start = (grid_size - self.output_patch_size,
                                   grid_size - self.output_patch_size)

        # --- Text I/O: use GPT-2 tokenizer ---
        print("Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Use reduced vocabulary to fit in GPU memory
        self.vocab_size = 8192  # Reduced from 50257 to save GPU memory
        print(f"  ... Tokenizer loaded with {self.vocab_size} tokens (reduced from {self.tokenizer.vocab_size}).")

        # Readout weights: output patch spikes -> logits over tokens
        self.output_patch_size_flat = self.output_patch_size * self.output_patch_size
        self.readout_W = cp.zeros((self.output_patch_size_flat, self.vocab_size),
                                  dtype=cp.float32)
        self.readout_b = cp.zeros(self.vocab_size, dtype=cp.float32)

        # Learning rate for readout (Fix #2: Lower learning rate)
        self.lr = 0.005  # REDUCED from 0.01

        # === ATTRACTOR LEARNING (Fix #2: Weaker to prevent lock-in) ===
        self.attractor_lr = 0.02  # REDUCED from 0.08
        self.attractor_strength = 0.1  # REDUCED from 0.2
        self.attractor_recurrent_boost = 0.05  # REDUCED from 0.15

        # === SHORT-TERM MEMORY: Activity traces ===
        self.activity_trace = cp.zeros(self.n_neurons, dtype=cp.float32)
        self.trace_decay = 0.98
        self.trace_boost = 0.05

        # Try to load saved pathways
        self._load_paths_if_any()

    # ---------------------------------------------------
    #  2D coordinate helpers
    # ---------------------------------------------------
    def _idx_to_2d(self, idx):
        return (idx // self.grid_size, idx % self.grid_size)

    def _2d_to_idx(self, row, col):
        return row * self.grid_size + col

    def _get_patch_indices(self, start_row, start_col, size):
        indices = []
        for r in range(start_row, start_row + size):
            for c in range(start_col, start_col + size):
                indices.append(self._2d_to_idx(r, c))
        return cp.array(indices, dtype=cp.int32)

    # ---------------------------------------------------
    #  Core dynamics with 2D topology
    # ---------------------------------------------------
    def step(self, external_input=None):
        presyn = self.neurons.spike.astype(cp.float32)
        incoming = presyn[self.inputs_idx]
        syn_current = cp.sum(incoming * self.weights, axis=1)
        presyn_2d = presyn.reshape(self.grid_size, self.grid_size)
        neigh_sum = cp.zeros_like(presyn_2d)

        for offset in range(-self.radius, self.radius + 1):
            if offset != 0:
                neigh_sum += cp.roll(presyn_2d, offset, axis=0)
                neigh_sum += cp.roll(presyn_2d, offset, axis=1)

        neigh_drive = self.neigh_strength * neigh_sum.flatten()
        attractor_drive = self.attractor_recurrent_boost * self.activity_trace * presyn

        # FIX #1: Add global inhibition proportional to network activity
        global_activity = cp.mean(presyn)
        global_inhib = -self.global_inhibition * global_activity

        total = syn_current + neigh_drive + attractor_drive + self.bias + self.global_drive + global_inhib
        total += self.noise_sigma * cp.random.randn(self.n_neurons, dtype=cp.float32)

        if external_input is not None:
            ext = cp.asarray(external_input, dtype=cp.float32)
            total += ext

        # FIX #1: Stronger homeostatic control
        self.firing_avg = 0.95 * self.firing_avg + 0.05 * presyn  # faster tracking
        delta = (self.firing_avg - self.target_rate) * self.homeostatic_strength
        self.neurons.threshold = cp.clip(self.neurons.threshold + delta, 0.5, 3.0)
        spikes = self.neurons.update(total)
        post_active = spikes.astype(cp.float32)[:, cp.newaxis]
        pre_active = presyn[self.inputs_idx]
        self.eligibility *= self.eligibility_decay
        self.eligibility += self.eligibility_boost * post_active * pre_active
        self.activity_trace *= self.trace_decay
        self.activity_trace += self.trace_boost * spikes.astype(cp.float32)

        if abs(self.reward_signal) > 0.01:
            weight_delta = self.reward_lr * self.reward_signal * self.eligibility
            self.weights += weight_delta

        # FIX #2: Weight decay to prevent runaway weights
        self.weights *= self.weight_decay
        self.weights = cp.clip(self.weights, -1.0, 1.0)

        self.reward_signal *= self.reward_decay
        return spikes

    # ---------------------------------------------------
    #  Text encoding / decoding with GPT-2 tokenizer
    # ---------------------------------------------------
    def _encode_text_to_input(self, text):
        stim = cp.zeros(self.n_neurons, dtype=cp.float32)
        input_indices = self._get_patch_indices(
            self.input_patch_start[0], self.input_patch_start[1], self.input_patch_size
        )
        n_input_neurons = len(input_indices)
        if n_input_neurons == 0: return stim

        token_ids = self.tokenizer.encode(text)

        for token_id in token_ids:
            cp.random.seed(token_id)
            n_active_neurons = 10
            neuron_indices_in_patch = cp.random.choice(n_input_neurons, n_active_neurons, replace=False)
            neuron_indices_global = input_indices[neuron_indices_in_patch]
            stim[neuron_indices_global] += 1.5

        return stim

    def _decode_token_from_spikes(self, spikes, temperature=0.5):
        output_indices = self._get_patch_indices(
            self.output_patch_start[0], self.output_patch_start[1], self.output_patch_size
        )
        x = spikes[output_indices].astype(cp.float32)
        logits = x @ self.readout_W + self.readout_b
        logits_np = cp.asnumpy(logits)

        if temperature <= 0:
            j = int(np.argmax(logits_np))
        else:
            logits_np /= temperature
            exps = np.exp(logits_np - np.max(logits_np))
            probs = exps / np.sum(exps)
            j = np.random.choice(self.vocab_size, p=probs)

        # Clip to valid vocabulary range
        j = min(j, self.vocab_size - 1)
        return j

    def _train_readout(self, spikes, target_token_id):
        if target_token_id is None:
            return

        # Skip training if token is outside our reduced vocabulary
        if target_token_id >= self.vocab_size:
            return

        output_indices = self._get_patch_indices(
            self.output_patch_start[0], self.output_patch_start[1], self.output_patch_size
        )
        x = spikes[output_indices].astype(cp.float32)
        logits = x @ self.readout_W + self.readout_b
        logits_np = cp.asnumpy(logits)
        exps = np.exp(logits_np - logits_np.max())
        probs = exps / exps.sum()
        grad = probs
        grad[target_token_id] -= 1.0
        grad_cp = cp.asarray(grad, dtype=cp.float32)

        # Memory-efficient update: iterate instead of outer product
        for i in range(len(x)):
            if x[i] > 0:  # only update for active neurons
                self.readout_W[i, :] -= self.lr * x[i] * grad_cp

        self.readout_b -= self.lr * grad_cp

    def _form_attractor(self, stim_mask):
        active_indices = cp.where(stim_mask)[0]
        if len(active_indices) == 0: return

        is_in_active_set = cp.zeros(self.n_neurons, dtype=cp.bool_)
        is_in_active_set[active_indices] = True
        inputs_in_set = is_in_active_set[self.inputs_idx]
        post_in_set = is_in_active_set[:, cp.newaxis]
        strengthen_mask = inputs_in_set & post_in_set
        current_weights = self.weights[strengthen_mask]
        new_weights = current_weights + self.attractor_lr * (self.attractor_strength - current_weights)
        self.weights[strengthen_mask] = new_weights

    # ---------------------------------------------------
    #  Reward signal control
    # ---------------------------------------------------
    def set_reward(self, reward):
        self.reward_signal = float(np.clip(reward, -1.0, 1.0))
        print(f"  [Reward signal set to {self.reward_signal:.2f}]")

    # ---------------------------------------------------
    #  Interaction: one input line -> training + reply
    # ---------------------------------------------------
    def interact_once(self, text, settle_steps=100, reply_len=16, train=True, temperature=0.5):
        if not text:
            return "", []

        ext = self._encode_text_to_input(text)
        spikes = None
        spike_counts = []
        for _ in range(settle_steps):
            spikes = self.step(external_input=ext)
            spike_counts.append(int(cp.sum(spikes)))

        if train:
            stim_mask = (ext > 0)
            self._form_attractor(stim_mask)
            token_ids = self.tokenizer.encode(text)

            # FIX #3: Supervised learning - train readout on settled spikes -> input tokens
            # Train multiple times on the same settled state for better learning
            for _ in range(3):  # repeat training for better retention
                for token_id in token_ids:
                    if token_id < self.vocab_size:  # skip out-of-range tokens
                        self._train_readout(spikes, token_id)

        reply_tokens = []
        last_token_stim = None
        for _ in range(reply_len):
            spikes = self.step(external_input=last_token_stim)
            next_token_id = self._decode_token_from_spikes(spikes, temperature=temperature)
            reply_tokens.append(next_token_id)
            next_text = self.tokenizer.decode([next_token_id])
            last_token_stim = self._encode_text_to_input(next_text)

        reply = self.tokenizer.decode(reply_tokens)
        return reply, spike_counts

    # ---------------------------------------------------
    #  Persistence
    # ---------------------------------------------------
    def _load_paths_if_any(self):
        if os.path.exists(PATH_FILE):
            try:
                print("Loading saved pathways from", PATH_FILE, "...")
                data = np.load(PATH_FILE)

                if "readout_W" in data:
                    old_shape = data["readout_W"].shape
                    expected_shape = (self.output_patch_size_flat, self.vocab_size)
                    if old_shape != expected_shape:
                        print(f"  Warning: readout shape mismatch {old_shape} vs {expected_shape}. Starting fresh.")
                    else:
                        self.readout_W = cp.asarray(data["readout_W"])
                        self.readout_b = cp.asarray(data["readout_b"])
                        print("  ... Loaded readout weights.")

                if "weights" in data:
                    if data["weights"].shape[0] != self.n_neurons:
                        print(f"  Warning: neuron count mismatch. Starting fresh.")
                    else:
                        self.weights = cp.asarray(data["weights"])
                        print("  ... Loaded core weights.")

                if "eligibility" in data:
                    self.eligibility = cp.asarray(data["eligibility"])
                    print("  ... Loaded eligibility traces.")

            except Exception as e:
                print("Could not load pathways:", e)
        else:
            print("No saved pathways found. Starting fresh.")

    def save_paths(self):
        print("Saving pathways to", PATH_FILE, "...")
        np.savez_compressed(
            PATH_FILE,
            readout_W=cp.asnumpy(self.readout_W),
            readout_b=cp.asnumpy(self.readout_b),
            weights=cp.asnumpy(self.weights),
            eligibility=cp.asnumpy(self.eligibility),
        )
        print("... Done.")

    def get_stats(self):
        return {
            "firing_rate": float(cp.mean(self.firing_avg)),
            "mean_weight": float(cp.mean(self.weights)),
            "mean_eligibility": float(cp.mean(self.eligibility)),
            "mean_activity_trace": float(cp.mean(self.activity_trace)),
            "reward_signal": self.reward_signal,
        }

# -------------------------------------------------------
#  Main interactive loop
# -------------------------------------------------------
def main():
    grid_size = 500
    fan_in = 32
    radius = 16
    temperature = 0.5

    print(f"Creating 2D brain with {grid_size}×{grid_size} = {grid_size*grid_size} neurons")
    print(f"  fan_in={fan_in}, radius={radius}")
    print(f"  Input patch: top-left 50×50")
    print(f"  Output patch: bottom-right 50×50")
    print(f"  Features: GPT-2 Tokenizer, Attractor ensembles, Eligibility traces, Reward plasticity")

    brain = GPUNeuronBrain(grid_size=grid_size, fan_in=fan_in, radius=radius)

    print("\nBrain running. Commands:")
    print("  <text>         - Normal input")
    print("  /good          - Give positive reward (+1.0)")
    print("  /bad           - Give negative reward (-1.0)")
    print("  /reward <val>  - Set custom reward (-1 to +1)")
    print("  /temp <val>    - Set sampling temperature (e.g., 0.5, 1.0)")
    print("  /stats         - Show brain statistics")
    print("  CTRL-C to stop.\n")

    try:
        while True:
            try:
                user = input(f"You (temp={temperature:.2f}): ")
            except EOFError:
                break

            user = user.rstrip("\n")
            if not user:
                continue

            if user == "/good":
                brain.set_reward(1.0)
                continue
            if user == "/bad":
                brain.set_reward(-1.0)
                continue
            if user.startswith("/reward"):
                parts = user.split()
                if len(parts) == 2:
                    try:
                        brain.set_reward(float(parts[1]))
                    except ValueError:
                        print("Usage: /reward <value> (between -1 and 1)")
                else:
                    print("Usage: /reward <value>")
                continue
            if user.startswith("/temp"):
                parts = user.split()
                if len(parts) == 2:
                    try:
                        val = float(parts[1])
                        if val >= 0:
                            temperature = val
                            print(f"  [Temperature set to {temperature:.2f}]")
                        else:
                            print("Temperature must be >= 0.")
                    except ValueError:
                        print("Usage: /temp <value> (e.g. 0.5)")
                else:
                    print("Usage: /temp <value>")
                continue
            if user == "/stats":
                stats = brain.get_stats()
                print("\n--- Brain Statistics ---")
                for key, val in stats.items():
                    print(f"  {key}: {val:.4f}")
                print("------------------------\n")
                continue

            reply, spike_counts = brain.interact_once(user, reply_len=16, temperature=temperature)

            if len(spike_counts) >= 4:
                indices = np.linspace(0, len(spike_counts) - 1, 4, dtype=int)
                for i in indices:
                    print(f"   [spikes={spike_counts[i]}]")
            else:
                for c in spike_counts:
                    print(f"   [spikes={c}]")

            print("Brain:", reply)

    except KeyboardInterrupt:
        print("\nStopping brain and saving pathways.")

    brain.save_paths()

if __name__ == "__main__":
    main()
