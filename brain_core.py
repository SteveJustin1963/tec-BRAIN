import os
import time
import json
import cupy as cp
import numpy as np

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
# -------------------------------------------------------
class GPUNeuronBrain:
    """
    GPU 'brain core' with:
      - local-neighborhood excitation (wave-like behaviour)
      - homeostatic firing-rate control (keeps it “alive”)
      - simple trainable readout (spikes -> characters)
      - text I/O and persistence to disk
    """

    def __init__(self, n_neurons=200_000, fan_in=32, radius=16):
        self.n_neurons = n_neurons
        self.fan_in = fan_in
        self.radius = radius

        # Neuron model (on GPU)
        self.neurons = HybridNeuronModel(n_neurons)

        # Random sparse connectivity
        self.inputs_idx = cp.random.randint(
            0, n_neurons, size=(n_neurons, fan_in), dtype=cp.int32
        )
        self.weights = (0.1 * cp.random.randn(n_neurons, fan_in)).astype(cp.float32)

        # Per-neuron bias (can stimulate regions later)
        self.bias = cp.zeros(n_neurons, dtype=cp.float32)

        # Global low-level tonic drive
        self.global_drive = 0.05

        # Noise
        self.noise_sigma = 0.03

        # Neighborhood excitation multiplier
        self.neigh_strength = 0.02

        # Homeostatic firing-rate smoothing
        self.firing_avg = cp.zeros(n_neurons, dtype=cp.float32)
        self.target_rate = 0.05  # aim for ~5% of neurons firing each step

        # --- Text I/O: define vocabulary and readout layer ---
        # Use basic printable ASCII (space..~)
        self.vocab = [chr(c) for c in range(32, 127)]
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        # Readout weights: spikes -> logits over characters
        # shape: (n_neurons, vocab_size)
        self.readout_W = cp.zeros((self.n_neurons, self.vocab_size),
                                  dtype=cp.float32)
        self.readout_b = cp.zeros(self.vocab_size, dtype=cp.float32)

        # Learning rate for readout
        self.lr = 0.01

        # Try to load saved pathways
        self._load_paths_if_any()

    # ---------------------------------------------------
    #  Core dynamics
    # ---------------------------------------------------
    def step(self, external_input=None):
        """
        One update tick:
        - Sparse random connectivity
        - Local wave propagation
        - Homeostatic firing-rate control
        """

        presyn = self.neurons.spike.astype(cp.float32)

        # 1) Synaptic current from random graph
        incoming = presyn[self.inputs_idx]
        syn_current = cp.sum(incoming * self.weights, axis=1)

        # 2) Neighborhood wave drive (circular 1D “ring”)
        neigh_sum = cp.zeros(self.n_neurons, dtype=cp.float32)
        for offset in range(-self.radius, self.radius + 1):
            if offset != 0:
                neigh_sum += cp.roll(presyn, offset)
        neigh_drive = self.neigh_strength * neigh_sum

        # 3) Combine drives
        total = syn_current + neigh_drive + self.bias + self.global_drive

        # 4) Add noise
        total += self.noise_sigma * cp.random.randn(self.n_neurons,
                                                    dtype=cp.float32)

        # 5) Optional external input
        if external_input is not None:
            ext = cp.asarray(external_input, dtype=cp.float32)
            total += ext

        # 6) Homeostatic control (per-neuron)
        self.firing_avg = 0.98 * self.firing_avg + 0.02 * presyn
        delta = (self.firing_avg - self.target_rate) * 0.5
        self.neurons.threshold = cp.clip(self.neurons.threshold + delta,
                                         0.5, 2.5)

        # 7) Update neuron dynamics
        spikes = self.neurons.update(total)
        return spikes

    # ---------------------------------------------------
    #  Text encoding / decoding
    # ---------------------------------------------------
    def _encode_text_to_input(self, text):
        """
        Very simple encoder:
          - For each character, activate a small block of neurons.
        This is deliberately crude; the GPU core just acts as a big
        noisy “reservoir” the readout will learn to interpret.
        """
        stim = cp.zeros(self.n_neurons, dtype=cp.float32)
        block_size = 16

        for ch in text:
            idx = self.char_to_idx.get(ch)
            if idx is None:
                continue
            base = (idx * block_size) % self.n_neurons
            end = base + block_size
            if end <= self.n_neurons:
                stim[base:end] += 1.0
            else:
                # wrap around at end
                wrap = end - self.n_neurons
                stim[base:] += 1.0
                stim[:wrap] += 1.0

        return stim

    def _decode_char_from_spikes(self, spikes):
        """
        Map a spike pattern to a single character via the readout layer.
        """
        x = spikes.astype(cp.float32)  # shape (n_neurons,)
        logits = x @ self.readout_W + self.readout_b  # (vocab_size,)

        # Work in NumPy for softmax / argmax for simplicity
        logits_np = cp.asnumpy(logits)
        j = int(np.argmax(logits_np))
        return self.idx_to_char.get(j, "?")

    def _train_readout(self, spikes, target_char):
        """
        One-step training of the readout weights using a simple
        softmax cross-entropy gradient.
        """
        idx = self.char_to_idx.get(target_char)
        if idx is None:
            return

        x = spikes.astype(cp.float32)  # (n_neurons,)
        logits = x @ self.readout_W + self.readout_b  # (vocab_size,)

        logits_np = cp.asnumpy(logits)
        # softmax
        exps = np.exp(logits_np - logits_np.max())
        probs = exps / exps.sum()

        # gradient dL/dlogits = probs - y_onehot
        grad = probs
        grad[idx] -= 1.0

        grad_cp = cp.asarray(grad, dtype=cp.float32)

        # Gradient step: W := W - lr * outer(x, grad)
        self.readout_W -= self.lr * cp.outer(x, grad_cp)
        self.readout_b -= self.lr * grad_cp

    # ---------------------------------------------------
    #  Interaction: one input line -> training + reply
    # ---------------------------------------------------
    def interact_once(self, text, settle_steps=80, reply_len=8, train=True):
        """
        - Encode the user's text as external drive
        - Let the brain settle while being driven
        - Train readout to associate that state with (for now)
          the last character of the input
        - Then run a few free steps and read out characters
        """
        if not text:
            return "", []

        ext = self._encode_text_to_input(text)

        # Let the network settle with this drive
        spikes = None
        spike_counts = []
        for _ in range(settle_steps):
            spikes = self.step(external_input=ext)
            spike_counts.append(int(cp.sum(spikes)))

        # Train readout on the final state for this input
        if train:
            target_char = text[-1]  # crude, but something to hook onto
            self._train_readout(spikes, target_char)

        # Now free-run the brain (no external input) and read characters
        reply_chars = []
        for _ in range(reply_len):
            spikes = self.step(external_input=None)
            reply_chars.append(self._decode_char_from_spikes(spikes))

        reply = "".join(reply_chars)
        return reply, spike_counts

    # ---------------------------------------------------
    #  Persistence
    # ---------------------------------------------------
    def _load_paths_if_any(self):
        if os.path.exists(PATH_FILE):
            try:
                data = np.load(PATH_FILE)
                self.readout_W = cp.asarray(data["readout_W"])
                self.readout_b = cp.asarray(data["readout_b"])
                print("Loaded saved pathways from", PATH_FILE)
            except Exception as e:
                print("Could not load pathways:", e)
        else:
            print("No saved pathways found. Starting fresh.")

    def save_paths(self):
        data = {
            "readout_W": cp.asnumpy(self.readout_W),
            "readout_b": cp.asnumpy(self.readout_b),
        }
        np.savez(PATH_FILE, **data)
        print("Saved pathways to", PATH_FILE)


# -------------------------------------------------------
#  Main interactive loop
# -------------------------------------------------------
def main():
    n_neurons = 200_000
    fan_in = 32
    radius = 16

    print(f"Creating brain with {n_neurons} neurons, fan_in={fan_in}, radius={radius}…")
    brain = GPUNeuronBrain(n_neurons=n_neurons, fan_in=fan_in, radius=radius)

    print("\nBrain running. Type text. CTRL-C to stop.\n")

    try:
        while True:
            try:
                user = input("You: ")
            except EOFError:
                break

            user = user.rstrip("\n")
            if not user:
                continue

            reply, spike_counts = brain.interact_once(user)

            # Minimal “is it alive?” feedback: show a few spike counts
            if len(spike_counts) >= 4:
                # show 4 roughly evenly-spaced samples from the settle period
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

