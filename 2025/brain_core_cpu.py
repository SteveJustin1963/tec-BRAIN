import os
import time
import json
import numpy as np

PATH_FILE = "brain_paths_cpu.npz"


# -------------------------------------------------------
#  Neuron model (CPU version)
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
        self.voltage   = np.zeros(n_neurons, dtype=np.float32)
        self.threshold = np.ones(n_neurons, dtype=np.float32) * 1.5
        self.leak      = np.ones(n_neurons, dtype=np.float32) * 0.97

        # Spiking + state-machine
        self.state               = np.zeros(n_neurons, dtype=np.uint8)
        self.spike               = np.zeros(n_neurons, dtype=np.bool_)
        self.refractory_counter  = np.zeros(n_neurons, dtype=np.uint8)

    def update(self, inputs):
        """
        inputs: CPU vector of synaptic + external drive
        """

        # Leaky integration
        self.voltage = self.voltage * self.leak + inputs

        # Spike rule
        self.spike = (self.voltage >= self.threshold)

        # Reset voltage on spike
        self.voltage = np.where(self.spike, 0.0, self.voltage)

        # Simple state machine:
        fired = self.spike.astype(np.uint8)
        self.state = np.where(fired == 1, 1, self.state)

        # State-dependent thresholds
        # (right now just 2 modes: default vs spiked-before)
        self.threshold = np.where(self.state == 1, 1.5, 1.0)

        return self.spike


# -------------------------------------------------------
#  CPU Brain core + learning readout + text I/O
# -------------------------------------------------------
class CPUNeuronBrain:
    """
    CPU 'brain core' with:
      - local-neighborhood excitation (wave-like behaviour)
      - homeostatic firing-rate control (keeps it "alive")
      - simple trainable readout (spikes -> characters)
      - text I/O and persistence to disk

    Optimized for CPU with reduced neuron count for laptop performance.
    """

    def __init__(self, n_neurons=50_000, fan_in=32, radius=16):
        self.n_neurons = n_neurons
        self.fan_in = fan_in
        self.radius = radius

        print(f"Initializing CPU brain with {n_neurons:,} neurons...")

        # Neuron model (on CPU)
        self.neurons = HybridNeuronModel(n_neurons)

        # Random sparse connectivity
        print("Creating connectivity matrix...")
        self.inputs_idx = np.random.randint(
            0, n_neurons, size=(n_neurons, fan_in), dtype=np.int32
        )
        self.weights = (0.1 * np.random.randn(n_neurons, fan_in)).astype(np.float32)

        # Per-neuron bias (can stimulate regions later)
        self.bias = np.zeros(n_neurons, dtype=np.float32)

        # Global low-level tonic drive
        self.global_drive = 0.05

        # Noise
        self.noise_sigma = 0.03

        # Neighborhood excitation multiplier
        self.neigh_strength = 0.02

        # Homeostatic firing-rate smoothing
        self.firing_avg = np.zeros(n_neurons, dtype=np.float32)
        self.target_rate = 0.05  # aim for ~5% of neurons firing each step

        # --- Text I/O: define vocabulary and readout layer ---
        # Use basic printable ASCII (space..~)
        self.vocab = [chr(c) for c in range(32, 127)]
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        # Readout weights: spikes -> logits over characters
        # shape: (n_neurons, vocab_size)
        print("Initializing readout layer...")
        self.readout_W = np.zeros((self.n_neurons, self.vocab_size),
                                  dtype=np.float32)
        self.readout_b = np.zeros(self.vocab_size, dtype=np.float32)

        # Learning rate for readout
        self.lr = 0.01

        # Try to load saved pathways
        self._load_paths_if_any()

        print("Brain initialization complete!")

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

        presyn = self.neurons.spike.astype(np.float32)

        # 1) Synaptic current from random graph
        incoming = presyn[self.inputs_idx]
        syn_current = np.sum(incoming * self.weights, axis=1)

        # 2) Neighborhood wave drive (circular 1D "ring")
        # CPU optimization: pre-allocate and use vectorized operations
        neigh_sum = np.zeros(self.n_neurons, dtype=np.float32)
        for offset in range(-self.radius, self.radius + 1):
            if offset != 0:
                neigh_sum += np.roll(presyn, offset)
        neigh_drive = self.neigh_strength * neigh_sum

        # 3) Combine drives
        total = syn_current + neigh_drive + self.bias + self.global_drive

        # 4) Add noise
        total += self.noise_sigma * np.random.randn(self.n_neurons).astype(np.float32)

        # 5) Optional external input
        if external_input is not None:
            ext = np.asarray(external_input, dtype=np.float32)
            total += ext

        # 6) Homeostatic control (per-neuron)
        self.firing_avg = 0.98 * self.firing_avg + 0.02 * presyn
        delta = (self.firing_avg - self.target_rate) * 0.5
        self.neurons.threshold = np.clip(self.neurons.threshold + delta,
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
        This is deliberately crude; the CPU core just acts as a big
        noisy "reservoir" the readout will learn to interpret.
        """
        stim = np.zeros(self.n_neurons, dtype=np.float32)
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
        x = spikes.astype(np.float32)  # shape (n_neurons,)
        logits = x @ self.readout_W + self.readout_b  # (vocab_size,)

        j = int(np.argmax(logits))
        return self.idx_to_char.get(j, "?")

    def _train_readout(self, spikes, target_char):
        """
        One-step training of the readout weights using a simple
        softmax cross-entropy gradient.
        """
        idx = self.char_to_idx.get(target_char)
        if idx is None:
            return

        x = spikes.astype(np.float32)  # (n_neurons,)
        logits = x @ self.readout_W + self.readout_b  # (vocab_size,)

        # softmax
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()

        # gradient dL/dlogits = probs - y_onehot
        grad = probs
        grad[idx] -= 1.0

        # Gradient step: W := W - lr * outer(x, grad)
        self.readout_W -= self.lr * np.outer(x, grad)
        self.readout_b -= self.lr * grad

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
            spike_counts.append(int(np.sum(spikes)))

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
                self.readout_W = data["readout_W"]
                self.readout_b = data["readout_b"]
                print("Loaded saved pathways from", PATH_FILE)
            except Exception as e:
                print("Could not load pathways:", e)
        else:
            print("No saved pathways found. Starting fresh.")

    def save_paths(self):
        data = {
            "readout_W": self.readout_W,
            "readout_b": self.readout_b,
        }
        np.savez(PATH_FILE, **data)
        print("Saved pathways to", PATH_FILE)


# -------------------------------------------------------
#  Main interactive loop
# -------------------------------------------------------
def main():
    # Reduced neuron count for CPU efficiency
    # 100k neurons should run well on a laptop with 16GB RAM
    # You can adjust this based on your performance needs
    n_neurons = 100_000  # vs 200k on GPU
    fan_in = 32
    radius = 16

    print(f"Creating CPU brain with {n_neurons:,} neurons, fan_in={fan_in}, radius={radius}...")
    print("This may take a moment to initialize...\n")

    start_time = time.time()
    brain = CPUNeuronBrain(n_neurons=n_neurons, fan_in=fan_in, radius=radius)
    init_time = time.time() - start_time

    print(f"\nInitialization took {init_time:.2f} seconds")
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

            step_start = time.time()
            reply, spike_counts = brain.interact_once(user)
            step_time = time.time() - step_start

            # Minimal "is it alive?" feedback: show a few spike counts
            if len(spike_counts) >= 4:
                # show 4 roughly evenly-spaced samples from the settle period
                indices = np.linspace(0, len(spike_counts) - 1, 4, dtype=int)
                for i in indices:
                    print(f"   [spikes={spike_counts[i]}]")
            else:
                for c in spike_counts:
                    print(f"   [spikes={c}]")

            print(f"Brain: {reply}")
            print(f"   (processing time: {step_time:.3f}s)\n")

    except KeyboardInterrupt:
        print("\nStopping brain and saving pathways.")

    brain.save_paths()


if __name__ == "__main__":
    main()
