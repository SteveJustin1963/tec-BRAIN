import os
import time
import warnings
import numpy as np

PATH_FILE = "brain_paths_gpt.npz"

# Check for transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers library not found.")
    print("Install with: pip install transformers")
    print("Falling back to basic mode...\n")


# -------------------------------------------------------
#  Neuron model with attractor dynamics
# -------------------------------------------------------
class HybridNeuronModel:
    """
    Spiking neuron with attractor dynamics for stable states
    """
    def __init__(self, n_neurons):
        self.n = n_neurons

        self.voltage   = np.zeros(n_neurons, dtype=np.float32)
        self.threshold = np.ones(n_neurons, dtype=np.float32) * 1.5
        self.leak      = np.ones(n_neurons, dtype=np.float32) * 0.95

        self.state     = np.zeros(n_neurons, dtype=np.uint8)
        self.spike     = np.zeros(n_neurons, dtype=np.bool_)

        # Attractor state - maintains recent activity
        self.attractor = np.zeros(n_neurons, dtype=np.float32)

    def update(self, inputs):
        """Update with attractor dynamics"""

        # Leaky integration
        self.voltage = self.voltage * self.leak + inputs

        # Add attractor feedback
        self.voltage += 0.1 * self.attractor

        # Spike rule
        self.spike = (self.voltage >= self.threshold)

        # Reset voltage on spike
        self.voltage = np.where(self.spike, 0.0, self.voltage)

        # Update attractor state
        spike_float = self.spike.astype(np.float32)
        self.attractor = 0.9 * self.attractor + 0.1 * spike_float

        # Simple state machine
        fired = self.spike.astype(np.uint8)
        self.state = np.where(fired == 1, 1, self.state)
        self.threshold = np.where(self.state == 1, 1.5, 1.0)

        return self.spike


# -------------------------------------------------------
#  Brain with GPT-2 integration
# -------------------------------------------------------
class GPTNeuralBrain:
    """
    Neural brain that uses GPT-2 for language understanding.
    The spiking network provides the "thinking" state that influences GPT output.
    """

    def __init__(self, n_neurons=100_000, fan_in=32, radius=16):
        self.n_neurons = n_neurons
        self.fan_in = fan_in
        self.radius = radius

        print(f"Initializing GPT-powered brain with {n_neurons:,} neurons...")

        # Load GPT-2 model
        if HAS_TRANSFORMERS:
            print("Loading DistilGPT2 language model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            self.gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self.gpt_model.eval()  # Inference mode

            # Set pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Language model loaded!")
        else:
            self.tokenizer = None
            self.gpt_model = None

        # Neural dynamics
        self.neurons = HybridNeuronModel(n_neurons)

        print("Creating connectivity matrix...")
        self.inputs_idx = np.random.randint(
            0, n_neurons, size=(n_neurons, fan_in), dtype=np.int32
        )
        self.weights = (0.1 * np.random.randn(n_neurons, fan_in)).astype(np.float32)

        self.bias = np.zeros(n_neurons, dtype=np.float32)
        self.global_drive = 0.05
        self.noise_sigma = 0.02
        self.neigh_strength = 0.02

        # Homeostatic control
        self.firing_avg = np.zeros(n_neurons, dtype=np.float32)
        self.target_rate = 0.05

        # Brain state -> GPT influence mapping
        # The brain's state will modulate GPT's output
        self.state_dim = 128  # Compressed brain state
        print("Creating brain-to-GPT interface...")
        self.brain_to_state = np.random.randn(n_neurons, self.state_dim).astype(np.float32) * 0.01

        # Conversation memory
        self.conversation_history = []
        self.max_history = 5

        # Load saved brain state
        self._load_paths_if_any()

        print("GPT-powered brain ready!")

    # ---------------------------------------------------
    #  Core dynamics
    # ---------------------------------------------------
    def step(self, external_input=None):
        """One time step"""
        presyn = self.neurons.spike.astype(np.float32)

        # Synaptic input
        incoming = presyn[self.inputs_idx]
        syn_current = np.sum(incoming * self.weights, axis=1)

        # Neighborhood interaction
        neigh_sum = np.zeros(self.n_neurons, dtype=np.float32)
        for offset in range(-self.radius, self.radius + 1):
            if offset != 0:
                neigh_sum += np.roll(presyn, offset)
        neigh_drive = self.neigh_strength * neigh_sum

        # Combine
        total = syn_current + neigh_drive + self.bias + self.global_drive

        # Noise
        total += self.noise_sigma * np.random.randn(self.n_neurons).astype(np.float32)

        # External input
        if external_input is not None:
            total += np.asarray(external_input, dtype=np.float32)

        # Homeostasis
        self.firing_avg = 0.98 * self.firing_avg + 0.02 * presyn
        delta = (self.firing_avg - self.target_rate) * 0.3
        self.neurons.threshold = np.clip(self.neurons.threshold + delta, 0.5, 2.5)

        # Update neurons
        spikes = self.neurons.update(total)
        return spikes

    # ---------------------------------------------------
    #  Brain state extraction
    # ---------------------------------------------------
    def get_brain_state(self):
        """Extract compressed state from brain activity"""
        spikes = self.neurons.spike.astype(np.float32)
        attractor = self.neurons.attractor

        # Combine spike pattern and attractor state
        combined = 0.5 * spikes + 0.5 * attractor

        # Project to lower dimension
        state = combined @ self.brain_to_state

        # Normalize
        state = np.tanh(state)

        return state

    # ---------------------------------------------------
    #  Text processing with GPT
    # ---------------------------------------------------
    def encode_text_to_input(self, text):
        """Simple text encoding for neural stimulation"""
        # Hash-based encoding (simple but consistent)
        stim = np.zeros(self.n_neurons, dtype=np.float32)

        words = text.lower().split()
        for word in words:
            # Use hash to map word to neuron region
            word_hash = hash(word) % self.n_neurons
            region_size = 100

            start = word_hash
            end = min(start + region_size, self.n_neurons)
            stim[start:end] += 0.5

        return stim

    def generate_response_with_gpt(self, user_text, brain_state, max_length=50):
        """
        Generate response using GPT-2, influenced by brain state.
        The brain's internal state modulates the response.
        """
        if not HAS_TRANSFORMERS or self.gpt_model is None:
            return "Error: GPT model not loaded. Install transformers library."

        # Build conversation context
        context = ""
        for role, text in self.conversation_history[-self.max_history:]:
            context += f"{role}: {text}\n"
        context += f"User: {user_text}\nAssistant:"

        # Tokenize with attention mask
        encoded = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True)
        inputs = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        # Generate with brain-influenced parameters
        # Brain state influences temperature and top_p
        brain_excitement = float(np.mean(np.abs(brain_state)))

        # Ensure valid temperature (must be > 0, typically 0.1 to 2.0)
        brain_excitement = np.clip(brain_excitement, 0.0, 1.0)
        temperature = float(0.8 + 0.5 * brain_excitement)  # 0.8 to 1.3
        temperature = max(0.1, min(temperature, 2.0))  # Safety bounds

        top_p = 0.92

        # Generate
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            outputs = self.gpt_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        # Clean up
        response = response.split("\n")[0].strip()

        # Limit length
        words = response.split()
        if len(words) > 20:
            response = " ".join(words[:20])

        return response

    # ---------------------------------------------------
    #  Interaction
    # ---------------------------------------------------
    def interact(self, user_text, settle_steps=40):
        """
        Process input and generate intelligent response.
        """
        if not user_text.strip():
            return "", []

        # Encode text to neural input
        ext_input = self.encode_text_to_input(user_text)

        # Let brain process the input
        spike_counts = []
        for _ in range(settle_steps):
            spikes = self.step(external_input=ext_input)
            spike_counts.append(int(np.sum(spikes)))

        # Extract brain state
        brain_state = self.get_brain_state()

        # Generate response using GPT + brain state
        response = self.generate_response_with_gpt(user_text, brain_state)

        # Update conversation history
        self.conversation_history.append(("User", user_text))
        self.conversation_history.append(("Assistant", response))

        # Keep only recent history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history*2:]

        return response, spike_counts, brain_state

    # ---------------------------------------------------
    #  Persistence
    # ---------------------------------------------------
    def _load_paths_if_any(self):
        if os.path.exists(PATH_FILE):
            try:
                data = np.load(PATH_FILE)
                self.brain_to_state = data["brain_to_state"]
                print("Loaded saved brain state from", PATH_FILE)
            except Exception as e:
                print("Could not load brain state:", e)
        else:
            print("No saved brain state found. Starting fresh.")

    def save_paths(self):
        data = {
            "brain_to_state": self.brain_to_state,
        }
        np.savez(PATH_FILE, **data)
        print("Saved brain state to", PATH_FILE)


# -------------------------------------------------------
#  Main
# -------------------------------------------------------
def main():
    if not HAS_TRANSFORMERS:
        print("\nERROR: This version requires the 'transformers' library.")
        print("Install it with:")
        print("  pip install transformers torch")
        print("\nNote: First run will download DistilGPT2 model (~80MB)")
        return

    n_neurons = 100_000
    fan_in = 32
    radius = 16

    print(f"\nCreating GPT-Powered Neural Brain...")
    print(f"Neurons: {n_neurons:,}\n")

    start_time = time.time()
    brain = GPTNeuralBrain(
        n_neurons=n_neurons,
        fan_in=fan_in,
        radius=radius
    )
    init_time = time.time() - start_time

    print(f"\nInitialization took {init_time:.2f} seconds")
    print("\n" + "="*60)
    print("GPT-POWERED NEURAL BRAIN - Ready!")
    print("="*60)
    print("\nThe neural network provides 'thinking' that influences GPT-2.")
    print("This creates more dynamic, brain-modulated responses.")
    print("\nType naturally. CTRL-C to stop.\n")

    try:
        while True:
            try:
                user = input("You: ")
            except EOFError:
                break

            user = user.strip()
            if not user:
                continue

            step_start = time.time()
            response, spike_counts, brain_state = brain.interact(user)
            step_time = time.time() - step_start

            # Show brain activity
            avg_spikes = np.mean(spike_counts)
            brain_excitement = np.mean(np.abs(brain_state))

            print(f"   [spikes: {avg_spikes:.0f}, brain activity: {brain_excitement:.3f}]")
            print(f"Brain: {response}")
            print(f"   (time: {step_time:.2f}s)\n")

    except KeyboardInterrupt:
        print("\n\nStopping and saving...")

    brain.save_paths()
    print("Done!")


if __name__ == "__main__":
    main()
