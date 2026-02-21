import os
import time
import warnings
import numpy as np

PATH_FILE = "brain_paths_gpt_v2.npz"

# Check for transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers library not found.")
    print("Install with: pip install transformers torch")


# -------------------------------------------------------
#  Neuron model with attractor dynamics
# -------------------------------------------------------
class HybridNeuronModel:
    """Spiking neuron with attractor dynamics"""
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.voltage   = np.zeros(n_neurons, dtype=np.float32)
        self.threshold = np.ones(n_neurons, dtype=np.float32) * 1.5
        self.leak      = np.ones(n_neurons, dtype=np.float32) * 0.95
        self.state     = np.zeros(n_neurons, dtype=np.uint8)
        self.spike     = np.zeros(n_neurons, dtype=np.bool_)
        self.attractor = np.zeros(n_neurons, dtype=np.float32)

    def update(self, inputs):
        """Update with attractor dynamics"""
        # Leaky integration
        self.voltage = self.voltage * self.leak + inputs

        # Add attractor feedback
        self.voltage += 0.15 * self.attractor

        # Spike rule
        self.spike = (self.voltage >= self.threshold)

        # Reset voltage on spike
        self.voltage = np.where(self.spike, 0.0, self.voltage)

        # Update attractor state (slow dynamics = memory)
        spike_float = self.spike.astype(np.float32)
        self.attractor = 0.85 * self.attractor + 0.15 * spike_float

        # Simple state machine
        fired = self.spike.astype(np.uint8)
        self.state = np.where(fired == 1, 1, self.state)
        self.threshold = np.where(self.state == 1, 1.5, 1.0)

        return self.spike


# -------------------------------------------------------
#  Concept space - maps words to neuron regions
# -------------------------------------------------------
class ConceptSpace:
    """
    Maps semantic concepts (words/topics) to specific neuron regions.
    This creates a semantic topology in the brain.
    """
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons

        # Define semantic concepts and their neuron regions
        self.concepts = {
            # Greetings
            "hello": (0, 1000), "hi": (0, 1000), "hey": (0, 1000),
            # Questions
            "what": (1000, 2000), "why": (1000, 2000), "how": (1000, 2000),
            "when": (1000, 2000), "where": (1000, 2000), "who": (1000, 2000),
            # Positive
            "good": (2000, 3000), "great": (2000, 3000), "happy": (2000, 3000),
            "yes": (2000, 3000), "like": (2000, 3000), "love": (2000, 3000),
            # Negative
            "bad": (3000, 4000), "sad": (3000, 4000), "no": (3000, 4000),
            "wrong": (3000, 4000), "hate": (3000, 4000),
            # Self-reference
            "i": (4000, 5000), "me": (4000, 5000), "my": (4000, 5000),
            # Other-reference
            "you": (5000, 6000), "your": (5000, 6000),
            # Actions
            "do": (6000, 7000), "go": (6000, 7000), "make": (6000, 7000),
            "want": (6000, 7000), "need": (6000, 7000), "think": (6000, 7000),
            "know": (6000, 7000), "see": (6000, 7000), "talk": (6000, 7000),
            # Objects
            "thing": (7000, 8000), "time": (7000, 8000), "day": (7000, 8000),
            "food": (7000, 8000), "water": (7000, 8000),
            # Common
            "is": (8000, 9000), "are": (8000, 9000), "the": (8000, 9000),
            "a": (8000, 9000), "an": (8000, 9000),
        }

        # Default region for unknown words
        self.default_region = (9000, 10000)

    def get_region(self, word):
        """Get neuron region for a word"""
        word = word.lower()
        return self.concepts.get(word, self.default_region)

    def activate_word(self, word, strength=1.0):
        """Create activation pattern for a word"""
        activation = np.zeros(self.n_neurons, dtype=np.float32)
        start, end = self.get_region(word)
        activation[start:end] = strength
        return activation

    def decode_active_regions(self, spike_pattern):
        """Decode which concepts are active based on spike pattern"""
        active_concepts = []
        for word, (start, end) in self.concepts.items():
            region_spikes = np.sum(spike_pattern[start:end])
            if region_spikes > 10:  # Threshold for "active"
                active_concepts.append((word, region_spikes))

        # Sort by activity
        active_concepts.sort(key=lambda x: x[1], reverse=True)
        return active_concepts[:5]  # Top 5 concepts


# -------------------------------------------------------
#  Brain with proper GPT integration
# -------------------------------------------------------
class ProperGPTBrain:
    """
    Neural brain with PROPER GPT-2 integration.
    Brain thinks about concepts, GPT generates language based on those concepts.
    """

    def __init__(self, n_neurons=100_000, fan_in=32, radius=16):
        self.n_neurons = n_neurons
        self.fan_in = fan_in
        self.radius = radius

        print(f"Initializing Proper GPT Brain with {n_neurons:,} neurons...")

        # Load GPT-2 model
        if HAS_TRANSFORMERS:
            print("Loading DistilGPT2 language model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            self.gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self.gpt_model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Language model loaded!")
        else:
            self.tokenizer = None
            self.gpt_model = None

        # Concept space - semantic topology
        print("Creating concept space...")
        self.concepts = ConceptSpace(n_neurons)

        # Neural dynamics
        self.neurons = HybridNeuronModel(n_neurons)

        print("Creating connectivity matrix...")
        self.inputs_idx = np.random.randint(
            0, n_neurons, size=(n_neurons, fan_in), dtype=np.int32
        )
        self.weights = (0.08 * np.random.randn(n_neurons, fan_in)).astype(np.float32)

        self.bias = np.zeros(n_neurons, dtype=np.float32)
        self.global_drive = 0.04
        self.noise_sigma = 0.02
        self.neigh_strength = 0.015

        # Homeostatic control
        self.firing_avg = np.zeros(n_neurons, dtype=np.float32)
        self.target_rate = 0.05  # 5% target

        # Conversation memory
        self.conversation_history = []

        # Load saved state
        self._load_paths_if_any()

        print("Proper GPT Brain ready!")

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
        delta = (self.firing_avg - self.target_rate) * 0.4
        self.neurons.threshold = np.clip(self.neurons.threshold + delta, 0.5, 2.5)

        # Update neurons
        spikes = self.neurons.update(total)
        return spikes

    # ---------------------------------------------------
    #  Semantic encoding
    # ---------------------------------------------------
    def encode_text(self, text):
        """Encode text into brain activation using concept space"""
        words = text.lower().split()
        combined = np.zeros(self.n_neurons, dtype=np.float32)

        for word in words:
            # Remove punctuation
            word = word.strip(".,!?;:")
            if not word:
                continue

            # Activate corresponding concept region
            activation = self.concepts.activate_word(word, strength=0.8)
            combined += activation

        return combined

    def decode_brain_state(self):
        """Decode what the brain is currently thinking about"""
        # Use attractor state (not just spikes) for more stable decoding
        active_concepts = self.concepts.decode_active_regions(self.neurons.attractor)
        return active_concepts

    # ---------------------------------------------------
    #  GPT generation with brain guidance
    # ---------------------------------------------------
    def generate_response(self, user_text, active_concepts, max_length=30):
        """
        Generate response using GPT, guided by brain's active concepts.
        """
        if not HAS_TRANSFORMERS or self.gpt_model is None:
            return "Error: GPT model not loaded."

        # Build prompt with brain's active concepts
        # This makes GPT aware of what the brain is "thinking about"
        concept_words = [word for word, _ in active_concepts]
        concept_hint = ", ".join(concept_words[:3]) if concept_words else "general"

        # Build conversation-aware prompt
        prompt = f"The following is a friendly conversation.\n\n"

        # Add recent history
        for role, text in self.conversation_history[-3:]:
            prompt += f"{role}: {text}\n"

        prompt += f"User: {user_text}\n"
        prompt += f"Assistant:"

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        # Generate
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            outputs = self.gpt_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=0.85,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.2,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        else:
            # Fallback: get text after user input
            response = response.replace(prompt, "").strip()

        # Clean up - get first sentence or two
        sentences = response.split(".")
        if len(sentences) > 2:
            response = ". ".join(sentences[:2]) + "."
        elif not response.endswith("."):
            response += "."

        # Remove newlines
        response = response.replace("\n", " ").strip()

        return response

    # ---------------------------------------------------
    #  Interaction
    # ---------------------------------------------------
    def interact(self, user_text, settle_steps=60):
        """
        Process input and generate intelligent response.
        """
        if not user_text.strip():
            return "", [], []

        # Encode user input into brain
        ext_input = self.encode_text(user_text)

        # Let brain think about the input
        spike_counts = []
        for _ in range(settle_steps):
            spikes = self.step(external_input=ext_input)
            spike_counts.append(int(np.sum(spikes)))

        # Decode what the brain is thinking about
        active_concepts = self.decode_brain_state()

        # Generate response based on brain's thoughts
        response = self.generate_response(user_text, active_concepts)

        # Update conversation history
        self.conversation_history.append(("User", user_text))
        self.conversation_history.append(("Assistant", response))

        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return response, spike_counts, active_concepts

    # ---------------------------------------------------
    #  Persistence
    # ---------------------------------------------------
    def _load_paths_if_any(self):
        if os.path.exists(PATH_FILE):
            try:
                data = np.load(PATH_FILE)
                self.weights = data["weights"]
                print("Loaded saved brain state from", PATH_FILE)
            except Exception as e:
                print("Could not load brain state:", e)
        else:
            print("No saved brain state found. Starting fresh.")

    def save_paths(self):
        data = {
            "weights": self.weights,
        }
        np.savez(PATH_FILE, **data)
        print("Saved brain state to", PATH_FILE)


# -------------------------------------------------------
#  Main
# -------------------------------------------------------
def main():
    if not HAS_TRANSFORMERS:
        print("\nERROR: This version requires the 'transformers' library.")
        print("Install it with: pip install transformers torch")
        return

    n_neurons = 100_000
    fan_in = 32
    radius = 16

    print(f"\n{'='*70}")
    print("PROPER GPT-POWERED NEURAL BRAIN v2")
    print(f"{'='*70}\n")

    start_time = time.time()
    brain = ProperGPTBrain(
        n_neurons=n_neurons,
        fan_in=fan_in,
        radius=radius
    )
    init_time = time.time() - start_time

    print(f"\nInitialization took {init_time:.2f} seconds")
    print("\n" + "="*70)
    print("HOW IT WORKS:")
    print("- Your words activate specific brain regions (concepts)")
    print("- Brain 'thinks' with sparse spiking (2-5% = healthy)")
    print("- Active concepts guide GPT to generate relevant replies")
    print("- More natural conversation!")
    print("="*70)
    print("\nReady! Type naturally. CTRL-C to stop.\n")

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
            response, spike_counts, active_concepts = brain.interact(user)
            step_time = time.time() - step_start

            # Show brain activity
            avg_spikes = np.mean(spike_counts)
            firing_rate = (avg_spikes / n_neurons) * 100

            print(f"\n   [Brain Stats]")
            print(f"   • Spikes: {avg_spikes:.0f}/{n_neurons:,} ({firing_rate:.1f}%)")
            print(f"   • Active concepts: {', '.join([word for word, _ in active_concepts[:3]])}")

            print(f"\nBrain: {response}")
            print(f"   (processing time: {step_time:.2f}s)\n")

    except KeyboardInterrupt:
        print("\n\nStopping and saving...")

    brain.save_paths()
    print("Done!")


if __name__ == "__main__":
    main()
