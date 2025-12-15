import os
import time
import warnings
import numpy as np

PATH_FILE = "brain_paths_reasoning.npz"

# Check for transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers library not found.")


# -------------------------------------------------------
#  Multi-layer reasoning brain architecture
# -------------------------------------------------------
class ReasoningNeuronModel:
    """
    Spiking neurons organized into reasoning layers
    """
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.voltage   = np.zeros(n_neurons, dtype=np.float32)
        self.threshold = np.ones(n_neurons, dtype=np.float32) * 1.5
        self.leak      = np.ones(n_neurons, dtype=np.float32) * 0.94
        self.spike     = np.zeros(n_neurons, dtype=np.bool_)

        # Working memory - maintains reasoning state
        self.working_memory = np.zeros(n_neurons, dtype=np.float32)

    def update(self, inputs):
        """Update with working memory integration"""
        # Leaky integration
        self.voltage = self.voltage * self.leak + inputs

        # Add working memory influence (slower, persistent)
        self.voltage += 0.2 * self.working_memory

        # Spike rule
        self.spike = (self.voltage >= self.threshold)

        # Reset voltage on spike
        self.voltage = np.where(self.spike, 0.0, self.voltage)

        # Update working memory (slow accumulation of activity)
        spike_float = self.spike.astype(np.float32)
        self.working_memory = 0.8 * self.working_memory + 0.2 * spike_float

        return self.spike


# -------------------------------------------------------
#  Reasoning layers - multi-level processing
# -------------------------------------------------------
class ReasoningLayers:
    """
    Hierarchical reasoning structure:
    - Input layer: receives sensory input (words)
    - Association layer: connects related concepts
    - Reasoning layer: higher-order processing
    - Decision layer: forms conclusions
    """
    def __init__(self, total_neurons=100_000):
        self.total_neurons = total_neurons

        # Divide neurons into functional layers
        # 20% input, 30% association, 40% reasoning, 10% decision
        self.input_size = int(total_neurons * 0.20)      # 20,000
        self.assoc_size = int(total_neurons * 0.30)      # 30,000
        self.reason_size = int(total_neurons * 0.40)     # 40,000 - BIGGEST for reasoning!
        self.decision_size = int(total_neurons * 0.10)   # 10,000

        # Layer boundaries
        self.input_start = 0
        self.input_end = self.input_size

        self.assoc_start = self.input_end
        self.assoc_end = self.assoc_start + self.assoc_size

        self.reason_start = self.assoc_end
        self.reason_end = self.reason_start + self.reason_size

        self.decision_start = self.reason_end
        self.decision_end = self.decision_start + self.decision_size

        print(f"\nBrain Architecture:")
        print(f"  Input Layer:      {self.input_size:,} neurons  (0-{self.input_end:,})")
        print(f"  Association Layer: {self.assoc_size:,} neurons  ({self.assoc_start:,}-{self.assoc_end:,})")
        print(f"  Reasoning Layer:   {self.reason_size:,} neurons  ({self.reason_start:,}-{self.reason_end:,}) ← THINKING")
        print(f"  Decision Layer:    {self.decision_size:,} neurons  ({self.decision_start:,}-{self.decision_end:,})")

        # Concept mappings for input layer
        self.concepts = self._build_concepts()

    def _build_concepts(self):
        """Map words to input layer regions"""
        concepts = {}
        region_size = 500

        # Common words mapped to input regions
        words = [
            # Greetings
            "hello", "hi", "hey", "bye",
            # Questions
            "what", "why", "how", "when", "where", "who", "which",
            # Positive
            "good", "great", "happy", "yes", "like", "love", "nice",
            # Negative
            "bad", "sad", "no", "wrong", "hate", "not",
            # Self
            "i", "me", "my", "mine",
            # Other
            "you", "your", "yours",
            # Actions
            "do", "go", "make", "want", "need", "think", "know", "understand",
            "see", "talk", "say", "tell", "ask", "learn", "teach", "help",
            "eat", "drink", "sleep", "work", "play", "read", "write",
            # Objects
            "thing", "time", "day", "food", "water", "book", "idea",
            # Meta
            "is", "are", "am", "was", "were", "be", "can", "will", "should",
        ]

        for i, word in enumerate(words):
            if i * region_size >= self.input_size:
                break
            start = i * region_size
            end = min(start + region_size, self.input_size)
            concepts[word] = (start, end)

        return concepts

    def activate_word(self, word):
        """Activate input layer region for a word"""
        activation = np.zeros(self.total_neurons, dtype=np.float32)
        word = word.lower().strip(".,!?;:")

        if word in self.concepts:
            start, end = self.concepts[word]
            activation[start:end] = 1.2
        else:
            # Unknown words go to random input region
            hash_val = hash(word) % self.input_size
            activation[hash_val:min(hash_val+200, self.input_size)] = 0.8

        return activation

    def get_layer_activity(self, spikes):
        """Analyze activity in each layer"""
        return {
            "input": np.sum(spikes[self.input_start:self.input_end]),
            "association": np.sum(spikes[self.assoc_start:self.assoc_end]),
            "reasoning": np.sum(spikes[self.reason_start:self.reason_end]),
            "decision": np.sum(spikes[self.decision_start:self.decision_end]),
        }


# -------------------------------------------------------
#  Reasoning Brain with multi-step thinking
# -------------------------------------------------------
class ReasoningBrain:
    """
    Brain that actually reasons before responding.
    Large reasoning layer processes input in multiple steps.
    """

    def __init__(self, n_neurons=100_000, fan_in=32):
        self.n_neurons = n_neurons
        self.fan_in = fan_in

        print(f"\nInitializing Reasoning Brain with {n_neurons:,} neurons...")

        # Load GPT
        if HAS_TRANSFORMERS:
            print("Loading GPT-2...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            self.gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self.gpt_model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("GPT-2 loaded!")
        else:
            self.tokenizer = None
            self.gpt_model = None

        # Reasoning layer architecture
        self.layers = ReasoningLayers(n_neurons)

        # Neurons
        self.neurons = ReasoningNeuronModel(n_neurons)

        # Connectivity - hierarchical
        print("\nCreating hierarchical connectivity...")

        # Input → Association (forward connections)
        self.input_to_assoc = self._create_forward_connections(
            self.layers.input_size, self.layers.assoc_size, fan_in=fan_in
        )

        # Association → Reasoning (strong forward connections)
        self.assoc_to_reason = self._create_forward_connections(
            self.layers.assoc_size, self.layers.reason_size, fan_in=fan_in*2
        )

        # Reasoning → Decision (convergent connections)
        self.reason_to_decision = self._create_forward_connections(
            self.layers.reason_size, self.layers.decision_size, fan_in=fan_in*3
        )

        # Recurrent within reasoning layer (for multi-step thinking)
        self.reason_recurrent = np.random.randint(
            0, self.layers.reason_size,
            size=(self.layers.reason_size, fan_in),
            dtype=np.int32
        )
        self.reason_weights = (0.12 * np.random.randn(self.layers.reason_size, fan_in)).astype(np.float32)

        # Bias and dynamics
        self.bias = np.zeros(n_neurons, dtype=np.float32)
        self.global_drive = 0.03
        self.noise_sigma = 0.025

        # Homeostasis per layer
        self.firing_avg = np.zeros(n_neurons, dtype=np.float32)
        self.target_rate = 0.05

        # Conversation history
        self.conversation_history = []

        # Reasoning trace (for display)
        self.reasoning_trace = []

        print("Reasoning brain ready!\n")

    def _create_forward_connections(self, src_size, tgt_size, fan_in):
        """Create forward connections between layers"""
        connections = []
        weights = []

        for i in range(tgt_size):
            # Random connections from source layer
            src_indices = np.random.randint(0, src_size, size=fan_in, dtype=np.int32)
            conn_weights = (0.1 * np.random.randn(fan_in)).astype(np.float32)

            connections.append(src_indices)
            weights.append(conn_weights)

        return (np.array(connections), np.array(weights))

    def step(self, external_input=None):
        """One reasoning step"""
        spikes = self.neurons.spike.astype(np.float32)

        total_input = np.zeros(self.n_neurons, dtype=np.float32)

        # External input to input layer
        if external_input is not None:
            total_input += np.asarray(external_input, dtype=np.float32)

        # Input → Association
        input_spikes = spikes[self.layers.input_start:self.layers.input_end]
        conn_idx, conn_weights = self.input_to_assoc
        assoc_input = np.sum(input_spikes[conn_idx] * conn_weights, axis=1)
        total_input[self.layers.assoc_start:self.layers.assoc_end] += assoc_input

        # Association → Reasoning
        assoc_spikes = spikes[self.layers.assoc_start:self.layers.assoc_end]
        conn_idx, conn_weights = self.assoc_to_reason
        reason_input = np.sum(assoc_spikes[conn_idx] * conn_weights, axis=1)
        total_input[self.layers.reason_start:self.layers.reason_end] += reason_input

        # Reasoning recurrent (thinking loop!)
        reason_spikes = spikes[self.layers.reason_start:self.layers.reason_end]
        reason_recurrent_input = np.sum(reason_spikes[self.reason_recurrent] * self.reason_weights, axis=1)
        total_input[self.layers.reason_start:self.layers.reason_end] += reason_recurrent_input * 0.8

        # Reasoning → Decision
        conn_idx, conn_weights = self.reason_to_decision
        decision_input = np.sum(reason_spikes[conn_idx] * conn_weights, axis=1)
        total_input[self.layers.decision_start:self.layers.decision_end] += decision_input

        # Global drive and noise
        total_input += self.global_drive
        total_input += self.noise_sigma * np.random.randn(self.n_neurons).astype(np.float32)

        # Homeostasis
        self.firing_avg = 0.97 * self.firing_avg + 0.03 * spikes
        delta = (self.firing_avg - self.target_rate) * 0.35
        self.neurons.threshold = np.clip(self.neurons.threshold + delta, 0.4, 2.5)

        # Update neurons
        new_spikes = self.neurons.update(total_input)
        return new_spikes

    def encode_text(self, text):
        """Encode text to input layer"""
        words = text.lower().split()
        combined = np.zeros(self.n_neurons, dtype=np.float32)

        for word in words:
            activation = self.layers.activate_word(word)
            combined += activation

        return combined

    def reason_about(self, user_text, reasoning_steps=100):
        """
        Multi-step reasoning process.
        Brain thinks through the input before generating response.
        """
        self.reasoning_trace = []

        # Encode input
        ext_input = self.encode_text(user_text)

        # PHASE 1: Input processing (25 steps)
        print("  → Processing input...")
        for _ in range(25):
            spikes = self.step(external_input=ext_input)

        activity = self.layers.get_layer_activity(spikes)
        self.reasoning_trace.append(("Input processing", activity.copy()))

        # PHASE 2: Association & reasoning (50 steps - main thinking!)
        print("  → Reasoning...")
        for i in range(50):
            spikes = self.step(external_input=ext_input * 0.5)  # Sustained but weaker

            # Sample reasoning activity every 10 steps
            if i % 10 == 0:
                activity = self.layers.get_layer_activity(spikes)
                self.reasoning_trace.append((f"Reasoning step {i//10 + 1}", activity.copy()))

        # PHASE 3: Decision formation (25 steps)
        print("  → Forming decision...")
        for _ in range(25):
            spikes = self.step(external_input=None)  # Free running

        activity = self.layers.get_layer_activity(spikes)
        self.reasoning_trace.append(("Decision", activity.copy()))

        return spikes

    def generate_response(self, user_text, max_length=40):
        """Generate GPT response guided by reasoning state"""
        if not HAS_TRANSFORMERS or self.gpt_model is None:
            return "Error: GPT not loaded."

        # Extract reasoning state (decision layer activity)
        decision_state = self.neurons.working_memory[self.layers.decision_start:self.layers.decision_end]
        reasoning_intensity = float(np.mean(decision_state))

        # Build prompt with conversation history
        prompt = "The following is a thoughtful conversation.\n\n"

        for role, text in self.conversation_history[-4:]:
            prompt += f"{role}: {text}\n"

        prompt += f"User: {user_text}\nAssistant:"

        # Tokenize
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Temperature based on reasoning intensity
        temperature = 0.75 + 0.25 * reasoning_intensity
        temperature = max(0.1, min(temperature, 1.5))

        # Generate
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            outputs = self.gpt_model.generate(
                encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.92,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        # Clean up
        sentences = response.split(".")
        if len(sentences) > 2:
            response = ". ".join(sentences[:2]) + "."

        response = response.replace("\n", " ").strip()

        return response

    def interact(self, user_text):
        """Full interaction with reasoning"""
        if not user_text.strip():
            return "", []

        print(f"\n{'─'*70}")
        print("REASONING PROCESS:")

        # Multi-step reasoning
        final_spikes = self.reason_about(user_text)

        # Generate response
        print("  → Generating response...\n")
        response = self.generate_response(user_text)

        # Update history
        self.conversation_history.append(("User", user_text))
        self.conversation_history.append(("Assistant", response))

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return response, self.reasoning_trace

    def save_paths(self):
        """Save brain state"""
        data = {
            "reason_weights": self.reason_weights,
        }
        np.savez(PATH_FILE, **data)
        print("Saved brain state.")


# -------------------------------------------------------
#  Main
# -------------------------------------------------------
def main():
    if not HAS_TRANSFORMERS:
        print("\nERROR: Requires transformers library.")
        print("Install: pip install transformers torch")
        return

    print(f"\n{'='*70}")
    print("REASONING NEURAL BRAIN")
    print("With dedicated reasoning layers")
    print(f"{'='*70}\n")

    brain = ReasoningBrain(n_neurons=100_000, fan_in=32)

    print(f"\n{'='*70}")
    print("READY - The brain will REASON before responding")
    print("Watch the reasoning process happen in real-time!")
    print(f"{'='*70}\n")

    try:
        while True:
            try:
                user = input("You: ")
            except EOFError:
                break

            user = user.strip()
            if not user:
                continue

            start_time = time.time()
            response, reasoning_trace = brain.interact(user)
            total_time = time.time() - start_time

            # Show reasoning activity
            print(f"{'─'*70}")
            print("REASONING ACTIVITY:")
            for phase, activity in reasoning_trace:
                print(f"  {phase:20s} → Input:{activity['input']:4.0f} | "
                      f"Assoc:{activity['association']:4.0f} | "
                      f"Reason:{activity['reasoning']:5.0f} ← | "
                      f"Decision:{activity['decision']:4.0f}")

            print(f"\n{'─'*70}")
            print(f"Brain: {response}")
            print(f"(Total time: {total_time:.2f}s)")
            print(f"{'─'*70}\n")

    except KeyboardInterrupt:
        print("\n\nStopping...")

    brain.save_paths()


if __name__ == "__main__":
    main()
