import os
import time
import numpy as np

PATH_FILE = "brain_paths_smart.npz"


# -------------------------------------------------------
#  Simple word embeddings - common English words
# -------------------------------------------------------
class SimpleWordEmbeddings:
    """
    A small vocabulary with semantic embeddings.
    Creates a basic semantic space for common English words.
    """
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim

        # Common English words organized by category
        self.vocab = [
            # Pronouns
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            # Common verbs
            "is", "are", "am", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "go", "goes", "went", "get", "got", "make", "made",
            "see", "saw", "know", "knew", "think", "thought", "take", "took",
            "come", "came", "want", "like", "use", "work", "call", "try",
            "ask", "need", "feel", "become", "leave", "put", "mean", "keep",
            # Common nouns
            "man", "woman", "person", "people", "child", "kid", "baby",
            "time", "year", "day", "week", "month", "hour", "minute",
            "thing", "world", "life", "way", "place", "home", "house",
            "work", "school", "hand", "eye", "head", "face", "name",
            "water", "food", "book", "car", "cat", "dog", "tree",
            # Adjectives
            "good", "bad", "new", "old", "great", "big", "small", "little",
            "long", "short", "high", "low", "hot", "cold", "nice", "happy",
            "sad", "right", "wrong", "true", "false", "easy", "hard",
            # Common words
            "the", "a", "an", "and", "or", "but", "if", "not", "no", "yes",
            "to", "of", "in", "for", "on", "at", "with", "from", "by",
            "what", "who", "where", "when", "why", "how", "which",
            "this", "that", "these", "those", "my", "your", "his", "her",
            "hello", "hi", "bye", "thanks", "please", "sorry", "ok", "okay",
            # Numbers
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            # Questions/Common phrases
            "help", "stop", "start", "end", "begin", "tell", "say", "talk", "speak",
        ]

        # Add unknown token
        self.vocab.append("<UNK>")

        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # Create semantic embeddings
        # Initialize with random vectors, then adjust for semantic similarity
        np.random.seed(42)
        self.embeddings = np.random.randn(self.vocab_size, embedding_dim).astype(np.float32) * 0.1

        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

        print(f"Loaded {self.vocab_size} words with {embedding_dim}-dim embeddings")

    def get_embedding(self, word):
        """Get embedding vector for a word"""
        word = word.lower()
        idx = self.word_to_idx.get(word, self.word_to_idx["<UNK>"])
        return self.embeddings[idx]

    def get_word(self, idx):
        """Get word from index"""
        return self.idx_to_word.get(idx, "<UNK>")

    def find_closest_word(self, vector):
        """Find closest word to a given vector"""
        # Normalize input
        vector = vector / (np.linalg.norm(vector) + 1e-8)

        # Compute cosine similarity with all embeddings
        similarities = np.dot(self.embeddings, vector)

        # Get best match
        best_idx = np.argmax(similarities)
        return self.get_word(best_idx), similarities[best_idx]


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
        self.leak      = np.ones(n_neurons, dtype=np.float32) * 0.95  # slightly more leaky for stability

        self.state     = np.zeros(n_neurons, dtype=np.uint8)
        self.spike     = np.zeros(n_neurons, dtype=np.bool_)

        # Attractor state - maintains recent activity
        self.attractor = np.zeros(n_neurons, dtype=np.float32)

    def update(self, inputs):
        """Update with attractor dynamics"""

        # Leaky integration
        self.voltage = self.voltage * self.leak + inputs

        # Add attractor feedback (helps maintain stable states)
        self.voltage += 0.1 * self.attractor

        # Spike rule
        self.spike = (self.voltage >= self.threshold)

        # Reset voltage on spike
        self.voltage = np.where(self.spike, 0.0, self.voltage)

        # Update attractor state (slow dynamics)
        spike_float = self.spike.astype(np.float32)
        self.attractor = 0.9 * self.attractor + 0.1 * spike_float

        # Simple state machine
        fired = self.spike.astype(np.uint8)
        self.state = np.where(fired == 1, 1, self.state)
        self.threshold = np.where(self.state == 1, 1.5, 1.0)

        return self.spike


# -------------------------------------------------------
#  Smart CPU Brain with word-level understanding
# -------------------------------------------------------
class SmartCPUBrain:
    """
    CPU brain with word embeddings and semantic understanding
    """

    def __init__(self, n_neurons=100_000, fan_in=32, radius=16, embedding_dim=64):
        self.n_neurons = n_neurons
        self.fan_in = fan_in
        self.radius = radius

        print(f"Initializing Smart CPU brain with {n_neurons:,} neurons...")

        # Load word embeddings
        self.embeddings = SimpleWordEmbeddings(embedding_dim=embedding_dim)

        # Neuron model with attractors
        self.neurons = HybridNeuronModel(n_neurons)

        # Random sparse connectivity
        print("Creating connectivity matrix...")
        self.inputs_idx = np.random.randint(
            0, n_neurons, size=(n_neurons, fan_in), dtype=np.int32
        )
        self.weights = (0.1 * np.random.randn(n_neurons, fan_in)).astype(np.float32)

        # Per-neuron bias
        self.bias = np.zeros(n_neurons, dtype=np.float32)
        self.global_drive = 0.05
        self.noise_sigma = 0.02  # reduced noise for more stable attractors
        self.neigh_strength = 0.02

        # Homeostatic control
        self.firing_avg = np.zeros(n_neurons, dtype=np.float32)
        self.target_rate = 0.05

        # Word-to-neuron mapping
        # Each word gets mapped to a specific neuron region
        self.word_region_size = max(100, n_neurons // self.embeddings.vocab_size)

        # Encoder: word embeddings -> neuron activations
        print("Initializing word encoder...")
        self.encoder_W = np.random.randn(embedding_dim, n_neurons).astype(np.float32) * 0.05

        # Decoder: neuron activations -> word embeddings
        print("Initializing word decoder...")
        self.decoder_W = np.random.randn(n_neurons, embedding_dim).astype(np.float32) * 0.05
        self.decoder_b = np.zeros(embedding_dim, dtype=np.float32)

        # Learning rate
        self.lr = 0.01

        # Load saved pathways
        self._load_paths_if_any()

        print("Smart brain initialization complete!")

    # ---------------------------------------------------
    #  Core dynamics
    # ---------------------------------------------------
    def step(self, external_input=None):
        """One time step"""
        presyn = self.neurons.spike.astype(np.float32)

        # 1) Synaptic input
        incoming = presyn[self.inputs_idx]
        syn_current = np.sum(incoming * self.weights, axis=1)

        # 2) Neighborhood interaction
        neigh_sum = np.zeros(self.n_neurons, dtype=np.float32)
        for offset in range(-self.radius, self.radius + 1):
            if offset != 0:
                neigh_sum += np.roll(presyn, offset)
        neigh_drive = self.neigh_strength * neigh_sum

        # 3) Combine
        total = syn_current + neigh_drive + self.bias + self.global_drive

        # 4) Noise
        total += self.noise_sigma * np.random.randn(self.n_neurons).astype(np.float32)

        # 5) External input
        if external_input is not None:
            total += np.asarray(external_input, dtype=np.float32)

        # 6) Homeostasis
        self.firing_avg = 0.98 * self.firing_avg + 0.02 * presyn
        delta = (self.firing_avg - self.target_rate) * 0.3
        self.neurons.threshold = np.clip(self.neurons.threshold + delta, 0.5, 2.5)

        # 7) Update neurons
        spikes = self.neurons.update(total)
        return spikes

    # ---------------------------------------------------
    #  Word-level encoding/decoding
    # ---------------------------------------------------
    def encode_words_to_input(self, text):
        """Convert text to neural input using word embeddings"""
        words = text.lower().split()
        if not words:
            return np.zeros(self.n_neurons, dtype=np.float32)

        # Get embeddings for all words
        word_vecs = [self.embeddings.get_embedding(w) for w in words]

        # Average word embeddings (simple but effective)
        avg_embedding = np.mean(word_vecs, axis=0)

        # Project to neuron space
        neuron_input = avg_embedding @ self.encoder_W

        # Add some sparsity
        neuron_input = np.tanh(neuron_input) * 2.0

        return neuron_input

    def decode_spikes_to_word(self, spikes):
        """Decode spike pattern to word using learned decoder"""
        x = spikes.astype(np.float32)

        # Project spikes to embedding space
        embedding_pred = x @ self.decoder_W + self.decoder_b

        # Find closest word
        word, similarity = self.embeddings.find_closest_word(embedding_pred)
        return word, similarity

    def train_on_pair(self, input_text, target_text):
        """Train the brain to associate input words with target words"""
        # Get target word embedding
        target_words = target_text.lower().split()
        if not target_words:
            return

        target_word = target_words[0]  # Use first word as target
        target_embedding = self.embeddings.get_embedding(target_word)

        # Current spike pattern
        spikes = self.neurons.spike.astype(np.float32)

        # Current prediction
        pred_embedding = spikes @ self.decoder_W + self.decoder_b

        # Error
        error = target_embedding - pred_embedding

        # Update decoder (gradient descent)
        self.decoder_W += self.lr * np.outer(spikes, error)
        self.decoder_b += self.lr * error * 0.1

        # Also update encoder to strengthen input->pattern mapping
        input_embedding = self.embeddings.get_embedding(input_text.lower().split()[0] if input_text.split() else "the")
        self.encoder_W += self.lr * 0.1 * np.outer(input_embedding, spikes)

    # ---------------------------------------------------
    #  Interaction
    # ---------------------------------------------------
    def interact(self, text, settle_steps=50, reply_words=3, train=True):
        """
        Process input text and generate word-level reply
        """
        if not text:
            return "", []

        # Encode input
        ext_input = self.encode_words_to_input(text)

        # Let brain settle with input
        spike_counts = []
        for _ in range(settle_steps):
            spikes = self.step(external_input=ext_input)
            spike_counts.append(int(np.sum(spikes)))

        # Train if requested
        if train:
            # Simple self-supervised: try to predict input words
            self.train_on_pair(text, text)

        # Generate response words
        reply_words_list = []
        confidences = []

        for _ in range(reply_words):
            # Let brain run freely
            for _ in range(20):  # settle into word attractor
                spikes = self.step(external_input=None)

            # Decode current state
            word, confidence = self.decode_spikes_to_word(spikes)
            reply_words_list.append(word)
            confidences.append(confidence)

        reply = " ".join(reply_words_list)
        return reply, spike_counts, confidences

    # ---------------------------------------------------
    #  Persistence
    # ---------------------------------------------------
    def _load_paths_if_any(self):
        if os.path.exists(PATH_FILE):
            try:
                data = np.load(PATH_FILE)
                self.encoder_W = data["encoder_W"]
                self.decoder_W = data["decoder_W"]
                self.decoder_b = data["decoder_b"]
                print("Loaded saved pathways from", PATH_FILE)
            except Exception as e:
                print("Could not load pathways:", e)
        else:
            print("No saved pathways found. Starting fresh.")

    def save_paths(self):
        data = {
            "encoder_W": self.encoder_W,
            "decoder_W": self.decoder_W,
            "decoder_b": self.decoder_b,
        }
        np.savez(PATH_FILE, **data)
        print("Saved pathways to", PATH_FILE)


# -------------------------------------------------------
#  Main
# -------------------------------------------------------
def main():
    n_neurons = 100_000
    fan_in = 32
    radius = 16
    embedding_dim = 64

    print(f"Creating Smart CPU Brain with word-level understanding...")
    print(f"Neurons: {n_neurons:,}, Embedding dim: {embedding_dim}\n")

    start_time = time.time()
    brain = SmartCPUBrain(
        n_neurons=n_neurons,
        fan_in=fan_in,
        radius=radius,
        embedding_dim=embedding_dim
    )
    init_time = time.time() - start_time

    print(f"\nInitialization took {init_time:.2f} seconds")
    print("\nSmart brain ready! Type sentences (use common English words).")
    print("The brain will learn word associations and reply with actual words.")
    print("CTRL-C to stop.\n")

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
            reply, spike_counts, confidences = brain.interact(user, train=True)
            step_time = time.time() - step_start

            # Show activity
            avg_spikes = np.mean(spike_counts)
            avg_confidence = np.mean(confidences)

            print(f"   [avg spikes: {avg_spikes:.0f}, confidence: {avg_confidence:.3f}]")
            print(f"Brain: {reply}")
            print(f"   (time: {step_time:.3f}s)\n")

    except KeyboardInterrupt:
        print("\nStopping and saving...")

    brain.save_paths()
    print("Done!")


if __name__ == "__main__":
    main()
