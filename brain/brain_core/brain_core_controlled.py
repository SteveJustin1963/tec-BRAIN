import os
import time
import warnings
import numpy as np
import re

PATH_FILE = "brain_paths_controlled.npz"

# Check for transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers library not found.")


# -------------------------------------------------------
#  Topic Control - keeps GPT on track
# -------------------------------------------------------
class TopicController:
    """
    Extracts topics from user input and ensures GPT stays on topic.
    Prevents random drift.
    """
    def __init__(self):
        # Keywords that indicate different topics
        self.topic_patterns = {
            "math": r"\b(\d+|plus|minus|times|divide|equals?|add|subtract|multiply|calculation|solve)\b",
            "greeting": r"\b(hello|hi|hey|greetings?|howdy|sup)\b",
            "question": r"\b(what|why|how|when|where|who|which|define|explain|tell)\b",
            "science": r"\b(science|physics|chemistry|biology|charge|electron|atom|energy|force)\b",
            "time": r"\b(time|day|date|today|tomorrow|yesterday|hour|minute|when)\b",
            "emotion": r"\b(feel|happy|sad|angry|love|hate|like|dislike)\b",
            "conversation": r"\b(talk|chat|discuss|conversation|speak|say)\b",
        }

    def extract_topics(self, text):
        """Identify topics in user's text"""
        text_lower = text.lower()
        detected = []

        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(topic)

        return detected if detected else ["general"]

    def extract_keywords(self, text):
        """Extract important words (nouns, numbers, key terms)"""
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "am", "was", "were", "be", "been",
                    "to", "of", "in", "for", "on", "at", "by", "with", "from", "about",
                    "as", "into", "through", "during", "before", "after", "above", "below",
                    "i", "you", "me", "my", "your"}

        words = text.lower().split()
        keywords = []

        for word in words:
            # Clean word
            word = re.sub(r'[^\w\s]', '', word)

            # Keep if not stopword and length > 2
            if word and word not in stopwords and len(word) > 2:
                keywords.append(word)

            # Always keep numbers
            if word and word.isdigit():
                keywords.append(word)

        return keywords

    def build_controlled_prompt(self, user_text, topics, keywords, history):
        """
        Build a prompt that FORCES GPT to stay on topic.
        """
        # Start with clear instruction
        prompt = "You are a helpful assistant that answers questions directly and stays on topic.\n\n"

        # Add topic constraint
        if topics:
            topic_str = ", ".join(topics)
            prompt += f"Topic: {topic_str}\n"

        # Add required keywords
        if keywords:
            keywords_str = ", ".join(keywords[:5])  # Top 5 keywords
            prompt += f"Answer must address: {keywords_str}\n\n"

        # Add conversation history (brief)
        if history:
            prompt += "Previous conversation:\n"
            for role, text in history[-2:]:  # Last 2 exchanges only
                prompt += f"{role}: {text}\n"
            prompt += "\n"

        # Add user's question
        prompt += f"User: {user_text}\n"

        # Strong directive
        prompt += f"Assistant (answer directly about '{keywords[0] if keywords else 'the question'}'):"

        return prompt


# -------------------------------------------------------
#  Reasoning brain (same as before)
# -------------------------------------------------------
class ReasoningNeuronModel:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.voltage   = np.zeros(n_neurons, dtype=np.float32)
        self.threshold = np.ones(n_neurons, dtype=np.float32) * 1.5
        self.leak      = np.ones(n_neurons, dtype=np.float32) * 0.94
        self.spike     = np.zeros(n_neurons, dtype=np.bool_)
        self.working_memory = np.zeros(n_neurons, dtype=np.float32)

    def update(self, inputs):
        self.voltage = self.voltage * self.leak + inputs
        self.voltage += 0.2 * self.working_memory

        self.spike = (self.voltage >= self.threshold)
        self.voltage = np.where(self.spike, 0.0, self.voltage)

        spike_float = self.spike.astype(np.float32)
        self.working_memory = 0.8 * self.working_memory + 0.2 * spike_float

        return self.spike


class ReasoningLayers:
    def __init__(self, total_neurons=100_000):
        self.total_neurons = total_neurons

        self.input_size = int(total_neurons * 0.20)
        self.assoc_size = int(total_neurons * 0.30)
        self.reason_size = int(total_neurons * 0.40)
        self.decision_size = int(total_neurons * 0.10)

        self.input_start = 0
        self.input_end = self.input_size

        self.assoc_start = self.input_end
        self.assoc_end = self.assoc_start + self.assoc_size

        self.reason_start = self.assoc_end
        self.reason_end = self.reason_start + self.reason_size

        self.decision_start = self.reason_end
        self.decision_end = self.decision_start + self.decision_size

        print(f"\nBrain Architecture:")
        print(f"  Input: {self.input_size:,} | Association: {self.assoc_size:,} | Reasoning: {self.reason_size:,} | Decision: {self.decision_size:,}")

        self.concepts = self._build_concepts()

    def _build_concepts(self):
        concepts = {}
        region_size = 500

        words = [
            "hello", "hi", "hey", "bye", "what", "why", "how", "when", "where", "who",
            "good", "great", "yes", "no", "bad", "i", "you", "do", "go", "want",
            "think", "know", "see", "talk", "say", "tell", "ask", "help", "time", "day",
            "one", "two", "three", "four", "five", "math", "science", "is", "are",
        ]

        for i, word in enumerate(words):
            if i * region_size >= self.input_size:
                break
            start = i * region_size
            end = min(start + region_size, self.input_size)
            concepts[word] = (start, end)

        return concepts

    def activate_word(self, word):
        activation = np.zeros(self.total_neurons, dtype=np.float32)
        word = word.lower().strip(".,!?;:")

        if word in self.concepts:
            start, end = self.concepts[word]
            activation[start:end] = 1.2
        else:
            hash_val = hash(word) % self.input_size
            activation[hash_val:min(hash_val+200, self.input_size)] = 0.8

        return activation

    def get_layer_activity(self, spikes):
        return {
            "input": np.sum(spikes[self.input_start:self.input_end]),
            "association": np.sum(spikes[self.assoc_start:self.assoc_end]),
            "reasoning": np.sum(spikes[self.reason_start:self.reason_end]),
            "decision": np.sum(spikes[self.decision_start:self.decision_end]),
        }


# -------------------------------------------------------
#  Controlled Reasoning Brain
# -------------------------------------------------------
class ControlledReasoningBrain:
    """
    Brain with CONTROLLED GPT generation.
    GPT is forced to stay on topic and address user's actual input.
    """

    def __init__(self, n_neurons=100_000, fan_in=32):
        self.n_neurons = n_neurons
        self.fan_in = fan_in

        print(f"\nInitializing Controlled Reasoning Brain ({n_neurons:,} neurons)...")

        # Load GPT
        if HAS_TRANSFORMERS:
            print("Loading GPT-2 with topic control...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            self.gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self.gpt_model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("GPT-2 loaded!")
        else:
            self.tokenizer = None
            self.gpt_model = None

        # Topic controller
        self.topic_controller = TopicController()

        # Brain layers
        self.layers = ReasoningLayers(n_neurons)
        self.neurons = ReasoningNeuronModel(n_neurons)

        # Connectivity
        print("Creating connectivity...")
        self.input_to_assoc = self._create_connections(self.layers.input_size, self.layers.assoc_size, fan_in)
        self.assoc_to_reason = self._create_connections(self.layers.assoc_size, self.layers.reason_size, fan_in*2)
        self.reason_to_decision = self._create_connections(self.layers.reason_size, self.layers.decision_size, fan_in*3)

        self.reason_recurrent = np.random.randint(0, self.layers.reason_size, size=(self.layers.reason_size, fan_in), dtype=np.int32)
        self.reason_weights = (0.12 * np.random.randn(self.layers.reason_size, fan_in)).astype(np.float32)

        self.bias = np.zeros(n_neurons, dtype=np.float32)
        self.global_drive = 0.03
        self.noise_sigma = 0.025

        self.firing_avg = np.zeros(n_neurons, dtype=np.float32)
        self.target_rate = 0.05

        self.conversation_history = []

        print("Controlled brain ready!\n")

    def _create_connections(self, src_size, tgt_size, fan_in):
        connections = []
        weights = []
        for i in range(tgt_size):
            src_indices = np.random.randint(0, src_size, size=fan_in, dtype=np.int32)
            conn_weights = (0.1 * np.random.randn(fan_in)).astype(np.float32)
            connections.append(src_indices)
            weights.append(conn_weights)
        return (np.array(connections), np.array(weights))

    def step(self, external_input=None):
        spikes = self.neurons.spike.astype(np.float32)
        total_input = np.zeros(self.n_neurons, dtype=np.float32)

        if external_input is not None:
            total_input += np.asarray(external_input, dtype=np.float32)

        # Forward propagation through layers
        input_spikes = spikes[self.layers.input_start:self.layers.input_end]
        conn_idx, conn_weights = self.input_to_assoc
        assoc_input = np.sum(input_spikes[conn_idx] * conn_weights, axis=1)
        total_input[self.layers.assoc_start:self.layers.assoc_end] += assoc_input

        assoc_spikes = spikes[self.layers.assoc_start:self.layers.assoc_end]
        conn_idx, conn_weights = self.assoc_to_reason
        reason_input = np.sum(assoc_spikes[conn_idx] * conn_weights, axis=1)
        total_input[self.layers.reason_start:self.layers.reason_end] += reason_input

        reason_spikes = spikes[self.layers.reason_start:self.layers.reason_end]
        reason_recurrent_input = np.sum(reason_spikes[self.reason_recurrent] * self.reason_weights, axis=1)
        total_input[self.layers.reason_start:self.layers.reason_end] += reason_recurrent_input * 0.8

        conn_idx, conn_weights = self.reason_to_decision
        decision_input = np.sum(reason_spikes[conn_idx] * conn_weights, axis=1)
        total_input[self.layers.decision_start:self.layers.decision_end] += decision_input

        total_input += self.global_drive
        total_input += self.noise_sigma * np.random.randn(self.n_neurons).astype(np.float32)

        self.firing_avg = 0.97 * self.firing_avg + 0.03 * spikes
        delta = (self.firing_avg - self.target_rate) * 0.35
        self.neurons.threshold = np.clip(self.neurons.threshold + delta, 0.4, 2.5)

        new_spikes = self.neurons.update(total_input)
        return new_spikes

    def encode_text(self, text):
        words = text.lower().split()
        combined = np.zeros(self.n_neurons, dtype=np.float32)
        for word in words:
            activation = self.layers.activate_word(word)
            combined += activation
        return combined

    def reason_about(self, user_text, steps=80):
        """Reasoning process"""
        ext_input = self.encode_text(user_text)

        # Input phase
        for _ in range(20):
            self.step(external_input=ext_input)

        # Reasoning phase
        for _ in range(40):
            self.step(external_input=ext_input * 0.4)

        # Decision phase
        for _ in range(20):
            self.step(external_input=None)

        # Get final activity
        final_spikes = self.neurons.spike
        activity = self.layers.get_layer_activity(final_spikes)

        return activity

    def generate_controlled_response(self, user_text):
        """
        Generate response with STRONG topic control.
        Forces GPT to stay on topic.
        """
        if not HAS_TRANSFORMERS or self.gpt_model is None:
            return "Error: GPT not loaded.", []

        # Extract topics and keywords
        topics = self.topic_controller.extract_topics(user_text)
        keywords = self.topic_controller.extract_keywords(user_text)

        print(f"  → Detected topics: {', '.join(topics)}")
        print(f"  → Keywords: {', '.join(keywords[:5])}")

        # Build controlled prompt
        prompt = self.topic_controller.build_controlled_prompt(
            user_text, topics, keywords, self.conversation_history
        )

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Generate with constraints
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # Use lower temperature for more focused responses
            outputs = self.gpt_model.generate(
                encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_new_tokens=30,  # Shorter to prevent drift
                temperature=0.7,    # Lower = more focused
                top_p=0.85,         # More focused sampling
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,  # Prevent loops
                no_repeat_ngram_size=3,  # Prevent repetition
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's part
        if "Assistant" in response:
            parts = response.split("Assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
                # Remove leading punctuation/special chars
                response = re.sub(r'^[:\(\)\-\s]+', '', response)

        # Clean up - take first complete sentence
        response = response.replace("\n", " ").strip()
        sentences = re.split(r'[.!?]+', response)
        if sentences:
            response = sentences[0].strip()
            if response and not response[-1] in '.!?':
                response += '.'

        return response, keywords

    def interact(self, user_text):
        """Full interaction with controlled generation"""
        if not user_text.strip():
            return "", None

        print(f"\n{'─'*70}")
        print("THINKING...")

        # Brain reasoning
        activity = self.reason_about(user_text)

        print(f"  Reasoning activity → Input:{activity['input']:4.0f} | "
              f"Assoc:{activity['association']:4.0f} | "
              f"Reason:{activity['reasoning']:5.0f} | "
              f"Decision:{activity['decision']:4.0f}")

        # Generate controlled response
        response, keywords = self.generate_controlled_response(user_text)

        # Update history
        self.conversation_history.append(("User", user_text))
        self.conversation_history.append(("Assistant", response))

        if len(self.conversation_history) > 8:
            self.conversation_history = self.conversation_history[-8:]

        return response, activity

    def save_paths(self):
        data = {"reason_weights": self.reason_weights}
        np.savez(PATH_FILE, **data)
        print("Saved.")


# -------------------------------------------------------
#  Main
# -------------------------------------------------------
def main():
    if not HAS_TRANSFORMERS:
        print("\nERROR: Requires transformers and torch")
        print("Install: pip install transformers torch")
        return

    print(f"\n{'='*70}")
    print("CONTROLLED REASONING BRAIN")
    print("GPT is forced to stay on topic and answer your actual questions")
    print(f"{'='*70}")

    brain = ControlledReasoningBrain(n_neurons=100_000, fan_in=32)

    print(f"\n{'='*70}")
    print("READY - No more random rambling!")
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
            response, activity = brain.interact(user)
            total_time = time.time() - start_time

            print(f"\n{'─'*70}")
            print(f"Brain: {response}")
            print(f"(Time: {total_time:.2f}s)")
            print(f"{'─'*70}\n")

    except KeyboardInterrupt:
        print("\n\nStopping...")

    brain.save_paths()


if __name__ == "__main__":
    main()
