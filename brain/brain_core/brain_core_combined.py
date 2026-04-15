import os
import time
import json
import numpy as np
import re
import warnings

# Check for transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

PATH_FILE = "brain_paths_combined.npz"
CONCEPT_WORDS_FILE = "concept_words.json"

# -------------------------------------------------------
#  Neuron model (from brain_core_cpu.py)
# -------------------------------------------------------
class HybridNeuronModel:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.voltage   = np.zeros(n_neurons, dtype=np.float32)
        self.threshold = np.ones(n_neurons, dtype=np.float32) * 1.5
        self.leak      = np.ones(n_neurons, dtype=np.float32) * 0.97
        self.state               = np.zeros(n_neurons, dtype=np.uint8)
        self.spike               = np.zeros(n_neurons, dtype=np.bool_)
        self.refractory_counter  = np.zeros(n_neurons, dtype=np.uint8)

    def update(self, inputs):
        self.voltage = self.voltage * self.leak + inputs
        self.spike = (self.voltage >= self.threshold)
        self.voltage = np.where(self.spike, 0.0, self.voltage)
        fired = self.spike.astype(np.uint8)
        self.state = np.where(fired == 1, 1, self.state)
        self.threshold = np.where(self.state == 1, 1.5, 1.0)
        return self.spike

# -------------------------------------------------------
#  Hebbian Learning (from brain_core_learning.py)
# -------------------------------------------------------
class HebbianLearning:
    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate

    def update_weights(self, weights, pre_activity, post_activity):
        # Ensure post_activity is a column vector and pre_activity is a row vector
        delta_w = self.lr * np.outer(post_activity, pre_activity)
        new_weights = weights + delta_w
        new_weights = np.clip(new_weights, -1.0, 1.0)
        return new_weights

# -------------------------------------------------------
#  Topic Controller (from brain_core_learning.py)
# -------------------------------------------------------
class TopicController:
    def __init__(self):
        self.topic_patterns = {
            "math": r"\b(\d+|plus|minus|times|divide|equals?|add|subtract|multiply|calculation|solve)\b",
            "greeting": r"\b(hello|hi|hey|greetings?|howdy)\b",
            "question": r"\b(what|why|how|when|where|who|which|define|explain|tell)\b",
            "science": r"\b(science|physics|chemistry|biology|charge|electron|atom|energy|force|gravity)\b",
            "time": r"\b(time|day|date|today|tomorrow|yesterday|hour|minute|when)\b",
        }

    def extract_topics(self, text):
        text_lower = text.lower()
        detected = []
        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(topic)
        return detected if detected else ["general"]

# -------------------------------------------------------
#  Combined Brain - Advanced Integration
# -------------------------------------------------------
class CombinedBrain:
    def __init__(self, n_neurons=100_000, fan_in=32, radius=16, output_size=1024):
        self.n_neurons = n_neurons
        self.output_size = output_size
        
        print(f"Initializing Combined Brain with {n_neurons:,} neurons and {output_size} output concepts...")

        # Spiking Reservoir (from brain_core_cpu)
        self.reservoir = HybridNeuronModel(n_neurons)
        self.inputs_idx = np.random.randint(0, n_neurons, size=(n_neurons, fan_in), dtype=np.int32)
        self.recurrent_weights = (0.1 * np.random.randn(n_neurons, fan_in)).astype(np.float32)
        self.bias = np.zeros(n_neurons, dtype=np.float32)
        self.global_drive = 0.05
        self.noise_sigma = 0.03
        self.neigh_strength = 0.02
        self.radius = radius
        self.firing_avg = np.zeros(n_neurons, dtype=np.float32)
        self.target_rate = 0.05

        # Learning & Generation (from brain_core_learning)
        self.hebbian = HebbianLearning(learning_rate=0.0005)
        self.topic_controller = TopicController()
        if HAS_TRANSFORMERS:
            self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            self.gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self.gpt_model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None
            self.gpt_model = None

        # Readout Layer (new)
        self.W_readout = np.random.randn(output_size, n_neurons).astype(np.float32) * 0.05
        self.b_readout = np.zeros(output_size, dtype=np.float32)
        self._init_concept_words()

        # State
        self.conversation_history = []
        self.total_training_examples = 0
        self.total_interactions = 0

        self._load_brain_state()
        print("Combined Brain initialization complete!")

    def _init_concept_words(self):
        if os.path.exists(CONCEPT_WORDS_FILE):
            with open(CONCEPT_WORDS_FILE, 'r') as f:
                self.concept_words = json.load(f)
            if len(self.concept_words) != self.output_size:
                print("Warning: Concept words file size mismatch. Re-initializing.")
                self._generate_concept_words()
        else:
            self._generate_concept_words()

    def _generate_concept_words(self):
        # For simplicity, using generic concept words. A better approach would be to use a real vocabulary.
        words = ["concept" + str(i) for i in range(self.output_size)]
        self.concept_words = words
        with open(CONCEPT_WORDS_FILE, 'w') as f:
            json.dump(self.concept_words, f)

    def step_reservoir(self, external_input=None):
        presyn = self.reservoir.spike.astype(np.float32)
        incoming = presyn[self.inputs_idx]
        syn_current = np.sum(incoming * self.recurrent_weights, axis=1)
        
        neigh_sum = np.zeros(self.n_neurons, dtype=np.float32)
        for offset in range(-self.radius, self.radius + 1):
            if offset != 0:
                neigh_sum += np.roll(presyn, offset)
        neigh_drive = self.neigh_strength * neigh_sum

        total = syn_current + neigh_drive + self.bias + self.global_drive
        total += self.noise_sigma * np.random.randn(self.n_neurons).astype(np.float32)
        
        if external_input is not None:
            total += external_input

        self.firing_avg = 0.98 * self.firing_avg + 0.02 * presyn
        delta = (self.firing_avg - self.target_rate) * 0.5
        self.reservoir.threshold = np.clip(self.reservoir.threshold + delta, 0.5, 2.5)
        
        return self.reservoir.update(total)

    def _encode_text_to_reservoir_input(self, text):
        stim = np.zeros(self.n_neurons, dtype=np.float32)
        for i, char in enumerate(text):
            hash_val = hash(char) % self.n_neurons
            stim[hash_val] += 0.5
        return stim

    def _encode_text_to_output_target(self, text):
        target = np.zeros(self.output_size, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            hash_val = hash(word) % self.output_size
            target[hash_val] = 1.0
        return target

    def forward_pass(self, text, steps=80):
        input_stim = self._encode_text_to_reservoir_input(text)
        for _ in range(steps):
            spikes = self.step_reservoir(external_input=input_stim)
        
        # Readout from the reservoir's final state
        readout_input = spikes.astype(np.float32)
        output_activity = np.tanh(self.W_readout @ readout_input + self.b_readout)
        return output_activity, readout_input

    def learn_from_example(self, user_text, expected_response):
        _, reservoir_spikes = self.forward_pass(user_text)
        target_output = self._encode_text_to_output_target(expected_response)
        
        self.W_readout = self.hebbian.update_weights(self.W_readout, reservoir_spikes, target_output)
        
        self.total_training_examples += 1

    def generate_response(self, user_text):
        if not self.gpt_model:
            return "I am not equipped to respond without the necessary libraries.", 0.0, [], []

        output_activity, _ = self.forward_pass(user_text)
        
        # Use output activity to influence GPT-2 prompt
        top_indices = np.argsort(output_activity)[-5:] # top 5 concepts
        context_keywords = [self.concept_words[i] for i in top_indices if output_activity[i] > 0.1]
        
        topics = self.topic_controller.extract_topics(user_text)
        
        prompt = "You are a helpful assistant.\n"
        if context_keywords:
            prompt += f"Context: {', '.join(context_keywords)}\n"
        if topics:
            prompt += f"Topic: {', '.join(topics)}\n\n"
        
        for role, text in self.conversation_history[-3:]:
            prompt += f"{role}: {text}\n"
        prompt += f"User: {user_text}\nAssistant:"
        
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            outputs = self.gpt_model.generate(encoded['input_ids'], attention_mask=encoded['attention_mask'],
                max_new_tokens=40, temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        response = response.replace("\n", " ").strip()
        if not response:
            response = "I am not sure how to respond to that."
            
        return response, np.mean(output_activity**2), topics, context_keywords

    def interact(self, user_text):
        response, confidence, topics, keywords = self.generate_response(user_text)
        self.conversation_history.append(("User", user_text))
        self.conversation_history.append(("Assistant", response))
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        self.total_interactions += 1
        return response, confidence, topics, keywords

    def _load_brain_state(self):
        if os.path.exists(PATH_FILE):
            try:
                data = np.load(PATH_FILE)
                self.recurrent_weights = data["recurrent_weights"]
                self.W_readout = data["W_readout"]
                self.b_readout = data["b_readout"]
                self.total_training_examples = int(data.get("total_training", 0))
                self.total_interactions = int(data.get("total_interactions", 0))
                print(f"✓ Loaded saved brain state from {PATH_FILE}!")
            except Exception as e:
                print(f"Could not load brain state: {e}")

    def save_brain_state(self):
        data = {
            "recurrent_weights": self.recurrent_weights,
            "W_readout": self.W_readout,
            "b_readout": self.b_readout,
            "total_training": self.total_training_examples,
            "total_interactions": self.total_interactions,
        }
        np.savez(PATH_FILE, **data)
        print(f"\n✓ Saved brain state to {PATH_FILE}")

def main():
    if not HAS_TRANSFORMERS:
        print("\nERROR: Requires 'transformers' and 'torch'. Please run: pip install transformers torch")
        return

    brain = CombinedBrain()

    if brain.total_training_examples == 0:
        print("\n" + "="*70)
        print("FIRST RUN - Would you like to pre-train on example conversations?")
        DEFAULT_TRAINING_CONVERSATIONS = [
            {"user": "hello", "assistant": "Hi there! How can I help?"},
            {"user": "what is your name", "assistant": "I am a combined brain, a hybrid neural system."},
            {"user": "how do you work", "assistant": "I process information in a spiking neural network and use that to inform my responses."},
            {"user": "what is 2+2", "assistant": "The answer is 4."},
        ]
        choice = input(f"\nTrain on {len(DEFAULT_TRAINING_CONVERSATIONS)} built-in examples? (y/n): ").strip().lower()
        if choice == 'y':
            for conv in DEFAULT_TRAINING_CONVERSATIONS:
                brain.learn_from_example(conv["user"], conv["assistant"])
            brain.save_brain_state()

    print(f"\n{'='*70}\nREADY. Type 'train' for manual training, 'stats' for stats, or 'quit' to exit.\n{'='*70}\n")

    try:
        while True:
            user = input("You: ").strip()
            if not user: continue
            if user.lower() == 'quit': break
            
            if user.lower() == 'stats':
                print(f"\nTraining examples: {brain.total_training_examples}, Interactions: {brain.total_interactions}\n")
                continue

            if user.lower() == 'train':
                print("\nEnter training examples (type 'done' when finished):")
                while True:
                    user_ex = input("  User: ").strip()
                    if user_ex.lower() == 'done': break
                    asst_ex = input("  Assistant: ").strip()
                    if asst_ex:
                        brain.learn_from_example(user_ex, asst_ex)
                        print("  ✓ Learned!")
                brain.save_brain_state()
                continue

            start_time = time.time()
            response, confidence, topics, keywords = brain.interact(user)
            elapsed = time.time() - start_time

            print(f"\n  [Topics: {', '.join(topics)} | Context: {', '.join(keywords)} | Confidence: {confidence:.3f}]")
            print(f"Brain: {response}")
            print(f"  (Time: {elapsed:.2f}s)\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    
    brain.save_brain_state()
    print("Done!")

if __name__ == "__main__":
    main()
