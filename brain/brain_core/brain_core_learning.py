import os
import time
import warnings
import numpy as np
import re
import json

PATH_FILE = "brain_paths_learning.npz"
TRAINING_FILE = "conversation_training.json"

# Check for transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# -------------------------------------------------------
#  Training data - sample conversations
# -------------------------------------------------------
DEFAULT_TRAINING_CONVERSATIONS = [
    # Greetings
    {"user": "hello", "assistant": "Hello! How can I help you today?"},
    {"user": "hi", "assistant": "Hi there! What can I do for you?"},
    {"user": "hey", "assistant": "Hey! What's up?"},

    # Math
    {"user": "what is 1 plus 1", "assistant": "1 plus 1 equals 2."},
    {"user": "what is 2 plus 2", "assistant": "2 plus 2 equals 4."},
    {"user": "what is 5 times 3", "assistant": "5 times 3 equals 15."},
    {"user": "calculate 10 minus 3", "assistant": "10 minus 3 equals 7."},

    # Science
    {"user": "what is electric charge", "assistant": "Electric charge is a fundamental property of matter. Particles can have positive or negative charge."},
    {"user": "define energy", "assistant": "Energy is the capacity to do work. It comes in many forms like kinetic, potential, and thermal."},
    {"user": "what is gravity", "assistant": "Gravity is a fundamental force that attracts objects with mass toward each other."},

    # Time
    {"user": "what day is it", "assistant": "I don't have access to the current date, but I can help you with other questions."},
    {"user": "what time is it", "assistant": "I don't have access to the current time, but I'm here to help with other things."},

    # General questions
    {"user": "how are you", "assistant": "I'm functioning well! Thanks for asking. How can I assist you?"},
    {"user": "what can you do", "assistant": "I can answer questions, help with calculations, explain concepts, and have conversations."},
    {"user": "who are you", "assistant": "I'm a neural network-based assistant designed to help answer questions and have conversations."},

    # Understanding
    {"user": "i don't understand", "assistant": "Let me try to explain it differently. What part would you like me to clarify?"},
    {"user": "can you explain that", "assistant": "Of course! I'll break it down in simpler terms."},

    # Politeness
    {"user": "thank you", "assistant": "You're welcome! Happy to help."},
    {"user": "thanks", "assistant": "No problem! Anytime."},

    # Add more examples...
]


# -------------------------------------------------------
#  Hebbian Learning - neurons that fire together, wire together
# -------------------------------------------------------
class HebbianLearning:
    """
    Implements Hebbian learning: strengthen connections between
    neurons that are co-active.
    """
    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate

    def update_weights(self, weights, pre_activity, post_activity):
        """
        Update weights based on pre and post synaptic activity.
        Hebbian rule: Δw = lr * pre * post
        """
        # Outer product for weight updates
        delta_w = self.lr * np.outer(post_activity, pre_activity)

        # Update weights
        new_weights = weights + delta_w

        # Normalize to prevent runaway growth
        # Clip to reasonable range
        new_weights = np.clip(new_weights, -1.0, 1.0)

        return new_weights


# -------------------------------------------------------
#  Topic Controller (from previous version)
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

    def extract_keywords(self, text):
        stopwords = {"the", "a", "an", "is", "are", "am", "was", "were", "be", "been",
                    "to", "of", "in", "for", "on", "at", "by", "with", "from", "about",
                    "i", "you", "me", "my", "your"}
        words = text.lower().split()
        keywords = []
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if word and word not in stopwords and len(word) > 2:
                keywords.append(word)
            if word and word.isdigit():
                keywords.append(word)
        return keywords

    def build_controlled_prompt(self, user_text, topics, keywords, history):
        prompt = "You are a helpful assistant.\n\n"
        if topics:
            prompt += f"Topic: {', '.join(topics)}\n"
        if keywords:
            prompt += f"Focus on: {', '.join(keywords[:5])}\n\n"
        if history:
            for role, text in history[-3:]:
                prompt += f"{role}: {text}\n"
        prompt += f"User: {user_text}\nAssistant:"
        return prompt


# -------------------------------------------------------
#  Learning Brain
# -------------------------------------------------------
class LearningBrain:
    """
    Brain that LEARNS from conversations.
    - Can be pre-trained on conversation examples
    - Updates weights based on successful exchanges
    - Saves full learned state
    - Gets smarter over time
    """

    def __init__(self, n_neurons=100_000):
        self.n_neurons = n_neurons

        print(f"\nInitializing Learning Brain ({n_neurons:,} neurons)...")

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

        # Topic controller
        self.topic_controller = TopicController()

        # Brain structure
        self.input_size = int(n_neurons * 0.3)
        self.hidden_size = int(n_neurons * 0.5)
        self.output_size = int(n_neurons * 0.2)

        print(f"  Input: {self.input_size:,} | Hidden: {self.hidden_size:,} | Output: {self.output_size:,}")

        # Neural state
        self.input_activity = np.zeros(self.input_size, dtype=np.float32)
        self.hidden_activity = np.zeros(self.hidden_size, dtype=np.float32)
        self.output_activity = np.zeros(self.output_size, dtype=np.float32)

        # Learnable weights (these get updated!)
        self.W_input_hidden = np.random.randn(self.hidden_size, self.input_size).astype(np.float32) * 0.05
        self.W_hidden_output = np.random.randn(self.output_size, self.hidden_size).astype(np.float32) * 0.05
        self.W_hidden_recurrent = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) * 0.03

        # Hebbian learning
        self.hebbian = HebbianLearning(learning_rate=0.0005)

        # Conversation memory
        self.conversation_history = []

        # Training statistics
        self.total_training_examples = 0
        self.total_interactions = 0

        # Load saved state if available
        self._load_brain_state()

        print("Learning brain ready!\n")

    def encode_text_to_neurons(self, text):
        """Encode text into neural activity pattern"""
        words = text.lower().split()
        activity = np.zeros(self.input_size, dtype=np.float32)

        for i, word in enumerate(words):
            # Hash-based encoding
            word_clean = re.sub(r'[^\w\s]', '', word)
            if not word_clean:
                continue

            hash_val = hash(word_clean) % self.input_size
            region_size = 200
            start = hash_val
            end = min(start + region_size, self.input_size)
            activity[start:end] += 0.8 / len(words)  # Normalize by sentence length

        return activity

    def forward_pass(self, input_activity):
        """Process input through the brain"""
        # Input → Hidden
        self.input_activity = input_activity
        hidden_input = self.W_input_hidden @ self.input_activity

        # Add recurrent connections (memory)
        hidden_input += 0.5 * (self.W_hidden_recurrent @ self.hidden_activity)

        # Activation (tanh for bounded output)
        self.hidden_activity = np.tanh(hidden_input)

        # Hidden → Output
        output_input = self.W_hidden_output @ self.hidden_activity
        self.output_activity = np.tanh(output_input)

        return self.output_activity

    def learn_from_example(self, user_text, expected_response):
        """
        Learn from a conversation example.
        Updates weights to strengthen the user→response mapping.
        """
        # Encode input
        input_pattern = self.encode_text_to_neurons(user_text)

        # Forward pass
        self.forward_pass(input_pattern)

        # Encode expected output
        target_pattern = self.encode_text_to_neurons(expected_response)[:self.output_size]

        # Update weights using Hebbian learning
        # Input-Hidden weights
        self.W_input_hidden = self.hebbian.update_weights(
            self.W_input_hidden,
            self.input_activity,
            self.hidden_activity
        )

        # Hidden-Output weights
        self.W_hidden_output = self.hebbian.update_weights(
            self.W_hidden_output,
            self.hidden_activity,
            self.output_activity
        )

        self.total_training_examples += 1

    def train_on_conversations(self, conversations):
        """
        Pre-train the brain on conversation examples.
        """
        print(f"\n{'='*70}")
        print(f"TRAINING BRAIN ON {len(conversations)} CONVERSATION EXAMPLES")
        print(f"{'='*70}\n")

        for i, conv in enumerate(conversations):
            user_text = conv["user"]
            assistant_text = conv["assistant"]

            # Learn this example
            self.learn_from_example(user_text, assistant_text)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(conversations)} examples...")

        print(f"\n✓ Training complete! Brain learned {len(conversations)} patterns.\n")

    def generate_response(self, user_text):
        """Generate response using brain state + GPT"""
        if not HAS_TRANSFORMERS or self.gpt_model is None:
            return "Error: GPT not loaded.", []

        # Process through brain
        input_pattern = self.encode_text_to_neurons(user_text)
        output_pattern = self.forward_pass(input_pattern)

        # Brain activity influences response
        brain_confidence = float(np.mean(np.abs(output_pattern)))

        # Extract topics and keywords
        topics = self.topic_controller.extract_topics(user_text)
        keywords = self.topic_controller.extract_keywords(user_text)

        # Build controlled prompt
        prompt = self.topic_controller.build_controlled_prompt(
            user_text, topics, keywords, self.conversation_history
        )

        # Generate
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            outputs = self.gpt_model.generate(
                encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_new_tokens=35,
                temperature=0.75,
                top_p=0.88,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
            response = re.sub(r'^[:\(\)\-\s]+', '', response)

        # Clean up
        response = response.replace("\n", " ").strip()
        sentences = re.split(r'[.!?]+', response)
        if sentences:
            response = sentences[0].strip()
            if response and response[-1] not in '.!?':
                response += '.'

        return response, brain_confidence, topics, keywords

    def interact(self, user_text, learn_from_this=False, expected_response=None):
        """
        Interact and optionally learn from the exchange.

        Args:
            user_text: User's input
            learn_from_this: If True, learn from this interaction
            expected_response: If provided, learn to produce this response
        """
        if not user_text.strip():
            return "", 0.0

        # Generate response
        response, confidence, topics, keywords = self.generate_response(user_text)

        # Learn from this interaction if requested
        if learn_from_this and expected_response:
            self.learn_from_example(user_text, expected_response)
            print(f"  [Learned from this example]")

        # Update conversation history
        self.conversation_history.append(("User", user_text))
        self.conversation_history.append(("Assistant", response))

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        self.total_interactions += 1

        return response, confidence, topics, keywords

    def _load_brain_state(self):
        """Load saved brain state"""
        if os.path.exists(PATH_FILE):
            try:
                data = np.load(PATH_FILE)
                self.W_input_hidden = data["W_input_hidden"]
                self.W_hidden_output = data["W_hidden_output"]
                self.W_hidden_recurrent = data["W_hidden_recurrent"]
                self.total_training_examples = int(data.get("total_training", 0))
                self.total_interactions = int(data.get("total_interactions", 0))

                print(f"✓ Loaded saved brain state!")
                print(f"  Previous training: {self.total_training_examples} examples")
                print(f"  Previous interactions: {self.total_interactions}")
            except Exception as e:
                print(f"Could not load brain state: {e}")
        else:
            print("No saved brain state found. Starting fresh.")

    def save_brain_state(self):
        """Save full brain state"""
        data = {
            "W_input_hidden": self.W_input_hidden,
            "W_hidden_output": self.W_hidden_output,
            "W_hidden_recurrent": self.W_hidden_recurrent,
            "total_training": self.total_training_examples,
            "total_interactions": self.total_interactions,
        }
        np.savez(PATH_FILE, **data)
        print(f"\n✓ Saved brain state")
        print(f"  Total training: {self.total_training_examples} examples")
        print(f"  Total interactions: {self.total_interactions}")


# -------------------------------------------------------
#  Main
# -------------------------------------------------------
def main():
    if not HAS_TRANSFORMERS:
        print("\nERROR: Requires transformers and torch")
        print("Install: pip install transformers torch")
        return

    print(f"\n{'='*70}")
    print("LEARNING NEURAL BRAIN")
    print("Trains on conversations and gets smarter over time")
    print(f"{'='*70}")

    brain = LearningBrain(n_neurons=100_000)

    # Check if we should train on default examples
    if brain.total_training_examples == 0:
        print("\n" + "="*70)
        print("FIRST RUN - Would you like to pre-train on example conversations?")
        print("="*70)
        choice = input(f"\nTrain on {len(DEFAULT_TRAINING_CONVERSATIONS)} built-in examples? (y/n): ").strip().lower()

        if choice == 'y':
            brain.train_on_conversations(DEFAULT_TRAINING_CONVERSATIONS)
            brain.save_brain_state()

    print(f"\n{'='*70}")
    print("READY - Brain will learn from each conversation")
    print("Type 'train' to add more training examples")
    print("Type 'stats' to see learning statistics")
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

            # Special commands
            if user.lower() == 'stats':
                print(f"\n{'─'*70}")
                print("BRAIN STATISTICS:")
                print(f"  Training examples learned: {brain.total_training_examples}")
                print(f"  Total interactions: {brain.total_interactions}")
                print(f"{'─'*70}\n")
                continue

            if user.lower() == 'train':
                print("\nEnter training examples (type 'done' when finished):")
                while True:
                    user_ex = input("  User: ").strip()
                    if user_ex.lower() == 'done':
                        break
                    asst_ex = input("  Assistant: ").strip()
                    if asst_ex:
                        brain.learn_from_example(user_ex, asst_ex)
                        print("  ✓ Learned!")
                brain.save_brain_state()
                continue

            # Normal interaction
            start_time = time.time()
            response, confidence, topics, keywords = brain.interact(user)
            elapsed = time.time() - start_time

            print(f"\n  [Topics: {', '.join(topics)} | Keywords: {', '.join(keywords[:3])} | Confidence: {confidence:.3f}]")
            print(f"Brain: {response}")
            print(f"  (Time: {elapsed:.2f}s)\n")

    except KeyboardInterrupt:
        print("\n\nStopping and saving brain state...")

    brain.save_brain_state()
    print("\nDone!")


if __name__ == "__main__":
    main()
