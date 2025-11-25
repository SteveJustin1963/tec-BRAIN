#!/usr/bin/env python3
"""
Gentle training script - uses weak rewards to prevent runaway excitation
"""
import sys
import time

# Simple training patterns
training_data = [
    "hello",
    "hi",
    "goodbye",
    "yes",
    "no",
    "one",
    "two",
    "three",
    "red",
    "blue",
    "what",
    "why",
    "how",
    "good",
    "bad",
]

print("=" * 60)
print("GENTLE TRAINING MODE")
print("=" * 60)
print(f"Training with {len(training_data)} simple words")
print("Using weak rewards (+0.3) to prevent hyperexcitation")
print()

for i, word in enumerate(training_data, 1):
    print(f"\n[{i}/{len(training_data)}] Training: '{word}'")
    print(word, flush=True)
    time.sleep(0.05)

    # Use weak reward
    print("/reward 0.3", flush=True)
    time.sleep(0.05)

    if i % 5 == 0:
        print("/stats", flush=True)
        time.sleep(0.1)

print("\n" + "=" * 60)
print("TRAINING COMPLETE - Testing responses")
print("=" * 60)

test_words = ["hello", "goodbye", "one", "red", "what"]
for word in test_words:
    print(f"\nTest: {word}")
    print(word, flush=True)
    time.sleep(0.2)

print("\n/stats", flush=True)
