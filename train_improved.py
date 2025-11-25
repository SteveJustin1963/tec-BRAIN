#!/usr/bin/env python3
"""
Improved training - uses the fixed brain with supervised learning
"""
import sys
import time

# Simple echo training - teach brain to respond with similar words
training_pairs = [
    "hello",
    "hi",
    "goodbye",
    "yes",
    "no",
    "one",
    "two",
    "red",
    "blue",
    "cat",
    "dog",
]

print("=" * 60)
print("IMPROVED TRAINING (With Fixes)")
print("=" * 60)
print("Fixes applied:")
print("  1. Global inhibition + stronger homeostatic control")
print("  2. Weight decay + reduced learning rates")
print("  3. Supervised echo learning")
print(f"\nTraining with {len(training_pairs)} words")
print()

for i, word in enumerate(training_pairs, 1):
    print(f"\n[{i}/{len(training_pairs)}] Training: '{word}'")
    print(word, flush=True)
    time.sleep(0.1)

    if i % 5 == 0:
        print("/stats", flush=True)
        time.sleep(0.1)

print("\n" + "=" * 60)
print("TESTING")
print("=" * 60)

test_words = ["hello", "goodbye", "one", "red"]
for word in test_words:
    print(f"\nTest: {word}")
    print(word, flush=True)
    time.sleep(0.2)

print("\n/stats", flush=True)
