#!/usr/bin/env python3
"""
Training script for the GPU brain - feeds it many sentences
"""
import sys
import time

# Training sentences - simple patterns to help it learn
training_data = [
    # Greetings
    ("hello", "hi there"),
    ("hi", "hello"),
    ("good morning", "good morning to you"),
    ("how are you", "I am doing well"),
    ("what's up", "not much"),

    # Simple math
    ("what is 1 plus 1", "2"),
    ("what is 2 plus 2", "4"),
    ("what is 3 plus 3", "6"),
    ("what is 5 plus 5", "10"),
    ("what is 10 plus 10", "20"),

    # Simple facts
    ("what color is the sky", "blue"),
    ("what color is grass", "green"),
    ("what color is the sun", "yellow"),
    ("what is water", "H2O"),
    ("what is fire", "hot"),

    # Simple questions
    ("what is your name", "I am a brain"),
    ("who are you", "I am a neural network"),
    ("where are you", "I am in a computer"),
    ("what are you", "I am artificial intelligence"),
    ("why are you here", "to learn and help"),

    # Animals
    ("what sound does a dog make", "woof"),
    ("what sound does a cat make", "meow"),
    ("what sound does a cow make", "moo"),
    ("what sound does a bird make", "tweet"),
    ("what do dogs eat", "food"),

    # Colors
    ("name a color", "red"),
    ("another color", "blue"),
    ("one more color", "green"),
    ("tell me a color", "yellow"),
    ("what is a color", "purple"),

    # Numbers
    ("count to three", "one two three"),
    ("what comes after one", "two"),
    ("what comes after two", "three"),
    ("what comes before two", "one"),
    ("first number", "one"),

    # Yes/No
    ("is the sky blue", "yes"),
    ("is grass red", "no"),
    ("is water wet", "yes"),
    ("can birds fly", "yes"),
    ("can fish fly", "no"),

    # Simple vocab
    ("what is good", "positive"),
    ("what is bad", "negative"),
    ("what is happy", "joyful"),
    ("what is sad", "unhappy"),
    ("what is big", "large"),

    # Opposites
    ("opposite of hot", "cold"),
    ("opposite of big", "small"),
    ("opposite of up", "down"),
    ("opposite of left", "right"),
    ("opposite of day", "night"),

    # More greetings
    ("hey", "hello"),
    ("howdy", "hi"),
    ("greetings", "hello there"),
    ("welcome", "thank you"),
    ("goodbye", "farewell"),

    # Time
    ("what is morning", "early day"),
    ("what is night", "late day"),
    ("what is noon", "middle of day"),
    ("what is evening", "end of day"),
    ("when is bedtime", "at night"),

    # Emotions
    ("what is love", "affection"),
    ("what is anger", "rage"),
    ("what is fear", "scared"),
    ("what is joy", "happiness"),
    ("what is surprise", "unexpected"),

    # Simple actions
    ("what is walk", "move with legs"),
    ("what is run", "fast walk"),
    ("what is jump", "leap up"),
    ("what is sleep", "rest"),
    ("what is eat", "consume food"),

    # Weather
    ("what is rain", "water falling"),
    ("what is snow", "frozen rain"),
    ("what is wind", "moving air"),
    ("what is sun", "bright star"),
    ("what is cloud", "water vapor"),

    # Shapes
    ("what is a circle", "round shape"),
    ("what is a square", "four sides"),
    ("what is a triangle", "three sides"),
    ("what is a line", "straight mark"),
    ("what is a dot", "small point"),

    # Foods
    ("what is bread", "baked food"),
    ("what is water", "drink"),
    ("what is fruit", "sweet food"),
    ("what is meat", "protein"),
    ("what is vegetable", "plant food"),

    # Final set
    ("tell me something", "okay"),
    ("speak", "hello"),
    ("say anything", "something"),
    ("respond", "yes"),
    ("are you there", "yes I am here"),
]

def main():
    print("=" * 60)
    print("TRAINING THE BRAIN")
    print("=" * 60)
    print(f"Training with {len(training_data)} sentence pairs")
    print("This will take several minutes...")
    print()

    for i, (input_text, expected_output) in enumerate(training_data, 1):
        print(f"\n[{i}/{len(training_data)}] Training on: '{input_text}'")
        print(f"  Expected: '{expected_output}'")

        # Send the input
        print(input_text, flush=True)
        time.sleep(0.1)  # Small delay to let the brain process

        # Give positive reward for training
        print("/good", flush=True)
        time.sleep(0.05)

        if i % 10 == 0:
            # Check stats every 10 sentences
            print("/stats", flush=True)
            time.sleep(0.1)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNow testing with some queries...")
    print()

    # Test queries
    test_queries = [
        "hello",
        "what is 2 plus 2",
        "what color is the sky",
        "what is your name",
        "goodbye"
    ]

    for query in test_queries:
        print(f"\nTest: {query}")
        print(query, flush=True)
        time.sleep(0.2)

    print("\n/stats", flush=True)
    time.sleep(0.2)

if __name__ == "__main__":
    main()
