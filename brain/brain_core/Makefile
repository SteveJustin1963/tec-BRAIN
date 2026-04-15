# Brain Core — Spiking Neural Network
# Deps: sudo apt install libsdl2-dev
# LLM:  ollama serve  (uses gemma3:1b by default)

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lSDL2 -lpthread -lm

TARGET  = brain_core
SRC     = brain_core.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) brain_state.bin

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run
