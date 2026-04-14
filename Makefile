CC = gcc
CFLAGS = -Wall -Wextra -O2

LLAMA_DIR = /home/alberto/llama.cpp
LLAMA_BUILD = $(LLAMA_DIR)/build_vulkan

INCLUDES = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LDFLAGS = -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(LLAMA_BUILD)/bin
LIBS = -lllama -lggml -lggml-base -lm

TARGET = basi-cli
SRC = src/main.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run
