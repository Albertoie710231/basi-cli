CC = gcc
CFLAGS = -Wall -Wextra -O2

LLAMA_DIR = /home/alberto/llama.cpp
LLAMA_BUILD = $(LLAMA_DIR)/build_vulkan

INCLUDES = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -Isrc
LDFLAGS = -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(LLAMA_BUILD)/bin
LIBS = -lllama -lggml -lggml-base -lm

TARGET = basi-cli
SRCS = $(wildcard src/*.c)
OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) src/*.o

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run
