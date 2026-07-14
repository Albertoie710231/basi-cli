CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O2
# -Wno-unused-function: llama.cpp's jinja headers define a couple of unused
# static helpers; that's upstream, not our code.
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Wno-unused-function

# Point these at your llama.cpp checkout + build dir. Defaults assume a clone at
# $HOME/llama.cpp built with the Vulkan backend (build_vulkan) — Vulkan runs on
# any NVIDIA/AMD/Intel GPU. Override for a different location or backend, e.g.:
#   make LLAMA_DIR=/path/to/llama.cpp LLAMA_BUILD=/path/to/llama.cpp/build_cuda
LLAMA_DIR ?= $(HOME)/llama.cpp
LLAMA_BUILD ?= $(LLAMA_DIR)/build_vulkan

# BASI no longer LINKS libllama — generation, templating, tool grammar, embeddings
# all run in spawned llama-server processes (HTTP). It still needs a couple of
# llama.cpp HEADERS at compile time (the llama_chat_message POD + opaque handle
# types) and the vendored nlohmann/json header — compile-time only, so a llama.cpp
# rebuild can no longer ABI-break the binary (the old free(): invalid pointer gotcha).
INCLUDES = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -Isrc
CXXINCLUDES = $(INCLUDES) -I$(LLAMA_DIR)/vendor

LDFLAGS =
# -lvulkan: hwinfo.c probes GPU VRAM directly via Vulkan (stable system lib, no
# llama.cpp ABI coupling) for the picker's fit estimate.
LIBS = -lm -lvulkan

TARGET = basi-cli
SRCS = $(wildcard src/*.c)
CXXSRCS = $(wildcard src/*.cpp)
OBJS = $(SRCS:.c=.o) $(CXXSRCS:.cpp=.o)

# Link with the C++ driver so libstdc++ is pulled in for the shim.
$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) src/*.o

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run
