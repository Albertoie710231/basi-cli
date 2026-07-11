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

INCLUDES = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -Isrc
# The C++ shim (chat_tmpl.cpp) calls llama.cpp's chat-template engine, so it
# also needs the common headers and the vendored nlohmann/json headers.
CXXINCLUDES = $(INCLUDES) -I$(LLAMA_DIR)/common -I$(LLAMA_DIR)/vendor

LDFLAGS = -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(LLAMA_BUILD)/bin
# -lllama-common: the SHARED common lib (jinja chat-template engine), the same
# one llama-cli links — consistent with libllama.so.
LIBS = -lllama-common -lllama -lggml -lggml-base -lm -lvulkan

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
