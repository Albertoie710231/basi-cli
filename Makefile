CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O2
# -Wno-unused-function: nlohmann/json.hpp defines some unused static helpers.
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Wno-unused-function

# BASI is now fully self-contained: it links NO libllama and needs NO llama.cpp
# checkout to build. Generation, templating, tool grammar/parsing and embeddings
# all run in spawned llama-server processes over HTTP; the only third-party header,
# nlohmann/json, is vendored under src/vendor. BASI just needs the `llama-server`
# BINARY present at runtime (default path in srvgen.c; override BASI_SERVER_BIN).
# Dropping the libllama link killed the old ABI gotcha (a llama.cpp rebuild used to
# crash basi-cli with free(): invalid pointer).
INCLUDES = -Isrc -Isrc/vendor
CXXINCLUDES = $(INCLUDES)

LDFLAGS =
# -lvulkan: hwinfo.c probes GPU VRAM directly via Vulkan (a stable system lib) for
# the picker's fit estimate.
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
