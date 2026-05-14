CC = cc
CFLAGS = -Wall -Wextra -std=c99 -O3 -march=native -pthread -Isrc/vendor/zero-tensor
SDL_CFLAGS = $(shell sdl2-config --cflags)
SDL_LIBS = $(shell sdl2-config --libs)

SRC = src/main.c src/config.c src/training.c src/game.c src/brain.c src/genetic.c src/vendor/zero-tensor/tensor.c
OUT = out/snake-ai

all:
	mkdir -p out
	$(CC) $(CFLAGS) $(SDL_CFLAGS) $(SRC) -o $(OUT) $(SDL_LIBS) -lm -pthread

run: all
	./$(OUT)

build: all

clean:
	rm -f $(OUT)
