CC = cc
CFLAGS = -Wall -Wextra -std=c99
SDL_CFLAGS = $(shell sdl2-config --cflags)
SDL_LIBS = $(shell sdl2-config --libs)

SRC = src/main.c src/game.c src/ai.c src/brain.c src/agent.c src/genetic.c src/render.c
OUT = out/snake-ai

all:
	mkdir -p out
	$(CC) $(CFLAGS) $(SDL_CFLAGS) $(SRC) -o $(OUT) $(SDL_LIBS) -lm

run: all
	./$(OUT)

build: all

clean:
	rm -f $(OUT)
