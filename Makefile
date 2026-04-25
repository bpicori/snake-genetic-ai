CC = cc
CFLAGS = -Wall -Wextra -std=c99
SDL_CFLAGS = $(shell sdl2-config --cflags)
SDL_LIBS = $(shell sdl2-config --libs)

SRC = src/main.c src/game.c src/ai.c src/brain.c
OUT = bin/snake-ai

all:
	$(CC) $(CFLAGS) $(SDL_CFLAGS) $(SRC) -o $(OUT) $(SDL_LIBS) -lm

run: all
	./$(OUT)

clean:
	rm -f $(OUT)
