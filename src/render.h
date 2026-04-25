#ifndef RENDER_H
#define RENDER_H

#include <SDL.h>

#include "game.h"

#define CELL_SIZE 30

void render_game(SDL_Renderer* renderer, const Game* game);

#endif