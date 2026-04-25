#ifndef RENDER_H
#define RENDER_H

#include "game.h"
#include <SDL.h>

#define CELL_SIZE 30

void render_game(SDL_Renderer *renderer, const Game *game);

#endif