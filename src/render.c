#include "render.h"

#include <SDL.h>

static void render_cell(SDL_Renderer* renderer, Vec2 position) {
  SDL_Rect rect = {position.x * CELL_SIZE, position.y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1};
  SDL_RenderFillRect(renderer, &rect);
}

void render_game(SDL_Renderer* renderer, const Game* game) {
  SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 220, 40, 40, 255);
  render_cell(renderer, game->food);

  for (int i = game->snake.length - 1; i >= 0; i--) {
    if (i == 0) {
      SDL_SetRenderDrawColor(renderer, 255, 240, 80, 255);
    } else if (i == game->snake.length - 1) {
      SDL_SetRenderDrawColor(renderer, 0, 130, 70, 255);
    } else {
      SDL_SetRenderDrawColor(renderer, 0, 220, 80, 255);
    }

    render_cell(renderer, game->snake.body[i]);
  }

  SDL_RenderPresent(renderer);
}