#include "render.h"

#include <SDL.h>

void render_game(SDL_Renderer* renderer, const Game* game) {
  SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 220, 40, 40, 255);
  SDL_Rect food_rect = {game->food.x * CELL_SIZE, game->food.y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1};
  SDL_RenderFillRect(renderer, &food_rect);

  SDL_SetRenderDrawColor(renderer, 0, 220, 80, 255);
  for (int i = 0; i < game->snake.length; i++) {
    SDL_Rect rect = {game->snake.body[i].x * CELL_SIZE, game->snake.body[i].y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1};

    SDL_RenderFillRect(renderer, &rect);
  }

  SDL_RenderPresent(renderer);
}