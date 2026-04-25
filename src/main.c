#include "SDL_keycode.h"
#include "SDL_render.h"
#include "SDL_video.h"
#include "ai.h"
#include "game.h"
#include <SDL.h>
#include <stdbool.h>
#include <stdio.h>

#define CELL_SIZE 30

#define GRID_WIDTH 20
#define GRID_HEIGHT 20

#define WINDOW_WIDTH (GRID_WIDTH * CELL_SIZE)
#define WINDOW_HEIGHT (GRID_HEIGHT * CELL_SIZE)

#define RENDER_FPS 60
#define FRAME_DELAY_MS (1000 / RENDER_FPS)

#define SNAKE_MOVES_PER_SECOND 15
#define SNAKE_UPDATE_DELAY_MS (1000 / SNAKE_MOVES_PER_SECOND)

bool ai_enabled = true;

void draw_game(SDL_Renderer *renderer, const Game *game) {
  SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 220, 40, 40, 255);
  SDL_Rect food_rect = {game->food.x * CELL_SIZE, game->food.y * CELL_SIZE,
                        CELL_SIZE - 1, CELL_SIZE - 1};
  SDL_RenderFillRect(renderer, &food_rect);

  SDL_SetRenderDrawColor(renderer, 0, 220, 80, 255);
  for (int i = 0; i < game->snake.length; i++) {
    SDL_Rect rect = {game->snake.body[i].x * CELL_SIZE,
                     game->snake.body[i].y * CELL_SIZE, CELL_SIZE - 1,
                     CELL_SIZE - 1};

    SDL_RenderFillRect(renderer, &rect);
  }

  SDL_RenderPresent(renderer);
}

static void handle_input(bool *running, Game *game) {
  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      *running = false;
    }

    if (event.type == SDL_KEYDOWN) {
      switch (event.key.keysym.sym) {
      case SDLK_UP:
        game_set_direction(game, UP);
        break;
      case SDLK_DOWN:
        game_set_direction(game, DOWN);
        break;
      case SDLK_LEFT:
        game_set_direction(game, LEFT);
        break;
      case SDLK_RIGHT:
        game_set_direction(game, RIGHT);
        break;
      case SDLK_ESCAPE:
        *running = false;
        break;
      case SDLK_SPACE:
        ai_enabled = !ai_enabled;
        break;
      case SDLK_r:
        game_init(game);
        break;
      }
    }
  }
}

int main(void) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  SDL_Window *window = SDL_CreateWindow("Snake AI", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH,
                                        WINDOW_HEIGHT, SDL_WINDOW_SHOWN);

  if (window == NULL) {
    fprintf(stderr, "Window could not be created! SDL_Error: %s\n",
            SDL_GetError());
    SDL_Quit();
    return 1;
  }

  SDL_Renderer *renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

  if (renderer == NULL) {
    fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n",
            SDL_GetError());
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }

  bool running = true;

  Game game;
  game_init(&game);

  Uint32 last_update = SDL_GetTicks();
  const Uint32 update_delay = SNAKE_UPDATE_DELAY_MS;

  while (running) {
    handle_input(&running, &game);

    Uint32 now = SDL_GetTicks();

    if (now - last_update >= update_delay) {
      if (ai_enabled) {
        Direction direction = choose_direction_toward_food(&game);
        game_set_direction(&game, direction);
      }

      game_update(&game);
      last_update = now;
    }

    draw_game(renderer, &game);
    SDL_Delay(FRAME_DELAY_MS);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}