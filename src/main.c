#include <SDL.h>
#include <stdbool.h>
#include <stdio.h>

#include "SDL_keycode.h"
#include "SDL_render.h"
#include "SDL_video.h"
#include "brain.h"
#include "config.h"
#include "render.h"
#include "training.h"

#define WINDOW_WIDTH (GRID_WIDTH * CELL_SIZE)
#define WINDOW_HEIGHT (GRID_HEIGHT * CELL_SIZE)

#define RENDER_FPS 60
#define FRAME_DELAY_MS (1000 / RENDER_FPS)

#define SNAKE_MOVES_PER_SECOND 240
#define SNAKE_UPDATE_DELAY_MS (1000 / SNAKE_MOVES_PER_SECOND)

static bool init_sdl(SDL_Window** window, SDL_Renderer** renderer) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    return false;
  }

  *window = SDL_CreateWindow("Snake AI", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);

  if (*window == NULL) {
    fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
    SDL_Quit();
    return false;
  }

  *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);

  if (*renderer == NULL) {
    fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
    SDL_DestroyWindow(*window);
    SDL_Quit();
    return false;
  }

  return true;
}

static void cleanup_sdl(SDL_Window* window, SDL_Renderer* renderer) {
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

static void handle_input(bool* running) {
  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      *running = false;
    }

    if (event.type == SDL_KEYDOWN) {
      switch (event.key.keysym.sym) {
        case SDLK_ESCAPE:
          *running = false;
          break;
      }
    }
  }
}

static bool update_simulation(Game* game, TrainingSession* session) {
  Direction direction = brain_choose_direction(&training_session_best_agent(session)->brain, game);
  game_set_direction(game, direction);

  game_update(game);

  if (!game->alive || game->steps >= MAX_GAME_STEPS || game->steps_since_food >= MAX_STEPS_WITHOUT_FOOD) {
    return false;
  }

  return true;
}

static void update_window_title(SDL_Window* window, const Game* game) {
  char title[128];

  snprintf(title, sizeof(title), "Snake AI | Score: %d", game->score);

  SDL_SetWindowTitle(window, title);
}

int main(int argc, char* argv[]) {
  AppConfig config;
  AppConfigParseResult parse_result = app_config_parse(argc, argv, &config);
  if (parse_result == APP_CONFIG_PARSE_HELP) {
    return 0;
  }
  if (parse_result == APP_CONFIG_PARSE_ERROR) {
    return 1;
  }

  app_config_seed_random(&config);
  app_config_print_summary(&config);

  if (config.no_render) {
    return training_run_headless(&config);
  }

  if (!config.replay_only) {
    fprintf(stderr, "Rendered training is not supported. Use --train --no-render, or use --replay to watch a saved brain.\n");
    return 1;
  }

  SDL_Window* window = NULL;
  SDL_Renderer* renderer = NULL;
  if (!init_sdl(&window, &renderer)) {
    return 1;
  }

  bool running = true;

  Game game;
  game_init(&game);

  TrainingSession training_session;
  if (!training_session_init(&training_session, &config)) {
    cleanup_sdl(window, renderer);
    return 1;
  }

  Uint32 last_update = SDL_GetTicks();
  const Uint32 update_delay = SNAKE_UPDATE_DELAY_MS;
  bool simulation_active = true;

  while (running) {
    handle_input(&running);

    Uint32 now = SDL_GetTicks();

    if (simulation_active && now - last_update >= update_delay) {
      simulation_active = update_simulation(&game, &training_session);
      update_window_title(window, &game);
      last_update = now;
    }

    render_game(renderer, &game);
    SDL_Delay(FRAME_DELAY_MS);
  }

  cleanup_sdl(window, renderer);

  return 0;
}