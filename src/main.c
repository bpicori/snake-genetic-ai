#include <SDL.h>
#include <stdbool.h>
#include <stdio.h>

#include "SDL_keycode.h"
#include "SDL_render.h"
#include "SDL_video.h"
#include "agent.h"
#include "config.h"
#include "game.h"
#include "genetic.h"
#include "render.h"

#define WINDOW_WIDTH (GRID_WIDTH * CELL_SIZE)
#define WINDOW_HEIGHT (GRID_HEIGHT * CELL_SIZE)

#define RENDER_FPS 60
#define FRAME_DELAY_MS (1000 / RENDER_FPS)

#define SNAKE_MOVES_PER_SECOND 120
#define SNAKE_UPDATE_DELAY_MS (1000 / SNAKE_MOVES_PER_SECOND)

bool ai_enabled = true;
static float best_fitness_ever = 0.0f;

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

static void handle_input(bool* running, Game* game) {
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

static Agent* train_generations(Population* population, int count, const AppConfig* config) {
  Agent* best_agent = NULL;

  for (int i = 0; i < count; i++) {
    population_evaluate_parallel(population);

    best_agent = &population->agents[population->best_agent_index];

    printf(
        "Strategy %s | generation %d | best %.2f | average %.2f | score %d | steps %d | distance %d | "
        "mutation %.3f %.3f\n",
        population_strategy_name(config->strategy), population->generation, best_agent->fitness, population->average_fitness,
        best_agent->score, best_agent->steps, best_agent->distance_reward, population->mutation_rate, population->mutation_strength);

    if (best_agent->fitness > best_fitness_ever) {
      best_fitness_ever = best_agent->fitness;
      if (brain_save(&best_agent->brain, config->brain_path)) {
        printf("Best brain saved to %s\n", config->brain_path);
      } else {
        printf("Failed to save best brain to %s\n", config->brain_path);
      }
    }

    population_next_generation(population, config->strategy);
  }

  population_evaluate_parallel(population);
  best_agent = &population->agents[population->best_agent_index];

  return best_agent;
}

static bool setup_best_agent(Population* population, Agent* saved_agent, Agent** best_agent, const AppConfig* config) {
  if (config->replay_only) {
    if (!brain_load(&saved_agent->brain, config->brain_path)) {
      printf("No saved brain found at %s\n", config->brain_path);
      return false;
    }

    printf("Replay mode: loaded saved brain from %s\n", config->brain_path);
    *best_agent = saved_agent;
    return true;
  }

  if (brain_load(&population->agents[0].brain, config->brain_path)) {
    population->agents[0].fitness = 0.0f;
    population->agents[0].score = 0;
    population->agents[0].steps = 0;

    printf("Best brain loaded from %s\n", config->brain_path);
  }

  return true;
}

static void update_simulation(Game* game, Agent** best_agent, Population* population, const AppConfig* config) {
  if (ai_enabled) {
    Direction direction = agent_choose_direction(*best_agent, game);
    game_set_direction(game, direction);
  }

  game_update(game);

  if (!game->alive || game->steps >= MAX_GAME_STEPS || game->steps_since_food >= MAX_STEPS_WITHOUT_FOOD) {
    if (config->replay_only) {
      game_init(game);
    } else {
      population_next_generation(population, config->strategy);
      *best_agent = train_generations(population, config->generations, config);
      game_init(game);
    }
  }
}

static void update_window_title(SDL_Window* window, const Game* game) {
  char title[128];

  snprintf(title, sizeof(title), "Snake AI | Score: %d", game->score);

  SDL_SetWindowTitle(window, title);
}

static int run_headless_training(const AppConfig* config) {
  if (config->replay_only) {
    fprintf(stderr, "--no-render is for training; use replay without --no-render.\n");
    return 1;
  }

  Population population;
  population_init(&population);

  if (brain_load(&population.agents[0].brain, config->brain_path)) {
    population.agents[0].fitness = 0.0f;
    population.agents[0].score = 0;
    population.agents[0].steps = 0;
    printf("Best brain loaded from %s\n", config->brain_path);
  }

  train_generations(&population, config->generations, config);

  return 0;
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
    return run_headless_training(&config);
  }

  SDL_Window* window = NULL;
  SDL_Renderer* renderer = NULL;
  if (!init_sdl(&window, &renderer)) {
    return 1;
  }

  bool running = true;

  Game game;
  game_init(&game);

  Population population;
  population_init(&population);

  Agent saved_agent = {0};
  Agent* best_agent = NULL;
  if (!setup_best_agent(&population, &saved_agent, &best_agent, &config)) {
    cleanup_sdl(window, renderer);
    return 1;
  }

  if (!config.replay_only) {
    best_agent = train_generations(&population, config.generations, &config);
  }

  Uint32 last_update = SDL_GetTicks();
  const Uint32 update_delay = SNAKE_UPDATE_DELAY_MS;

  while (running) {
    handle_input(&running, &game);

    Uint32 now = SDL_GetTicks();

    if (now - last_update >= update_delay) {
      update_simulation(&game, &best_agent, &population, &config);
      update_window_title(window, &game);
      last_update = now;
    }

    render_game(renderer, &game);
    SDL_Delay(FRAME_DELAY_MS);
  }

  cleanup_sdl(window, renderer);

  return 0;
}