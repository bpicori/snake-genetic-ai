#include <SDL.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "SDL_keycode.h"
#include "SDL_render.h"
#include "SDL_video.h"
#include "agent.h"
#include "game.h"
#include "genetic.h"
#include "render.h"

#define WINDOW_WIDTH (GRID_WIDTH * CELL_SIZE)
#define WINDOW_HEIGHT (GRID_HEIGHT * CELL_SIZE)

#define RENDER_FPS 60
#define FRAME_DELAY_MS (1000 / RENDER_FPS)

#define SNAKE_MOVES_PER_SECOND 30
#define GENERATIONS_PER_REPLAY 500
#define SNAKE_UPDATE_DELAY_MS (1000 / SNAKE_MOVES_PER_SECOND)

#define BEST_BRAIN_PATH "out/best.brain"

bool ai_enabled = true;
static float best_fitness_ever = 0.0f;

typedef struct {
  bool replay_only;
  PopulationStrategy strategy;
} AppConfig;

static void print_usage(const char* program_name) {
  printf("Usage: %s [--train|--replay] [--strategy v1|v2|v3|v4|adaptive]\n", program_name);
  printf("  --train              train and periodically replay the best brain (default)\n");
  printf("  --replay             load out/best.brain and replay without training\n");
  printf("  --strategy <name>    choose v1, v2, v3, v4, or adaptive\n");
}

static bool parse_args(int argc, char* argv[], AppConfig* config) {
  config->replay_only = false;
  config->strategy = POPULATION_STRATEGY_ADAPTIVE_MUTATION;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--train") == 0) {
      config->replay_only = false;
    } else if (strcmp(argv[i], "--replay") == 0) {
      config->replay_only = true;
    } else if (strcmp(argv[i], "--strategy") == 0) {
      if (i + 1 >= argc || !population_strategy_from_name(argv[i + 1], &config->strategy)) {
        fprintf(stderr, "Invalid or missing strategy.\n");
        print_usage(argv[0]);
        return false;
      }
      i++;
    } else if (strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return false;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      print_usage(argv[0]);
      return false;
    }
  }

  return true;
}

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

static Agent* train_generations(Population* population, int count, PopulationStrategy strategy) {
  Agent* best_agent = NULL;

  for (int i = 0; i < count; i++) {
    population_evaluate(population);

    best_agent = &population->agents[population->best_agent_index];

    printf(
        "Strategy %s | generation %d | best %.2f | average %.2f | score %d | steps %d | distance %d | "
        "mutation %.3f %.3f\n",
        population_strategy_name(strategy), population->generation, best_agent->fitness, population->average_fitness, best_agent->score,
        best_agent->steps, best_agent->distance_reward, population->mutation_rate, population->mutation_strength);

    if (best_agent->fitness > best_fitness_ever) {
      best_fitness_ever = best_agent->fitness;
      if (brain_save(&best_agent->brain, BEST_BRAIN_PATH)) {
        printf("Best brain saved to %s\n", BEST_BRAIN_PATH);
      } else {
        printf("Failed to save best brain to %s\n", BEST_BRAIN_PATH);
      }
    }

    population_next_generation(population, strategy);
  }

  population_evaluate(population);
  best_agent = &population->agents[population->best_agent_index];

  return best_agent;
}

static bool setup_best_agent(Population* population, Agent* saved_agent, Agent** best_agent, const AppConfig* config) {
  if (config->replay_only) {
    if (!brain_load(&saved_agent->brain, BEST_BRAIN_PATH)) {
      printf("No saved brain found at %s\n", BEST_BRAIN_PATH);
      return false;
    }

    printf("Replay mode: loaded saved brain from %s\n", BEST_BRAIN_PATH);
    *best_agent = saved_agent;
    return true;
  }

  if (brain_load(&population->agents[0].brain, BEST_BRAIN_PATH)) {
    population->agents[0].fitness = 0.0f;
    population->agents[0].score = 0;
    population->agents[0].steps = 0;

    printf("Best brain loaded from %s\n", BEST_BRAIN_PATH);
  }

  *best_agent = train_generations(population, GENERATIONS_PER_REPLAY, config->strategy);
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
      *best_agent = train_generations(population, GENERATIONS_PER_REPLAY, config->strategy);
      game_init(game);
    }
  }
}

static void update_window_title(SDL_Window* window, const Game* game) {
  char title[128];

  snprintf(title, sizeof(title), "Snake AI | Score: %d", game->score);

  SDL_SetWindowTitle(window, title);
}

int main(int argc, char* argv[]) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  AppConfig config;
  if (!parse_args(argc, argv, &config)) {
    return 1;
  }

  printf("Mode: %s | strategy: %s\n", config.replay_only ? "replay" : "train", population_strategy_name(config.strategy));

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