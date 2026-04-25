#include <SDL.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
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

#define SNAKE_MOVES_PER_SECOND 120
#define DEFAULT_GENERATIONS 500
#define SNAKE_UPDATE_DELAY_MS (1000 / SNAKE_MOVES_PER_SECOND)

#define DEFAULT_BRAIN_PATH "out/best.brain"

bool ai_enabled = true;
static float best_fitness_ever = 0.0f;

typedef struct {
  bool replay_only;
  bool no_render;
  int generations;
  const char* brain_path;
  const char* csv_path;
  PopulationStrategy strategy;
} AppConfig;

static void print_usage(const char* program_name) {
  printf(
      "Usage: %s [--train|--replay] [--no-render] [--generations count] "
      "[--brain path] [--csv path] [--strategy v1|v2|v3|v4|adaptive|adaptive-conservative]\n",
      program_name);
  printf("  --train              train and periodically replay the best brain (default)\n");
  printf("  --replay             load the saved brain and replay without training\n");
  printf("  --no-render          train without opening an SDL window\n");
  printf("  --generations <n>    generations to train per batch, or total in --no-render mode\n");
  printf("  --brain <path>       brain file to load/save (default: out/best.brain)\n");
  printf("  --csv <path>         write training metrics to a CSV file\n");
  printf("  --strategy <name>    choose v1, v2, v3, v4, adaptive, or adaptive-conservative\n");
}

static bool parse_args(int argc, char* argv[], AppConfig* config) {
  config->replay_only = false;
  config->no_render = false;
  config->generations = DEFAULT_GENERATIONS;
  config->brain_path = DEFAULT_BRAIN_PATH;
  config->csv_path = NULL;
  config->strategy = POPULATION_STRATEGY_ADAPTIVE_MUTATION;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--train") == 0) {
      config->replay_only = false;
    } else if (strcmp(argv[i], "--replay") == 0) {
      config->replay_only = true;
    } else if (strcmp(argv[i], "--no-render") == 0) {
      config->no_render = true;
    } else if (strcmp(argv[i], "--generations") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing generation count.\n");
        print_usage(argv[0]);
        return false;
      }

      config->generations = atoi(argv[i + 1]);
      if (config->generations <= 0) {
        fprintf(stderr, "Generation count must be greater than zero.\n");
        print_usage(argv[0]);
        return false;
      }
      i++;
    } else if (strcmp(argv[i], "--brain") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing brain path.\n");
        print_usage(argv[0]);
        return false;
      }

      config->brain_path = argv[i + 1];
      i++;
    } else if (strcmp(argv[i], "--csv") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing CSV path.\n");
        print_usage(argv[0]);
        return false;
      }

      config->csv_path = argv[i + 1];
      i++;
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

static void write_csv_header(FILE* csv_file) {
  if (csv_file != NULL) {
    fprintf(csv_file, "generation,strategy,best_fitness,average_fitness,score,steps,distance_reward,mutation_rate,mutation_strength\n");
  }
}

static void write_csv_row(FILE* csv_file, const Population* population, const Agent* best_agent, PopulationStrategy strategy) {
  if (csv_file != NULL) {
    fprintf(
        csv_file, "%d,%s,%.2f,%.2f,%d,%d,%d,%.3f,%.3f\n", population->generation, population_strategy_name(strategy),
        best_agent->fitness, population->average_fitness, best_agent->score, best_agent->steps, best_agent->distance_reward,
        population->mutation_rate, population->mutation_strength);
  }
}

static Agent* train_generations(Population* population, int count, const AppConfig* config, FILE* csv_file) {
  Agent* best_agent = NULL;

  for (int i = 0; i < count; i++) {
    population_evaluate(population);

    best_agent = &population->agents[population->best_agent_index];

    printf(
        "Strategy %s | generation %d | best %.2f | average %.2f | score %d | steps %d | distance %d | "
        "mutation %.3f %.3f\n",
        population_strategy_name(config->strategy), population->generation, best_agent->fitness, population->average_fitness,
        best_agent->score, best_agent->steps, best_agent->distance_reward, population->mutation_rate, population->mutation_strength);

    write_csv_row(csv_file, population, best_agent, config->strategy);

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

  population_evaluate(population);
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

static void update_simulation(Game* game, Agent** best_agent, Population* population, const AppConfig* config, FILE* csv_file) {
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
      *best_agent = train_generations(population, config->generations, config, csv_file);
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

  FILE* csv_file = NULL;
  if (config->csv_path != NULL) {
    csv_file = fopen(config->csv_path, "w");
    if (csv_file == NULL) {
      fprintf(stderr, "Failed to open CSV file: %s\n", config->csv_path);
      return 1;
    }
    write_csv_header(csv_file);
  }

  Population population;
  population_init(&population);

  if (brain_load(&population.agents[0].brain, config->brain_path)) {
    population.agents[0].fitness = 0.0f;
    population.agents[0].score = 0;
    population.agents[0].steps = 0;
    printf("Best brain loaded from %s\n", config->brain_path);
  }

  train_generations(&population, config->generations, config, csv_file);

  if (csv_file != NULL) {
    fclose(csv_file);
  }

  return 0;
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

  printf(
      "Mode: %s | strategy: %s | generations: %d | brain: %s\n", config.replay_only ? "replay" : "train",
      population_strategy_name(config.strategy), config.generations, config.brain_path);

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

  FILE* csv_file = NULL;
  if (config.csv_path != NULL) {
    csv_file = fopen(config.csv_path, "w");
    if (csv_file == NULL) {
      fprintf(stderr, "Failed to open CSV file: %s\n", config.csv_path);
      cleanup_sdl(window, renderer);
      return 1;
    }
    write_csv_header(csv_file);
  }

  Agent saved_agent = {0};
  Agent* best_agent = NULL;
  if (!setup_best_agent(&population, &saved_agent, &best_agent, &config)) {
    if (csv_file != NULL) {
      fclose(csv_file);
    }
    cleanup_sdl(window, renderer);
    return 1;
  }

  if (!config.replay_only) {
    best_agent = train_generations(&population, config.generations, &config, csv_file);
  }

  Uint32 last_update = SDL_GetTicks();
  const Uint32 update_delay = SNAKE_UPDATE_DELAY_MS;

  while (running) {
    handle_input(&running, &game);

    Uint32 now = SDL_GetTicks();

    if (now - last_update >= update_delay) {
      update_simulation(&game, &best_agent, &population, &config, csv_file);
      update_window_title(window, &game);
      last_update = now;
    }

    render_game(renderer, &game);
    SDL_Delay(FRAME_DELAY_MS);
  }

  if (csv_file != NULL) {
    fclose(csv_file);
  }

  cleanup_sdl(window, renderer);

  return 0;
}