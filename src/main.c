#include "SDL_keycode.h"
#include "SDL_render.h"
#include "SDL_video.h"
#include <SDL.h>
#include <stdbool.h>
#include <stdio.h>

#include "agent.h"
#include "game.h"
#include "genetic.h"

#define CELL_SIZE 30

#define GRID_WIDTH 20
#define GRID_HEIGHT 20

#define WINDOW_WIDTH (GRID_WIDTH * CELL_SIZE)
#define WINDOW_HEIGHT (GRID_HEIGHT * CELL_SIZE)

#define RENDER_FPS 60
#define FRAME_DELAY_MS (1000 / RENDER_FPS)

#define SNAKE_MOVES_PER_SECOND 30
#define GENERATIONS_PER_REPLAY 2000
#define SNAKE_UPDATE_DELAY_MS (1000 / SNAKE_MOVES_PER_SECOND)

#define BEST_BRAIN_PATH "out/best.brain"

bool ai_enabled = true;
static float best_fitness_ever = 0.0f;

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

static Agent *train_generations(Population *population, int count) {
  Agent *best_agent = NULL;

  for (int i = 0; i < count; i++) {
    population_evaluate(population);

    best_agent = &population->agents[population->best_agent_index];

    printf("Generation %d | fitness %.2f | score %d | steps %d | distance %d\n",
           population->generation, best_agent->fitness, best_agent->score,
           best_agent->steps, best_agent->distance_reward);

    if (best_agent->fitness > best_fitness_ever) {
      best_fitness_ever = best_agent->fitness;
      if (brain_save(&best_agent->brain, BEST_BRAIN_PATH)) {
        printf("Best brain saved to %s\n", BEST_BRAIN_PATH);
      } else {
        printf("Failed to save best brain to %s\n", BEST_BRAIN_PATH);
      }
    }

    population_next_generation(population);
  }

  population_evaluate(population);
  best_agent = &population->agents[population->best_agent_index];

  return best_agent;
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

  Population population;
  population_init(&population);

  if (brain_load(&population.agents[0].brain, BEST_BRAIN_PATH)) {
    population.agents[0].fitness = 0.0f;
    population.agents[0].score = 0;
    population.agents[0].steps = 0;

    printf("Best brain loaded from %s\n", BEST_BRAIN_PATH);
  }

  Agent *best_agent = train_generations(&population, GENERATIONS_PER_REPLAY);

  Uint32 last_update = SDL_GetTicks();
  const Uint32 update_delay = SNAKE_UPDATE_DELAY_MS;

  while (running) {
    handle_input(&running, &game);

    Uint32 now = SDL_GetTicks();

    if (now - last_update >= update_delay) {
      if (ai_enabled) {
        Direction direction = agent_choose_direction(best_agent, &game);
        game_set_direction(&game, direction);
      }

      game_update(&game);

      if (!game.alive || game.steps >= MAX_GAME_STEPS ||
          game.steps_since_food >= MAX_STEPS_WITHOUT_FOOD) {
        population_next_generation(&population);
        best_agent = train_generations(&population, GENERATIONS_PER_REPLAY);
        game_init(&game);
      }

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