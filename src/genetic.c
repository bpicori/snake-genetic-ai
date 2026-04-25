#include "genetic.h"

void population_init(Population *population) {
  population->generation = 0;
  population->best_agent_index = 0;
  population->best_fitness = 0.0f;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_randomize(&population->agents[i]);
  }
}

/*
 * Evaluate every agent in the population by letting it play one full game.
 *
 * Each agent starts from a fresh Game state. The agent repeatedly chooses a
 * direction using its brain, the game advances one step, and the loop stops
 * when the snake dies or reaches MAX_GAME_STEPS.
 *
 * After the game ends, we convert the result into a fitness score:
 *   score = food eaten
 *   steps = moves survived
 *
 * Example:
 *   Agent 0: score = 0, steps = 45  -> fitness = 45
 *   Agent 1: score = 2, steps = 130 -> fitness = 2130
 *   Agent 2: score = 1, steps = 220 -> fitness = 1220
 *
 * Agent 1 becomes the best agent because it has the highest fitness.
 *
 * This function does not create the next generation yet. It only measures how
 * good the current generation is.
 */
void population_evaluate(Population *population) {
  population->best_agent_index = 0;
  population->best_fitness = 0.0f;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    Game game;
    game_init(&game);

    while (game.alive && game.steps < MAX_GAME_STEPS &&
           game.steps_since_food < MAX_STEPS_WITHOUT_FOOD) {
      Direction direction =
          agent_choose_direction(&population->agents[i], &game);
      game_set_direction(&game, direction);
      game_update(&game);
    }

    agent_set_result(&population->agents[i], &game);

    if (population->agents[i].fitness > population->best_fitness) {
      population->best_fitness = population->agents[i].fitness;
      population->best_agent_index = i;
    }
  }
}

void population_next_generation(Population *population) {
  Agent best_agent = population->agents[population->best_agent_index];

  population->agents[0] = best_agent;

  for (int i = 1; i < POPULATION_SIZE; i++) {
    brain_copy(&population->agents[i].brain, &best_agent.brain);
    brain_mutate(&population->agents[i].brain, MUTATION_RATE,
                 MUTATION_STRENGTH);

    population->agents[i].fitness = 0.0f;
    population->agents[i].score = 0;
    population->agents[i].steps = 0;
  }

  population->generation++;
}