#include "genetic.h"
#include <stdlib.h>

static void sort_agents_by_fitness(Agent agents[], int count) {
  for (int i = 0; i < count - 1; i++) {
    for (int j = 0; j < count - i - 1; j++) {
      if (agents[j].fitness < agents[j + 1].fitness) {
        Agent temp = agents[j];
        agents[j] = agents[j + 1];
        agents[j + 1] = temp;
      }
    }
  }
}

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

/*
 * Build the next generation from the current evaluated population.
 *
 * First, sort agents by fitness so the best agents are at the front.
 *
 * Example with POPULATION_SIZE = 100, ELITE_COUNT = 5,
 * and PARENT_POOL_SIZE = 20:
 *
 *   agents 0-4:
 *     copied unchanged from the top 5 agents
 *
 *   agents 5-99:
 *     each one chooses a random parent from agents 0-19,
 *     copies that parent's brain, then mutates the copy
 *
 * The top 20 agents are a parent pool. They are not all copied directly;
 * they are candidates for producing the rest of the population.
 */
void population_next_generation_v2_elite_parent_pool(Population *population) {
  sort_agents_by_fitness(population->agents, POPULATION_SIZE);

  Agent next_agents[POPULATION_SIZE];

  for (int i = 0; i < ELITE_COUNT; i++) {
    next_agents[i] = population->agents[i];
  }

  for (int i = ELITE_COUNT; i < PARENT_POOL_SIZE; i++) {
    int parent_index = rand() % PARENT_POOL_SIZE;
    Agent parent = population->agents[parent_index];

    next_agents[i] = parent;

    brain_mutate(&next_agents[i].brain, MUTATION_RATE, MUTATION_STRENGTH);
    next_agents[i].fitness = 0.0f;
    next_agents[i].score = 0;
    next_agents[i].steps = 0;
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    population->agents[i] = next_agents[i];
  }

  population->best_agent_index = 0;
  population->best_fitness = population->agents[0].fitness;
  population->generation++;
}

/*
 * Build the next generation from the current evaluated population.
 *
 * The best agent is copied unchanged to the first position in the next
 * generation. The rest of the population is created by copying the best agent's
 * brain and mutating it.
 */
void population_next_generation_v1_best_only(Population *population) {
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

void population_next_generation(Population *population) {
  population_next_generation_v2_elite_parent_pool(population);
}