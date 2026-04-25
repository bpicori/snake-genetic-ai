#ifndef GENETIC_H
#define GENETIC_H

#include "agent.h"

#define POPULATION_SIZE 100
#define MAX_GAME_STEPS 500
#define MUTATION_RATE 0.05f
#define MUTATION_STRENGTH 0.2f
#define MAX_STEPS_WITHOUT_FOOD 100
#define ELITE_COUNT 5 // top 5 agents survive to the next generation
#define PARENT_POOL_SIZE                                                       \
  20 // 20 agents are selected to be parents for the next generation
#define EVALUATION_GAMES 3 // each agent is evaluated 3 times

#define MIN_MUTATION_RATE 0.01f
#define MAX_MUTATION_RATE 0.20f

#define MIN_MUTATION_STRENGTH 0.05f
#define MAX_MUTATION_STRENGTH 0.50f

// if fitness does not improve for 20 generations, increase mutation
// if fitness improves, reduce mutation
#define STAGNATION_THRESHOLD 20

typedef struct {
  Agent agents[POPULATION_SIZE];
  int generation;
  int best_agent_index;
  float best_fitness;

  float best_fitness_ever;
  int stagnant_generations;
  float mutation_rate;
  float mutation_strength;
} Population;

void population_init(Population *population);
void population_evaluate(Population *population);
void population_next_generation(Population *population);

void population_next_generation_v1_best_only(Population *population);
void population_next_generation_v2_elite_parent_pool(Population *population);
void population_next_generation_v3_crossover(Population *population);
void population_next_generation_v4_tournament_selection(Population *population);
void population_next_generation_v5_adaptive_mutation(Population *population);

#endif