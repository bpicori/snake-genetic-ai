#ifndef GENETIC_H
#define GENETIC_H

#include "agent.h"

#define POPULATION_SIZE 100
#define MAX_GAME_STEPS 500
#define MUTATION_RATE 0.05f
#define MUTATION_STRENGTH 0.2f
#define MAX_STEPS_WITHOUT_FOOD 100

typedef struct {
  Agent agents[POPULATION_SIZE];
  int generation;
  int best_agent_index;
  float best_fitness;
} Population;

void population_init(Population *population);
void population_evaluate(Population *population);
void population_next_generation(Population *population);

#endif