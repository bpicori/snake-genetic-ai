#ifndef GENETIC_H
#define GENETIC_H

#include "agent.h"

#define POPULATION_SIZE 100
#define MAX_GAME_STEPS 500

typedef struct {
  Agent agents[POPULATION_SIZE];
  int generation;
  int best_agent_index;
  float best_fitness;
} Population;

void population_init(Population *population);
void population_evaluate(Population *population);

#endif