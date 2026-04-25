#ifndef GENETIC_H
#define GENETIC_H

#include <stdbool.h>

#include "brain.h"

#define POPULATION_SIZE 100
#define MAX_GAME_STEPS 30000
#define MUTATION_RATE 0.05f
#define MUTATION_STRENGTH 0.2f
#define MAX_STEPS_WITHOUT_FOOD 3000
#define ELITE_COUNT 5        // top 5 agents survive to the next generation
#define PARENT_POOL_SIZE 20  // 20 agents are selected to be parents for the next generation
#define EVALUATION_GAMES 3   // each agent is evaluated 3 times

#define MIN_MUTATION_RATE 0.01f
#define MAX_MUTATION_RATE 0.20f

#define MIN_MUTATION_STRENGTH 0.05f
#define MAX_MUTATION_STRENGTH 0.50f

#define CONSERVATIVE_MAX_MUTATION_RATE 0.10f
#define CONSERVATIVE_MAX_MUTATION_STRENGTH 0.30f

// if fitness does not improve for 20 generations, increase mutation
// if fitness improves, reduce mutation
#define STAGNATION_THRESHOLD 20

#define THREAD_COUNT 4  

typedef struct {
  Brain brain;
  float fitness;
  int score;
  int steps;
  int distance_reward;
} Agent;

typedef enum {
  POPULATION_STRATEGY_BEST_ONLY,
  POPULATION_STRATEGY_ELITE_PARENT_POOL,
  POPULATION_STRATEGY_CROSSOVER,
  POPULATION_STRATEGY_TOURNAMENT,
  POPULATION_STRATEGY_ADAPTIVE_MUTATION,
  POPULATION_STRATEGY_ADAPTIVE_CONSERVATIVE
} PopulationStrategy;

typedef struct {
  Agent agents[POPULATION_SIZE];
  int generation;
  int best_agent_index;
  float best_fitness;
  float average_fitness;

  float best_fitness_ever;
  int stagnant_generations;
  float mutation_rate;
  float mutation_strength;
} Population;

void population_init(Population* population);
void population_evaluate(Population* population);
void population_evaluate_parallel(Population* population);
void population_next_generation(Population* population, PopulationStrategy strategy);
const char* population_strategy_name(PopulationStrategy strategy);
bool population_strategy_from_name(const char* name, PopulationStrategy* strategy);

void population_next_generation_v1_best_only(Population* population);
void population_next_generation_v2_elite_parent_pool(Population* population);
void population_next_generation_v3_crossover(Population* population);
void population_next_generation_v4_tournament_selection(Population* population);
void population_next_generation_v5_adaptive_mutation(Population* population);
void population_next_generation_v6_adaptive_conservative(Population* population);

#endif