#include "genetic.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "brain.h"
#include "game.h"
#include "rng.h"

typedef struct {
  Population* population;
  int start_index;
  int end_index;
} EvaluationJob;

typedef enum { PARENT_PICK_POOL, PARENT_PICK_TOURNAMENT } ParentPickMode;

static const struct {
  PopulationStrategy strategy;
  const char* primary;
  const char* alt1;
  const char* alt2;
} k_strategy_names[] = {
    {POPULATION_STRATEGY_BEST_ONLY, "v1", "best", NULL},
    {POPULATION_STRATEGY_ELITE_PARENT_POOL, "v2", "elite", NULL},
    {POPULATION_STRATEGY_CROSSOVER, "v3", "crossover", NULL},
    {POPULATION_STRATEGY_TOURNAMENT, "v4", "tournament", NULL},
    {POPULATION_STRATEGY_ADAPTIVE_MUTATION, "adaptive", "v5", NULL},
    {POPULATION_STRATEGY_ADAPTIVE_CONSERVATIVE, "adaptive-conservative", "v6", "conservative"},
};

static float fitness_from_game(const Game* game) {
  return (float)((game->score * game->score * 1000) + game->steps + (game->distance_reward * 5));
}

void agent_init(Agent* agent) {
  brain_init(&agent->brain);
  agent->fitness = 0.0f;
  agent->score = 0;
  agent->steps = 0;
  agent->distance_reward = 0;
}

void agent_destroy(Agent* agent) {
  brain_destroy(&agent->brain);
  agent->fitness = 0.0f;
  agent->score = 0;
  agent->steps = 0;
  agent->distance_reward = 0;
}

void agent_copy(Agent* dest, const Agent* src) {
  brain_copy(&dest->brain, &src->brain);
  dest->fitness = src->fitness;
  dest->score = src->score;
  dest->steps = src->steps;
  dest->distance_reward = src->distance_reward;
}

static void agent_randomize(Agent* agent) {
  brain_randomize(&agent->brain);
  agent->fitness = 0.0f;
  agent->score = 0;
  agent->steps = 0;
  agent->distance_reward = 0;
}

static void agent_reset_results(Agent* agent) {
  agent->fitness = 0.0f;
  agent->score = 0;
  agent->steps = 0;
  agent->distance_reward = 0;
}

static float clamp_float(float value, float min, float max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

static int compare_agent_fitness_desc(const void* a, const void* b) {
  const Agent* aa = (const Agent*)a;
  const Agent* bb = (const Agent*)b;
  if (aa->fitness < bb->fitness) {
    return 1;
  }
  if (aa->fitness > bb->fitness) {
    return -1;
  }
  return 0;
}

static int tournament_parent_index(const Agent agents[], int tournament_size) {
  int best_index = rng_int(POPULATION_SIZE);

  for (int i = 1; i < tournament_size; i++) {
    int candidate_index = rng_int(POPULATION_SIZE);

    if (agents[candidate_index].fitness > agents[best_index].fitness) {
      best_index = candidate_index;
    }
  }

  return best_index;
}

static void update_adaptive_mutation(Population* population, float max_rate, float max_strength, float stagnation_multiplier) {
  if (population->best_fitness > population->best_fitness_ever) {
    population->best_fitness_ever = population->best_fitness;
    population->stagnant_generations = 0;
    population->mutation_rate = clamp_float(population->mutation_rate * 0.95f, MIN_MUTATION_RATE, max_rate);
    population->mutation_strength = clamp_float(population->mutation_strength * 0.95f, MIN_MUTATION_STRENGTH, max_strength);
  } else {
    population->stagnant_generations++;
    if (population->stagnant_generations >= STAGNATION_THRESHOLD) {
      population->mutation_rate = clamp_float(population->mutation_rate * stagnation_multiplier, MIN_MUTATION_RATE, max_rate);
      population->mutation_strength =
          clamp_float(population->mutation_strength * stagnation_multiplier, MIN_MUTATION_STRENGTH, max_strength);
      population->stagnant_generations = 0;
    }
  }
}

static void population_compute_stats(Population* population) {
  population->best_agent_index = 0;
  population->best_fitness = 0.0f;
  population->average_fitness = 0.0f;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    population->average_fitness += population->agents[i].fitness;

    if (population->agents[i].fitness > population->best_fitness) {
      population->best_fitness = population->agents[i].fitness;
      population->best_agent_index = i;
    }
  }

  population->average_fitness /= POPULATION_SIZE;
}

static void population_swap_generation_buffers(Population* pop) {
  Agent* tmp = pop->agents;
  pop->agents = pop->next_generation;
  pop->next_generation = tmp;
}

static void evaluate_agent(Agent* agent) {
  float total_fitness = 0.0f;
  int best_score = 0;
  int best_steps = 0;
  int total_distance_reward = 0;
  for (int run = 0; run < EVALUATION_GAMES; run++) {
    Game game;
    game_init(&game);
    while (game.alive && game.steps < MAX_GAME_STEPS && game.steps_since_food < MAX_STEPS_WITHOUT_FOOD) {
      Direction direction = brain_choose_direction(&agent->brain, &game);
      game_set_direction(&game, direction);
      game_update(&game);
    }
    float run_fitness = fitness_from_game(&game);
    total_fitness += run_fitness;
    total_distance_reward += game.distance_reward;
    if (game.score > best_score || (game.score == best_score && game.steps > best_steps)) {
      best_score = game.score;
      best_steps = game.steps;
    }
  }
  agent->fitness = total_fitness / EVALUATION_GAMES;
  agent->score = best_score;
  agent->steps = best_steps;
  agent->distance_reward = total_distance_reward / EVALUATION_GAMES;
}

static void* evaluate_agents_worker(void* arg) {
  EvaluationJob* job = arg;
  for (int i = job->start_index; i < job->end_index; i++) {
    evaluate_agent(&job->population->agents[i]);
  }
  return NULL;
}

void population_init(Population* population) {
  population->generation = 0;
  population->best_agent_index = 0;
  population->best_fitness = 0.0f;
  population->average_fitness = 0.0f;
  population->best_fitness_ever = 0.0f;
  population->stagnant_generations = 0;
  population->mutation_rate = MUTATION_RATE;
  population->mutation_strength = MUTATION_STRENGTH;

  population->agents = population->pool_a;
  population->next_generation = population->pool_b;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_init(&population->agents[i]);
    agent_randomize(&population->agents[i]);
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_init(&population->next_generation[i]);
  }
}

void population_destroy(Population* population) {
  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_destroy(&population->pool_a[i]);
    agent_destroy(&population->pool_b[i]);
  }
}

void population_evaluate(Population* population) {
  for (int i = 0; i < POPULATION_SIZE; i++) {
    evaluate_agent(&population->agents[i]);
  }
  population_compute_stats(population);
}

void population_evaluate_parallel(Population* population) {
  pthread_t threads[THREAD_COUNT];
  EvaluationJob jobs[THREAD_COUNT];

  int agents_per_thread = POPULATION_SIZE / THREAD_COUNT;
  int remainder = POPULATION_SIZE % THREAD_COUNT;
  int start_index = 0;

  for (int i = 0; i < THREAD_COUNT; i++) {
    int count = agents_per_thread;

    if (i < remainder) {
      count++;
    }

    jobs[i].population = population;
    jobs[i].start_index = start_index;
    jobs[i].end_index = start_index + count;

    pthread_create(&threads[i], NULL, evaluate_agents_worker, &jobs[i]);

    start_index += count;
  }

  for (int i = 0; i < THREAD_COUNT; i++) {
    pthread_join(threads[i], NULL);
  }

  population_compute_stats(population);
}

static void build_next_generation_elite(
    Population* population, ParentPickMode pick_mode, bool use_crossover, float mutation_rate, float mutation_strength) {
  qsort(population->agents, POPULATION_SIZE, sizeof(Agent), compare_agent_fitness_desc);

  Agent* cur = population->agents;
  Agent* next = population->next_generation;

  for (int i = 0; i < ELITE_COUNT; i++) {
    agent_copy(&next[i], &cur[i]);
  }

  for (int i = ELITE_COUNT; i < POPULATION_SIZE; i++) {
    if (use_crossover) {
      agent_reset_results(&next[i]);
      int parent_a_index;
      int parent_b_index;
      if (pick_mode == PARENT_PICK_POOL) {
        parent_a_index = rng_int(PARENT_POOL_SIZE);
        parent_b_index = rng_int(PARENT_POOL_SIZE);
      } else {
        parent_a_index = tournament_parent_index(cur, 3);
        parent_b_index = tournament_parent_index(cur, 3);
      }
      brain_crossover(&next[i].brain, &cur[parent_a_index].brain, &cur[parent_b_index].brain);
    } else {
      int parent_index = rng_int(PARENT_POOL_SIZE);
      agent_copy(&next[i], &cur[parent_index]);
    }
    brain_mutate(&next[i].brain, mutation_rate, mutation_strength);
    agent_reset_results(&next[i]);
  }

  population_swap_generation_buffers(population);

  population->best_agent_index = 0;
  population->best_fitness = population->agents[0].fitness;
  population->generation++;
}

void population_next_generation_v1_best_only(Population* population) {
  const Agent* best_agent = &population->agents[population->best_agent_index];
  Agent* next = population->next_generation;

  agent_copy(&next[0], best_agent);

  for (int i = 1; i < POPULATION_SIZE; i++) {
    brain_copy(&next[i].brain, &best_agent->brain);
    brain_mutate(&next[i].brain, population->mutation_rate, population->mutation_strength);
    agent_reset_results(&next[i]);
  }

  population_swap_generation_buffers(population);
  population->best_agent_index = 0;
  population->best_fitness = population->agents[0].fitness;
  population->generation++;
}

void population_next_generation_v2_elite_parent_pool(Population* population) {
  build_next_generation_elite(population, PARENT_PICK_POOL, false, population->mutation_rate, population->mutation_strength);
}

void population_next_generation_v3_crossover(Population* population) {
  build_next_generation_elite(population, PARENT_PICK_POOL, true, population->mutation_rate, population->mutation_strength);
}

void population_next_generation_v4_tournament_selection(Population* population) {
  build_next_generation_elite(population, PARENT_PICK_TOURNAMENT, true, population->mutation_rate, population->mutation_strength);
}

void population_next_generation_v5_adaptive_mutation(Population* population) {
  update_adaptive_mutation(population, MAX_MUTATION_RATE, MAX_MUTATION_STRENGTH, 1.25f);
  build_next_generation_elite(population, PARENT_PICK_POOL, true, population->mutation_rate, population->mutation_strength);
}

void population_next_generation_v6_adaptive_conservative(Population* population) {
  update_adaptive_mutation(population, CONSERVATIVE_MAX_MUTATION_RATE, CONSERVATIVE_MAX_MUTATION_STRENGTH, 1.10f);
  build_next_generation_elite(population, PARENT_PICK_POOL, true, population->mutation_rate, population->mutation_strength);
}

void population_next_generation(Population* population, PopulationStrategy strategy) {
  switch (strategy) {
    case POPULATION_STRATEGY_BEST_ONLY:
      population_next_generation_v1_best_only(population);
      break;
    case POPULATION_STRATEGY_ELITE_PARENT_POOL:
      population_next_generation_v2_elite_parent_pool(population);
      break;
    case POPULATION_STRATEGY_CROSSOVER:
      population_next_generation_v3_crossover(population);
      break;
    case POPULATION_STRATEGY_TOURNAMENT:
      population_next_generation_v4_tournament_selection(population);
      break;
    case POPULATION_STRATEGY_ADAPTIVE_MUTATION:
      population_next_generation_v5_adaptive_mutation(population);
      break;
    case POPULATION_STRATEGY_ADAPTIVE_CONSERVATIVE:
      population_next_generation_v6_adaptive_conservative(population);
      break;
  }
}

const char* population_strategy_name(PopulationStrategy strategy) {
  for (size_t i = 0; i < sizeof(k_strategy_names) / sizeof(k_strategy_names[0]); i++) {
    if (k_strategy_names[i].strategy == strategy) {
      return k_strategy_names[i].primary;
    }
  }
  return "adaptive";
}

bool population_strategy_from_name(const char* name, PopulationStrategy* strategy) {
  for (size_t i = 0; i < sizeof(k_strategy_names) / sizeof(k_strategy_names[0]); i++) {
    if (strcmp(name, k_strategy_names[i].primary) == 0 ||
        (k_strategy_names[i].alt1 && strcmp(name, k_strategy_names[i].alt1) == 0) ||
        (k_strategy_names[i].alt2 && strcmp(name, k_strategy_names[i].alt2) == 0)) {
      *strategy = k_strategy_names[i].strategy;
      return true;
    }
  }
  return false;
}
