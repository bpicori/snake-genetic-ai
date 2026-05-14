#include "genetic.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "brain.h"

typedef struct {
  Population* population;
  int start_index;
  int end_index;
} EvaluationJob;

/* Scratch buffers for next-generation assembly (single-threaded path only). */
static Agent g_next_generation_scratch[POPULATION_SIZE];
static bool g_next_generation_scratch_ready = false;

static void ensure_next_generation_scratch(void) {
  if (g_next_generation_scratch_ready) {
    return;
  }
  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_init(&g_next_generation_scratch[i]);
  }
  g_next_generation_scratch_ready = true;
}

static void destroy_next_generation_scratch(void) {
  if (!g_next_generation_scratch_ready) {
    return;
  }
  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_destroy(&g_next_generation_scratch[i]);
  }
  g_next_generation_scratch_ready = false;
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

static void agent_set_result(Agent* agent, const Game* game) {
  agent->score = game->score;
  agent->steps = game->steps;
  agent->distance_reward = game->distance_reward;

  /*
   * Fitness is the score used by the genetic algorithm to rank agents.
   * Eating food is much more valuable than only surviving. Squaring the score
   * makes higher-scoring snakes stand out strongly during selection.
   */
  agent->fitness = (float)((game->score * game->score * 1000) + game->steps + (game->distance_reward * 5));
}

/*
 Clamps a float value between a minimum and maximum value.
 If the value is less than the minimum, return the minimum.
 If the value is greater than the maximum, return the maximum.
 Otherwise, return the value.

 Example:
   value = 1.5
   min = 0.0
   max = 1.0
   return 1.0
*/
static float clamp_float(float value, float min, float max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

// sort agents by fitness in descending order
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

/*
 * Selects one parent using tournament selection.
 *
 * Instead of choosing only from the top agents, this randomly samples a small
 * group from the whole population and returns the best agent from that group.
 *
 * Example with tournament_size = 3:
 *   pick agent 14 with fitness 800
 *   pick agent 63 with fitness 2400
 *   pick agent 7  with fitness 1200
 *
 *   agent 63 wins the tournament and becomes the selected parent.
 *
 * Larger tournaments favor stronger agents more aggressively.
 * Smaller tournaments keep more randomness and diversity.
 */
static int tournament_parent_index(const Agent agents[], int tournament_size) {
  int best_index = rand() % POPULATION_SIZE;

  for (int i = 1; i < tournament_size; i++) {
    int candidate_index = rand() % POPULATION_SIZE;

    if (agents[candidate_index].fitness > agents[best_index].fitness) {
      best_index = candidate_index;
    }
  }

  return best_index;
}

/*
 * Adjust mutation automatically based on training progress.
 *
 * If the current generation beats the best fitness ever seen, training is
 * improving. In that case, reduce mutation slightly so children stay closer to
 * the good solution and can fine-tune it.
 *
 * If the current generation does not improve, count it as stagnant. After
 * STAGNATION_THRESHOLD stagnant generations, increase mutation so the next
 * generations explore more different brains.
 *
 * Example:
 *   mutation_rate = 0.05
 *   mutation_strength = 0.20
 *
 *   New best fitness found:
 *     mutation_rate = 0.05 * 0.95 = 0.0475
 *     mutation_strength = 0.20 * 0.95 = 0.19
 *
 *   No improvement for STAGNATION_THRESHOLD generations:
 *     mutation_rate = 0.0475 * 1.25 = 0.059
 *     mutation_strength = 0.19 * 1.25 = 0.2375
 *
 * clamp_float keeps mutation inside safe min/max limits, so it never becomes
 * too small to explore or too large to preserve useful behavior.
 */
static void update_adaptive_mutation(Population* population, float max_rate, float max_strength, float stagnation_multiplier) {
  // found a new best fitness
  if (population->best_fitness > population->best_fitness_ever) {
    population->best_fitness_ever = population->best_fitness;
    population->stagnant_generations = 0;
    // reduce mutation rate and strength by 5% (0.95 means reduce by 5%)
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
    agent_set_result(agent, &game);
    total_fitness += agent->fitness;
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

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_init(&population->agents[i]);
    agent_randomize(&population->agents[i]);
  }
}

void population_destroy(Population* population) {
  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_destroy(&population->agents[i]);
  }
  destroy_next_generation_scratch();
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
void population_evaluate(Population* population) {
  population->best_agent_index = 0;
  population->best_fitness = 0.0f;
  population->average_fitness = 0.0f;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    evaluate_agent(&population->agents[i]);
    population->average_fitness += population->agents[i].fitness;

    // update the best agent
    if (population->agents[i].fitness > population->best_fitness) {
      population->best_fitness = population->agents[i].fitness;
      population->best_agent_index = i;
    }
  }

  population->average_fitness /= POPULATION_SIZE;
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
void population_next_generation_v2_elite_parent_pool(Population* population) {
  sort_agents_by_fitness(population->agents, POPULATION_SIZE);

  ensure_next_generation_scratch();
  Agent* next_agents = g_next_generation_scratch;

  for (int i = 0; i < ELITE_COUNT; i++) {
    agent_copy(&next_agents[i], &population->agents[i]);
  }

  for (int i = ELITE_COUNT; i < POPULATION_SIZE; i++) {
    int parent_index = rand() % PARENT_POOL_SIZE;

    agent_copy(&next_agents[i], &population->agents[parent_index]);

    brain_mutate(&next_agents[i].brain, MUTATION_RATE, MUTATION_STRENGTH);
    next_agents[i].fitness = 0.0f;
    next_agents[i].score = 0;
    next_agents[i].steps = 0;
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_copy(&population->agents[i], &next_agents[i]);
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
void population_next_generation_v1_best_only(Population* population) {
  const Agent* best_agent = &population->agents[population->best_agent_index];

  agent_copy(&population->agents[0], best_agent);

  for (int i = 1; i < POPULATION_SIZE; i++) {
    brain_copy(&population->agents[i].brain, &best_agent->brain);
    brain_mutate(&population->agents[i].brain, MUTATION_RATE, MUTATION_STRENGTH);

    population->agents[i].fitness = 0.0f;
    population->agents[i].score = 0;
    population->agents[i].steps = 0;
  }

  population->generation++;
}

/*
 * Build the next generation using elitism, parent selection, crossover,
 * and mutation.
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
 *     each one chooses two random parents from agents 0-19,
 *     mixes their brains using crossover, then mutates the child brain
 *
 * Crossover means each brain weight/bias is copied from either parent A or
 * parent B. Mutation then makes small random changes to the mixed brain.
 */
void population_next_generation_v3_crossover(Population* population) {
  sort_agents_by_fitness(population->agents, POPULATION_SIZE);

  ensure_next_generation_scratch();
  Agent* next_agents = g_next_generation_scratch;

  for (int i = 0; i < ELITE_COUNT; i++) {
    agent_copy(&next_agents[i], &population->agents[i]);
  }

  for (int i = ELITE_COUNT; i < POPULATION_SIZE; i++) {
    int parent_a_index = rand() % PARENT_POOL_SIZE;
    int parent_b_index = rand() % PARENT_POOL_SIZE;

    next_agents[i].fitness = 0.0f;
    next_agents[i].score = 0;
    next_agents[i].steps = 0;

    brain_crossover(&next_agents[i].brain, &population->agents[parent_a_index].brain, &population->agents[parent_b_index].brain);

    brain_mutate(&next_agents[i].brain, MUTATION_RATE, MUTATION_STRENGTH);
  }

  population->best_agent_index = 0;
  population->best_fitness = population->agents[0].fitness;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_copy(&population->agents[i], &next_agents[i]);
  }

  population->generation++;
}

void population_next_generation_v4_tournament_selection(Population* population) {
  sort_agents_by_fitness(population->agents, POPULATION_SIZE);

  ensure_next_generation_scratch();
  Agent* next_agents = g_next_generation_scratch;

  for (int i = 0; i < ELITE_COUNT; i++) {
    agent_copy(&next_agents[i], &population->agents[i]);
  }

  for (int i = ELITE_COUNT; i < POPULATION_SIZE; i++) {
    int parent_a_index = tournament_parent_index(population->agents, 3);
    int parent_b_index = tournament_parent_index(population->agents, 3);

    next_agents[i].fitness = 0.0f;
    next_agents[i].score = 0;
    next_agents[i].steps = 0;
    next_agents[i].distance_reward = 0;

    brain_crossover(&next_agents[i].brain, &population->agents[parent_a_index].brain, &population->agents[parent_b_index].brain);
    brain_mutate(&next_agents[i].brain, MUTATION_RATE, MUTATION_STRENGTH);
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_copy(&population->agents[i], &next_agents[i]);
  }

  population->best_agent_index = 0;
  population->best_fitness = population->agents[0].fitness;
  population->generation++;
}

void population_next_generation_v5_adaptive_mutation(Population* population) {
  update_adaptive_mutation(population, MAX_MUTATION_RATE, MAX_MUTATION_STRENGTH, 1.25f);

  sort_agents_by_fitness(population->agents, POPULATION_SIZE);

  ensure_next_generation_scratch();
  Agent* next_agents = g_next_generation_scratch;

  for (int i = 0; i < ELITE_COUNT; i++) {
    agent_copy(&next_agents[i], &population->agents[i]);
  }

  for (int i = ELITE_COUNT; i < POPULATION_SIZE; i++) {
    int parent_a_index = rand() % PARENT_POOL_SIZE;
    int parent_b_index = rand() % PARENT_POOL_SIZE;

    next_agents[i].fitness = 0.0f;
    next_agents[i].score = 0;
    next_agents[i].steps = 0;
    next_agents[i].distance_reward = 0;

    brain_crossover(&next_agents[i].brain, &population->agents[parent_a_index].brain, &population->agents[parent_b_index].brain);

    brain_mutate(&next_agents[i].brain, population->mutation_rate, population->mutation_strength);
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_copy(&population->agents[i], &next_agents[i]);
  }

  population->best_agent_index = 0;
  population->best_fitness = population->agents[0].fitness;
  population->generation++;
}

void population_next_generation_v6_adaptive_conservative(Population* population) {
  update_adaptive_mutation(population, CONSERVATIVE_MAX_MUTATION_RATE, CONSERVATIVE_MAX_MUTATION_STRENGTH, 1.10f);

  sort_agents_by_fitness(population->agents, POPULATION_SIZE);

  ensure_next_generation_scratch();
  Agent* next_agents = g_next_generation_scratch;

  for (int i = 0; i < ELITE_COUNT; i++) {
    agent_copy(&next_agents[i], &population->agents[i]);
  }

  for (int i = ELITE_COUNT; i < POPULATION_SIZE; i++) {
    int parent_a_index = rand() % PARENT_POOL_SIZE;
    int parent_b_index = rand() % PARENT_POOL_SIZE;

    next_agents[i].fitness = 0.0f;
    next_agents[i].score = 0;
    next_agents[i].steps = 0;
    next_agents[i].distance_reward = 0;

    brain_crossover(&next_agents[i].brain, &population->agents[parent_a_index].brain, &population->agents[parent_b_index].brain);

    brain_mutate(&next_agents[i].brain, population->mutation_rate, population->mutation_strength);
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    agent_copy(&population->agents[i], &next_agents[i]);
  }

  population->best_agent_index = 0;
  population->best_fitness = population->agents[0].fitness;
  population->generation++;
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
  switch (strategy) {
    case POPULATION_STRATEGY_BEST_ONLY:
      return "v1";
    case POPULATION_STRATEGY_ELITE_PARENT_POOL:
      return "v2";
    case POPULATION_STRATEGY_CROSSOVER:
      return "v3";
    case POPULATION_STRATEGY_TOURNAMENT:
      return "v4";
    case POPULATION_STRATEGY_ADAPTIVE_MUTATION:
      return "adaptive";
    case POPULATION_STRATEGY_ADAPTIVE_CONSERVATIVE:
      return "adaptive-conservative";
  }

  return "adaptive";
}

bool population_strategy_from_name(const char* name, PopulationStrategy* strategy) {
  if (strcmp(name, "v1") == 0 || strcmp(name, "best") == 0) {
    *strategy = POPULATION_STRATEGY_BEST_ONLY;
    return true;
  }

  if (strcmp(name, "v2") == 0 || strcmp(name, "elite") == 0) {
    *strategy = POPULATION_STRATEGY_ELITE_PARENT_POOL;
    return true;
  }

  if (strcmp(name, "v3") == 0 || strcmp(name, "crossover") == 0) {
    *strategy = POPULATION_STRATEGY_CROSSOVER;
    return true;
  }

  if (strcmp(name, "v4") == 0 || strcmp(name, "tournament") == 0) {
    *strategy = POPULATION_STRATEGY_TOURNAMENT;
    return true;
  }

  if (strcmp(name, "v5") == 0 || strcmp(name, "adaptive") == 0) {
    *strategy = POPULATION_STRATEGY_ADAPTIVE_MUTATION;
    return true;
  }

  if (strcmp(name, "v6") == 0 || strcmp(name, "adaptive-conservative") == 0 || strcmp(name, "conservative") == 0) {
    *strategy = POPULATION_STRATEGY_ADAPTIVE_CONSERVATIVE;
    return true;
  }

  return false;
}
