#include "training.h"

#include <stdio.h>

#include "brain.h"

bool training_session_init(TrainingSession* session, const AppConfig* config) {
  session->saved_agent = (Agent){0};
  session->best_agent = NULL;
  session->best_fitness_ever = 0.0f;

  population_init(&session->population);

  if (config->replay_only) {
    if (!brain_load(&session->saved_agent.brain, config->brain_path)) {
      printf("No saved brain found at %s\n", config->brain_path);
      return false;
    }

    printf("Replay mode: loaded saved brain from %s\n", config->brain_path);
    session->best_agent = &session->saved_agent;
    return true;
  }

  if (brain_load(&session->population.agents[0].brain, config->brain_path)) {
    session->population.agents[0].fitness = 0.0f;
    session->population.agents[0].score = 0;
    session->population.agents[0].steps = 0;

    printf("Best brain loaded from %s\n", config->brain_path);
  }

  return true;
}

Agent* training_session_best_agent(TrainingSession* session) { return session->best_agent; }

Agent* training_session_train_generations(TrainingSession* session, int count, const AppConfig* config) {
  Agent* best_agent = NULL;

  for (int i = 0; i < count; i++) {
    population_evaluate_parallel(&session->population);

    best_agent = &session->population.agents[session->population.best_agent_index];

    printf(
        "Strategy %s | generation %d | best %.2f | average %.2f | score %d | steps %d | distance %d | "
        "mutation %.3f %.3f\n",
        population_strategy_name(config->strategy), session->population.generation, best_agent->fitness, session->population.average_fitness,
        best_agent->score, best_agent->steps, best_agent->distance_reward, session->population.mutation_rate,
        session->population.mutation_strength);

    if (best_agent->fitness > session->best_fitness_ever) {
      session->best_fitness_ever = best_agent->fitness;
      if (brain_save(&best_agent->brain, config->brain_path)) {
        printf("Best brain saved to %s\n", config->brain_path);
      } else {
        printf("Failed to save best brain to %s\n", config->brain_path);
      }
    }

    population_next_generation(&session->population, config->strategy);
  }

  population_evaluate_parallel(&session->population);
  session->best_agent = &session->population.agents[session->population.best_agent_index];

  return session->best_agent;
}

int training_run_headless(const AppConfig* config) {
  if (config->replay_only) {
    fprintf(stderr, "--no-render is for training; use replay without --no-render.\n");
    return 1;
  }

  TrainingSession session;
  if (!training_session_init(&session, config)) {
    return 1;
  }

  training_session_train_generations(&session, config->generations, config);

  return 0;
}
