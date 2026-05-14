#ifndef TRAINING_H
#define TRAINING_H

#include <stdbool.h>

#include "config.h"
#include "genetic.h"

typedef struct {
  Population population;
  Agent saved_agent;
  Agent* best_agent;
  float best_fitness_ever;
} TrainingSession;

bool training_session_init(TrainingSession* session, const AppConfig* config);
void training_session_destroy(TrainingSession* session);
Agent* training_session_best_agent(TrainingSession* session);
Agent* training_session_train_generations(TrainingSession* session, int count, const AppConfig* config);
int training_run_headless(const AppConfig* config);

#endif
