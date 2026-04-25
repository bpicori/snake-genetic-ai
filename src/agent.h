#ifndef AGENT_H
#define AGENT_H

#include "brain.h"
#include "game.h"

typedef struct {
  Brain brain;
  float fitness;
  int score;
  int steps;
} Agent;

void agent_randomize(Agent *agent);
Direction agent_choose_direction(const Agent *agent, const Game *game);
void agent_set_result(Agent *agent, const Game *game);

#endif