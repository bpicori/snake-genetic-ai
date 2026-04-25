#include "agent.h"

void agent_randomize(Agent *agent) {
  brain_randomize(&agent->brain);
  agent->fitness = 0.0f;
  agent->score = 0;
  agent->steps = 0;
  agent->distance_reward = 0;
}

Direction agent_choose_direction(const Agent *agent, const Game *game) {
  return brain_choose_direction(&agent->brain, game);
}

void agent_set_result(Agent *agent, const Game *game) {
  agent->score = game->score;
  agent->steps = game->steps;
  agent->distance_reward = game->distance_reward;

  /*
   * Fitness is the score used by the genetic algorithm to rank agents.
   * Eating food is much more valuable than only surviving, so each food
   * point is weighted heavily.
   *
   * Example:
   *   score = 3, steps = 120 -> fitness = 3 * 1000 + 120 = 3120
   *   score = 0, steps = 300 -> fitness = 0 * 1000 + 300 = 300
   *
   * This makes the snake that ate food rank higher, even if another snake
   * survived longer without eating.
   */
  agent->fitness = (float)((game->score * 1000) + game->steps + (game->distance_reward * 5));
}