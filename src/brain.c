#include "brain.h"
#include <stdlib.h>
#include <math.h>


static float random_weight(void) {
  return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static Direction turn_left(Direction direction) {
  switch (direction) {
  case UP:
    return LEFT;
  case LEFT:
    return DOWN;
  case DOWN:
    return RIGHT;
  case RIGHT:
    return UP;
  }
  return direction;
}
static Direction turn_right(Direction direction) {
  switch (direction) {
  case UP:
    return RIGHT;
  case RIGHT:
    return DOWN;
  case DOWN:
    return LEFT;
  case LEFT:
    return UP;
  }
  return direction;
}

static Direction direction_from_action(Direction current, int action) {
  if (action == 0) {
    return turn_left(current);
  }

  if (action == 2) {
    return turn_right(current);
  }

  return current;
}

static void build_inputs(const Game *game, float inputs[BRAIN_INPUTS]) {
  Direction current = game->snake.direction;
  Vec2 head = game->snake.body[0];

  Direction left = turn_left(current);
  Direction right = turn_right(current);

  inputs[0] = game_is_direction_safe(game, current) ? 0.0f : 1.0f;
  inputs[1] = game_is_direction_safe(game, left) ? 0.0f : 1.0f;
  inputs[2] = game_is_direction_safe(game, right) ? 0.0f : 1.0f;

  inputs[3] = current == UP ? 1.0f : 0.0f;
  inputs[4] = current == DOWN ? 1.0f : 0.0f;
  inputs[5] = current == LEFT ? 1.0f : 0.0f;
  inputs[6] = current == RIGHT ? 1.0f : 0.0f;
  inputs[7] = game->food.y < head.y ? 1.0f : 0.0f;
  inputs[8] = game->food.y > head.y ? 1.0f : 0.0f;
  inputs[9] = game->food.x < head.x ? 1.0f : 0.0f;
  inputs[10] = game->food.x > head.x ? 1.0f : 0.0f;
}

void brain_randomize(Brain *brain) {
  for (int i = 0; i < BRAIN_INPUTS; i++) {
    for (int j = 0; j < BRAIN_HIDDEN; j++) {
      brain->w1[i][j] = random_weight();
    }
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    brain->b1[i] = random_weight();
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    for (int j = 0; j < BRAIN_OUTPUTS; j++) {
      brain->w2[i][j] = random_weight();
    }
  }

  for (int i = 0; i < BRAIN_OUTPUTS; i++) {
    brain->b2[i] = random_weight();
  }
}

Direction brain_choose_direction(const Brain *brain, const Game *game) {
  float inputs[BRAIN_INPUTS];
  float hidden[BRAIN_HIDDEN];
  float outputs[BRAIN_OUTPUTS];

  build_inputs(game, inputs);

  for (int j = 0; j < BRAIN_HIDDEN; j++) {
    float sum = brain->b1[j];

    for (int i = 0; i < BRAIN_INPUTS; i++) {
      sum += inputs[i] * brain->w1[i][j];
    }

    hidden[j] = tanhf(sum);
  }

  for (int j = 0; j < BRAIN_OUTPUTS; j++) {
    float sum = brain->b2[j];

    for (int i = 0; i < BRAIN_HIDDEN; i++) {
      sum += hidden[i] * brain->w2[i][j];
    }

    outputs[j] = sum;
  }

  int best_action = 0;

  for (int i = 1; i < BRAIN_OUTPUTS; i++) {
    if (outputs[i] > outputs[best_action]) {
      best_action = i;
    }
  }

  return direction_from_action(game->snake.direction, best_action);
}