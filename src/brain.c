#include "brain.h"
#include <math.h>
#include <stdlib.h>

static bool random_bool(void) { return rand() % 2 == 0; }

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

static float random_float(void) { return (float)rand() / (float)RAND_MAX; }

static float random_mutation(float mutation_strength) {
  return (random_float() * 2.0f - 1.0f) * mutation_strength;
}

static float mutate_value(float value, float mutation_rate,
                          float mutation_strength) {
  if (random_float() < mutation_rate) {
    value += random_mutation(mutation_strength);
  }

  return value;
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

void brain_copy(Brain *dest, const Brain *src) {
  for (int i = 0; i < BRAIN_INPUTS; i++) {
    for (int j = 0; j < BRAIN_HIDDEN; j++) {
      dest->w1[i][j] = src->w1[i][j];
    }
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    dest->b1[i] = src->b1[i];
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    for (int j = 0; j < BRAIN_OUTPUTS; j++) {
      dest->w2[i][j] = src->w2[i][j];
    }
  }

  for (int i = 0; i < BRAIN_OUTPUTS; i++) {
    dest->b2[i] = src->b2[i];
  }
}

void brain_mutate(Brain *brain, float mutation_rate, float mutation_strength) {
  for (int i = 0; i < BRAIN_INPUTS; i++) {
    for (int j = 0; j < BRAIN_HIDDEN; j++) {
      brain->w1[i][j] =
          mutate_value(brain->w1[i][j], mutation_rate, mutation_strength);
    }
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    brain->b1[i] = mutate_value(brain->b1[i], mutation_rate, mutation_strength);
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    for (int j = 0; j < BRAIN_OUTPUTS; j++) {
      brain->w2[i][j] =
          mutate_value(brain->w2[i][j], mutation_rate, mutation_strength);
    }
  }

  for (int i = 0; i < BRAIN_OUTPUTS; i++) {
    brain->b2[i] = mutate_value(brain->b2[i], mutation_rate, mutation_strength);
  }
}

void brain_crossover(Brain *child, const Brain *parent_a,
                     const Brain *parent_b) {
  for (int i = 0; i < BRAIN_INPUTS; i++) {
    for (int j = 0; j < BRAIN_HIDDEN; j++) {
      child->w1[i][j] = random_bool() ? parent_a->w1[i][j] : parent_b->w1[i][j];
    }
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    child->b1[i] = random_bool() ? parent_a->b1[i] : parent_b->b1[i];
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    for (int j = 0; j < BRAIN_OUTPUTS; j++) {
      child->w2[i][j] = random_bool() ? parent_a->w2[i][j] : parent_b->w2[i][j];
    }
  }

  for (int i = 0; i < BRAIN_OUTPUTS; i++) {
    child->b2[i] = random_bool() ? parent_a->b2[i] : parent_b->b2[i];
  }
}