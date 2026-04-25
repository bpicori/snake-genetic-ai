#include "brain.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Returns true or false randomly with about 50% chance.
static bool random_bool(void) { return rand() % 2 == 0; }

// Creates a random neural-network weight between -1.0 and 1.0.
static float random_weight(void) { return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; }

// Returns a random float between 0.0 and 1.0.
static float random_float(void) { return (float)rand() / (float)RAND_MAX; }

// Creates a small random change between -mutation_strength and
// +mutation_strength.
static float random_mutation(float mutation_strength) { return (random_float() * 2.0f - 1.0f) * mutation_strength; }
// Converts the current direction into the direction produced by turning left.
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

// Converts the current direction into the direction produced by turning right.
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

/*
 * Converts a neural-network action into an actual game direction.
 *
 * action 0 = turn left
 * action 1 = keep going straight
 * action 2 = turn right
 */
static Direction direction_from_action(Direction current, int action) {
  if (action == 0) {
    return turn_left(current);
  }

  if (action == 2) {
    return turn_right(current);
  }

  return current;
}

static Vec2 direction_delta(Direction direction) {
  switch (direction) {
    case UP:
      return (Vec2){0, -1};
    case DOWN:
      return (Vec2){0, 1};
    case LEFT:
      return (Vec2){-1, 0};
    case RIGHT:
      return (Vec2){1, 0};
  }

  return (Vec2){0, 0};
}

static float max_distance_for_direction(Direction direction) {
  return direction == LEFT || direction == RIGHT ? (float)GRID_WIDTH : (float)GRID_HEIGHT;
}

static float normalized_wall_distance(Vec2 head, Direction direction) {
  Vec2 delta = direction_delta(direction);
  int distance = 0;
  Vec2 position = head;

  while (true) {
    position.x += delta.x;
    position.y += delta.y;

    if (position.x < 0 || position.x >= GRID_WIDTH || position.y < 0 || position.y >= GRID_HEIGHT) {
      break;
    }

    distance++;
  }

  return (float)distance / max_distance_for_direction(direction);
}

static float normalized_body_distance(const Game* game, Vec2 head, Direction direction) {
  Vec2 delta = direction_delta(direction);
  int distance = 0;
  Vec2 position = head;

  while (true) {
    position.x += delta.x;
    position.y += delta.y;

    if (position.x < 0 || position.x >= GRID_WIDTH || position.y < 0 || position.y >= GRID_HEIGHT) {
      return 1.0f;
    }

    distance++;

    for (int i = 1; i < game->snake.length; i++) {
      if (game->snake.body[i].x == position.x && game->snake.body[i].y == position.y) {
        return (float)distance / max_distance_for_direction(direction);
      }
    }
  }
}

static bool position_is_blocked_for_flood_fill(const Game* game, Vec2 position) {
  if (position.x < 0 || position.x >= GRID_WIDTH || position.y < 0 || position.y >= GRID_HEIGHT) {
    return true;
  }
  for (int i = 1; i < game->snake.length; i++) {
    if (game->snake.body[i].x == position.x && game->snake.body[i].y == position.y) {
      return true;
    }
  }
  return false;
}

static float normalized_reachable_space(const Game* game, Vec2 start) {
  if (position_is_blocked_for_flood_fill(game, start)) {
    return 0.0f;
  }

  bool visited[GRID_HEIGHT][GRID_WIDTH] = {false};
  Vec2 queue[MAX_SNAKE_LENGTH];
  int head = 0;
  int tail = 0;
  int reachable = 0;
  queue[tail++] = start;
  visited[start.y][start.x] = true;
  while (head < tail) {
    Vec2 current = queue[head++];
    reachable++;
    Vec2 neighbors[4] = {
        {current.x, current.y - 1},
        {current.x, current.y + 1},
        {current.x - 1, current.y},
        {current.x + 1, current.y},
    };

    for (int i = 0; i < 4; i++) {
      Vec2 next = neighbors[i];
      if (next.x < 0 || next.x >= GRID_WIDTH || next.y < 0 || next.y >= GRID_HEIGHT) {
        continue;
      }
      if (visited[next.y][next.x] || position_is_blocked_for_flood_fill(game, next)) {
        continue;
      }
      visited[next.y][next.x] = true;
      queue[tail++] = next;
    }
  }
  return (float)reachable / (float)MAX_SNAKE_LENGTH;
}

/*
 * Converts the current Game state into the input values used by the brain.
 *
 * The inputs describe nearby danger, current movement direction, food position,
 * and how much free space exists around the snake head.
 */
static void build_inputs(const Game* game, float inputs[BRAIN_INPUTS]) {
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
  inputs[11] = (float)(game->food.x - head.x) / GRID_WIDTH;
  inputs[12] = (float)(game->food.y - head.y) / GRID_HEIGHT;
  inputs[13] = normalized_wall_distance(head, current);
  inputs[14] = normalized_wall_distance(head, left);
  inputs[15] = normalized_wall_distance(head, right);
  inputs[16] = normalized_body_distance(game, head, current);
  inputs[17] = normalized_body_distance(game, head, left);
  inputs[18] = normalized_body_distance(game, head, right);

  Vec2 straight_delta = direction_delta(current);
  Vec2 left_delta = direction_delta(left);
  Vec2 right_delta = direction_delta(right);

  Vec2 straight_head = {head.x + straight_delta.x, head.y + straight_delta.y};
  Vec2 left_head = {head.x + left_delta.x, head.y + left_delta.y};
  Vec2 right_head = {head.x + right_delta.x, head.y + right_delta.y};

  inputs[19] = normalized_reachable_space(game, straight_head);
  inputs[20] = normalized_reachable_space(game, left_head);
  inputs[21] = normalized_reachable_space(game, right_head);
}

/*
 * Mutates one weight or bias value.
 *
 * mutation_rate controls the chance that the value changes.
 * mutation_strength controls how large the change can be.
 *
 * Example:
 *   value = 0.70
 *   mutation_rate = 0.05      // 5% chance this value changes
 *   mutation_strength = 0.20  // change is between -0.20 and +0.20
 *
 * If this value is selected for mutation:
 *   random change = -0.13
 *   new value = 0.70 + (-0.13) = 0.57
 *
 * If it is not selected, it stays 0.70.
 */
static float mutate_value(float value, float mutation_rate, float mutation_strength) {
  if (random_float() < mutation_rate) {
    value += random_mutation(mutation_strength);
  }

  return value;
}

/*
 * Fills a brain with random weights and biases.
 *
 * This is used when creating the first generation of agents before any
 * evolution has happened.
 */
void brain_randomize(Brain* brain) {
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

/*
 * Runs the neural network and chooses the snake's next direction.
 *
 * Flow:
 *   1. Build input values from the game state.
 *   2. Compute hidden neuron values.
 *   3. Compute output scores.
 *   4. Pick the highest-scoring output.
 *   5. Convert that output into left, straight, or right movement.
 */
Direction brain_choose_direction(const Brain* brain, const Game* game) {
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

/*
 * Copies all weights and biases from one brain to another.
 *
 * Used when an agent survives unchanged or when creating a child from a parent.
 */
void brain_copy(Brain* dest, const Brain* src) {
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
/*
 * Applies random mutations to all weights and biases in the brain.
 *
 * Most values stay unchanged. A few values are nudged slightly, which creates
 * variation between generations.
 */
void brain_mutate(Brain* brain, float mutation_rate, float mutation_strength) {
  for (int i = 0; i < BRAIN_INPUTS; i++) {
    for (int j = 0; j < BRAIN_HIDDEN; j++) {
      brain->w1[i][j] = mutate_value(brain->w1[i][j], mutation_rate, mutation_strength);
    }
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    brain->b1[i] = mutate_value(brain->b1[i], mutation_rate, mutation_strength);
  }

  for (int i = 0; i < BRAIN_HIDDEN; i++) {
    for (int j = 0; j < BRAIN_OUTPUTS; j++) {
      brain->w2[i][j] = mutate_value(brain->w2[i][j], mutation_rate, mutation_strength);
    }
  }

  for (int i = 0; i < BRAIN_OUTPUTS; i++) {
    brain->b2[i] = mutate_value(brain->b2[i], mutation_rate, mutation_strength);
  }
}

/*
 * Creates a child brain from two parent brains.
 *
 * For every weight and bias, the child randomly inherits the value from either
 * parent A or parent B. Mutation can be applied afterward.
 */
void brain_crossover(Brain* child, const Brain* parent_a, const Brain* parent_b) {
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

/*
 * Saves the brain's weights and biases to a binary file.
 *
 * This lets us keep the best trained brain after the program exits.
 */
bool brain_save(const Brain* brain, const char* path) {
  FILE* file = fopen(path, "wb");

  if (file == NULL) {
    return false;
  }

  size_t written = fwrite(brain, sizeof(Brain), 1, file);
  fclose(file);

  return written == 1;
}

/*
 * Loads previously saved weights and biases from a binary file.
 *
 * Returns false if the file does not exist or cannot be read.
 */
bool brain_load(Brain* brain, const char* path) {
  FILE* file = fopen(path, "rb");

  if (file == NULL) {
    return false;
  }

  size_t read = fread(brain, sizeof(Brain), 1, file);
  fclose(file);

  return read == 1;
}