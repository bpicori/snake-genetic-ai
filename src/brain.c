#include "brain.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  BRAIN_FILE_MAGIC = 0x736e6b31u, /* "snk1" — toy format, not compatible with pre-tensor raw dumps */
  BRAIN_FILE_VERSION = 1u,
};

static const size_t k_brain_payload_floats =
    (size_t)BRAIN_INPUTS * BRAIN_HIDDEN + BRAIN_HIDDEN + (size_t)BRAIN_HIDDEN * BRAIN_OUTPUTS + BRAIN_OUTPUTS;

static bool random_bool(void) { return rand() % 2 == 0; }

static float random_float(void) { return (float)rand() / (float)RAND_MAX; }

static float random_mutation(float mutation_strength) { return (random_float() * 2.0f - 1.0f) * mutation_strength; }

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

static float mutate_value(float value, float mutation_rate, float mutation_strength) {
  if (random_float() < mutation_rate) {
    value += random_mutation(mutation_strength);
  }

  return value;
}

static void shift_rand_tensor_to_minus_one_one(Tensor* t) {
  for (size_t i = 0; i < t->element_count; i++) {
    t->data[i] = t->data[i] * 2.0f - 1.0f;
  }
}

void brain_init(Brain* brain) {
  brain->w1 = NULL;
  brain->b1 = NULL;
  brain->w2 = NULL;
  brain->b2 = NULL;

  size_t d_w1[] = {(size_t)BRAIN_INPUTS, (size_t)BRAIN_HIDDEN};
  size_t d_b1[] = {(size_t)BRAIN_HIDDEN};
  size_t d_w2[] = {(size_t)BRAIN_HIDDEN, (size_t)BRAIN_OUTPUTS};
  size_t d_b2[] = {(size_t)BRAIN_OUTPUTS};

  brain->w1 = tensor_create(2, d_w1);
  brain->b1 = tensor_create(1, d_b1);
  brain->w2 = tensor_create(2, d_w2);
  brain->b2 = tensor_create(1, d_b2);
}

void brain_destroy(Brain* brain) {
  tensor_destroy(brain->w1);
  tensor_destroy(brain->b1);
  tensor_destroy(brain->w2);
  tensor_destroy(brain->b2);
  brain->w1 = NULL;
  brain->b1 = NULL;
  brain->w2 = NULL;
  brain->b2 = NULL;
}

void brain_randomize(Brain* brain) {
  tensor_destroy(brain->w1);
  tensor_destroy(brain->b1);
  tensor_destroy(brain->w2);
  tensor_destroy(brain->b2);

  size_t d_w1[] = {(size_t)BRAIN_INPUTS, (size_t)BRAIN_HIDDEN};
  size_t d_b1[] = {(size_t)BRAIN_HIDDEN};
  size_t d_w2[] = {(size_t)BRAIN_HIDDEN, (size_t)BRAIN_OUTPUTS};
  size_t d_b2[] = {(size_t)BRAIN_OUTPUTS};

  brain->w1 = tensor_rand(2, d_w1);
  brain->b1 = tensor_rand(1, d_b1);
  brain->w2 = tensor_rand(2, d_w2);
  brain->b2 = tensor_rand(1, d_b2);

  shift_rand_tensor_to_minus_one_one(brain->w1);
  shift_rand_tensor_to_minus_one_one(brain->b1);
  shift_rand_tensor_to_minus_one_one(brain->w2);
  shift_rand_tensor_to_minus_one_one(brain->b2);
}

Direction brain_choose_direction(const Brain* brain, const Game* game) {
  float inputs[BRAIN_INPUTS];
  build_inputs(game, inputs);

  size_t x_dims[] = {1, (size_t)BRAIN_INPUTS};
  Tensor* x = tensor_from_array(2, x_dims, inputs);

  Tensor* z1 = tensor_matmul(x, brain->w1);
  Tensor* z1_b = tensor_add_bias(z1, brain->b1);
  Tensor* h = tensor_tanh(z1_b);
  Tensor* z2 = tensor_matmul(h, brain->w2);
  Tensor* y = tensor_add_bias(z2, brain->b2);

  int best_action = (int)tensor_argmax(y);

  tensor_destroy(y);
  tensor_destroy(z2);
  tensor_destroy(h);
  tensor_destroy(z1_b);
  tensor_destroy(z1);
  tensor_destroy(x);

  return direction_from_action(game->snake.direction, best_action);
}

void brain_copy(Brain* dest, const Brain* src) {
  memcpy(dest->w1->data, src->w1->data, dest->w1->element_count * sizeof(float));
  memcpy(dest->b1->data, src->b1->data, dest->b1->element_count * sizeof(float));
  memcpy(dest->w2->data, src->w2->data, dest->w2->element_count * sizeof(float));
  memcpy(dest->b2->data, src->b2->data, dest->b2->element_count * sizeof(float));
}

void brain_mutate(Brain* brain, float mutation_rate, float mutation_strength) {
  for (size_t i = 0; i < brain->w1->element_count; i++) {
    brain->w1->data[i] = mutate_value(brain->w1->data[i], mutation_rate, mutation_strength);
  }

  for (size_t i = 0; i < brain->b1->element_count; i++) {
    brain->b1->data[i] = mutate_value(brain->b1->data[i], mutation_rate, mutation_strength);
  }

  for (size_t i = 0; i < brain->w2->element_count; i++) {
    brain->w2->data[i] = mutate_value(brain->w2->data[i], mutation_rate, mutation_strength);
  }

  for (size_t i = 0; i < brain->b2->element_count; i++) {
    brain->b2->data[i] = mutate_value(brain->b2->data[i], mutation_rate, mutation_strength);
  }
}

void brain_crossover(Brain* child, const Brain* parent_a, const Brain* parent_b) {
  for (size_t i = 0; i < child->w1->element_count; i++) {
    child->w1->data[i] = random_bool() ? parent_a->w1->data[i] : parent_b->w1->data[i];
  }

  for (size_t i = 0; i < child->b1->element_count; i++) {
    child->b1->data[i] = random_bool() ? parent_a->b1->data[i] : parent_b->b1->data[i];
  }

  for (size_t i = 0; i < child->w2->element_count; i++) {
    child->w2->data[i] = random_bool() ? parent_a->w2->data[i] : parent_b->w2->data[i];
  }

  for (size_t i = 0; i < child->b2->element_count; i++) {
    child->b2->data[i] = random_bool() ? parent_a->b2->data[i] : parent_b->b2->data[i];
  }
}

bool brain_save(const Brain* brain, const char* path) {
  FILE* file = fopen(path, "wb");

  if (file == NULL) {
    return false;
  }

  uint32_t magic = BRAIN_FILE_MAGIC;
  uint32_t version = BRAIN_FILE_VERSION;

  if (fwrite(&magic, sizeof magic, 1, file) != 1 || fwrite(&version, sizeof version, 1, file) != 1) {
    fclose(file);
    return false;
  }

  size_t n = 0;
  n += fwrite(brain->w1->data, sizeof(float), brain->w1->element_count, file);
  n += fwrite(brain->b1->data, sizeof(float), brain->b1->element_count, file);
  n += fwrite(brain->w2->data, sizeof(float), brain->w2->element_count, file);
  n += fwrite(brain->b2->data, sizeof(float), brain->b2->element_count, file);

  fclose(file);

  return n == k_brain_payload_floats;
}

bool brain_load(Brain* brain, const char* path) {
  FILE* file = fopen(path, "rb");

  if (file == NULL) {
    return false;
  }

  uint32_t magic = 0;
  uint32_t version = 0;

  if (fread(&magic, sizeof magic, 1, file) != 1 || fread(&version, sizeof version, 1, file) != 1) {
    fclose(file);
    return false;
  }

  if (magic != BRAIN_FILE_MAGIC || version != BRAIN_FILE_VERSION) {
    fclose(file);
    return false;
  }

  size_t n = 0;
  n += fread(brain->w1->data, sizeof(float), brain->w1->element_count, file);
  n += fread(brain->b1->data, sizeof(float), brain->b1->element_count, file);
  n += fread(brain->w2->data, sizeof(float), brain->w2->element_count, file);
  n += fread(brain->b2->data, sizeof(float), brain->b2->element_count, file);

  fclose(file);

  return n == k_brain_payload_floats;
}
