#include "brain.h"

#include <stdint.h>
#include <stdio.h>

#include "rng.h"

enum {
  BRAIN_FILE_MAGIC = 0x736e6b31u, /* "snk1" — toy format, not compatible with pre-tensor raw dumps */
  BRAIN_FILE_VERSION = 1u,
};

static const size_t k_brain_payload_floats =
    (size_t)BRAIN_INPUTS * BRAIN_HIDDEN + BRAIN_HIDDEN + (size_t)BRAIN_HIDDEN * BRAIN_OUTPUTS + BRAIN_OUTPUTS;

#define BRAIN_NUM_TENSORS 4

static void brain_tensors(const Brain* brain, Tensor* tensors[BRAIN_NUM_TENSORS]) {
  tensors[0] = brain->w1;
  tensors[1] = brain->b1;
  tensors[2] = brain->w2;
  tensors[3] = brain->b2;
}

static void brain_alloc_tensors(Brain* brain) {
  size_t d_w1[] = {(size_t)BRAIN_INPUTS, (size_t)BRAIN_HIDDEN};
  size_t d_b1[] = {(size_t)BRAIN_HIDDEN};
  size_t d_w2[] = {(size_t)BRAIN_HIDDEN, (size_t)BRAIN_OUTPUTS};
  size_t d_b2[] = {(size_t)BRAIN_OUTPUTS};

  brain->w1 = tensor_create(2, d_w1);
  brain->b1 = tensor_create(1, d_b1);
  brain->w2 = tensor_create(2, d_w2);
  brain->b2 = tensor_create(1, d_b2);
}

static Direction direction_from_action(Direction current, int action) {
  if (action == BRAIN_ACTION_TURN_LEFT) {
    return game_turn_left(current);
  }

  if (action == BRAIN_ACTION_TURN_RIGHT) {
    return game_turn_right(current);
  }

  return current;
}

static float max_distance_for_direction(Direction direction) {
  return direction == LEFT || direction == RIGHT ? (float)GRID_WIDTH : (float)GRID_HEIGHT;
}

static float normalized_wall_distance(Vec2 head, Vec2 delta, Direction direction) {
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

static float normalized_body_distance(const Game* game, Vec2 head, Vec2 delta, Direction direction) {
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

  Direction left = game_turn_left(current);
  Direction right = game_turn_right(current);

  Vec2 d_cur = game_direction_delta(current);
  Vec2 d_left = game_direction_delta(left);
  Vec2 d_right = game_direction_delta(right);

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
  inputs[13] = normalized_wall_distance(head, d_cur, current);
  inputs[14] = normalized_wall_distance(head, d_left, left);
  inputs[15] = normalized_wall_distance(head, d_right, right);
  inputs[16] = normalized_body_distance(game, head, d_cur, current);
  inputs[17] = normalized_body_distance(game, head, d_left, left);
  inputs[18] = normalized_body_distance(game, head, d_right, right);

  Vec2 straight_head = {head.x + d_cur.x, head.y + d_cur.y};
  Vec2 left_head = {head.x + d_left.x, head.y + d_left.y};
  Vec2 right_head = {head.x + d_right.x, head.y + d_right.y};

  inputs[19] = normalized_reachable_space(game, straight_head);
  inputs[20] = normalized_reachable_space(game, left_head);
  inputs[21] = normalized_reachable_space(game, right_head);
}

static float mutate_value(float value, float mutation_rate, float mutation_strength) {
  if (rng_uniform(0.0f, 1.0f) < mutation_rate) {
    value += rng_uniform(-mutation_strength, mutation_strength);
  }

  return value;
}

void brain_init(Brain* brain) {
  brain->w1 = NULL;
  brain->b1 = NULL;
  brain->w2 = NULL;
  brain->b2 = NULL;
  brain_alloc_tensors(brain);
}

void brain_destroy(Brain* brain) {
  Tensor* ts[BRAIN_NUM_TENSORS];
  brain_tensors(brain, ts);
  for (int i = 0; i < BRAIN_NUM_TENSORS; i++) {
    tensor_destroy(ts[i]);
  }
  brain->w1 = NULL;
  brain->b1 = NULL;
  brain->w2 = NULL;
  brain->b2 = NULL;
}

void brain_randomize(Brain* brain) {
  Tensor* ts[BRAIN_NUM_TENSORS];
  brain_tensors(brain, ts);
  for (int i = 0; i < BRAIN_NUM_TENSORS; i++) {
    tensor_fill_uniform(ts[i], -1.0f, 1.0f);
  }
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
  tensor_copy_into(dest->w1, src->w1);
  tensor_copy_into(dest->b1, src->b1);
  tensor_copy_into(dest->w2, src->w2);
  tensor_copy_into(dest->b2, src->b2);
}

void brain_mutate(Brain* brain, float mutation_rate, float mutation_strength) {
  Tensor* ts[BRAIN_NUM_TENSORS];
  brain_tensors(brain, ts);
  for (int ti = 0; ti < BRAIN_NUM_TENSORS; ti++) {
    Tensor* t = ts[ti];
    for (size_t i = 0; i < t->element_count; i++) {
      t->data[i] = mutate_value(t->data[i], mutation_rate, mutation_strength);
    }
  }
}

void brain_crossover(Brain* child, const Brain* parent_a, const Brain* parent_b) {
  Tensor* c[BRAIN_NUM_TENSORS];
  Tensor* a[BRAIN_NUM_TENSORS];
  Tensor* b[BRAIN_NUM_TENSORS];
  brain_tensors(child, c);
  brain_tensors(parent_a, a);
  brain_tensors(parent_b, b);

  for (int ti = 0; ti < BRAIN_NUM_TENSORS; ti++) {
    for (size_t i = 0; i < c[ti]->element_count; i++) {
      c[ti]->data[i] = rng_bool() ? a[ti]->data[i] : b[ti]->data[i];
    }
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

  Tensor* ts[BRAIN_NUM_TENSORS];
  brain_tensors(brain, ts);

  size_t n = 0;
  for (int i = 0; i < BRAIN_NUM_TENSORS; i++) {
    n += tensor_write_raw(ts[i], file);
  }

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

  Tensor* ts[BRAIN_NUM_TENSORS];
  brain_tensors(brain, ts);

  size_t n = 0;
  for (int i = 0; i < BRAIN_NUM_TENSORS; i++) {
    n += tensor_read_raw(ts[i], file);
  }

  fclose(file);

  return n == k_brain_payload_floats;
}
