#include "game.h"

#include <stdlib.h>

#include "rng.h"

static bool vec2_equals(Vec2 a, Vec2 b) { return a.x == b.x && a.y == b.y; }

Vec2 game_direction_delta(Direction direction) {
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

Direction game_turn_left(Direction direction) {
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

Direction game_turn_right(Direction direction) {
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

static bool snake_contains_position(Snake* snake, Vec2 position) {
  for (int i = 0; i < snake->length; i++) {
    if (vec2_equals(snake->body[i], position)) {
      return true;
    }
  }
  return false;
}

static void spawn_food(Game* game) {
  Vec2 position;
  do {
    position = (Vec2){rng_int(GRID_WIDTH), rng_int(GRID_HEIGHT)};
  } while (snake_contains_position(&game->snake, position));

  game->food = position;
}

static Vec2 next_head_for_direction(const Game* game, Direction direction) {
  Vec2 head = game->snake.body[0];
  Vec2 d = game_direction_delta(direction);
  head.x += d.x;
  head.y += d.y;
  return head;
}

// Manhattan distance to the food
static int distance_to_food(const Game* game, Vec2 position) {
  int dx = game->food.x - position.x;
  int dy = game->food.y - position.y;

  if (dx < 0) dx = -dx;
  if (dy < 0) dy = -dy;

  return dx + dy;
}

bool game_is_direction_safe(const Game* game, Direction direction) {
  Vec2 next_head = next_head_for_direction(game, direction);
  if (next_head.x < 0 || next_head.x >= GRID_WIDTH || next_head.y < 0 || next_head.y >= GRID_HEIGHT) {
    return false;
  }
  const Snake* snake = &game->snake;
  for (int i = 1; i < snake->length; i++) {
    if (snake->body[i].x == next_head.x && snake->body[i].y == next_head.y) {
      return false;
    }
  }
  return true;
}

void game_init(Game* game) {
  game->snake.length = 1;
  game->snake.body[0] = (Vec2){5, 5};
  game->snake.direction = RIGHT;
  game->steps_since_food = 0;
  game->distance_reward = 0;

  spawn_food(game);

  game->score = 0;
  game->alive = true;
  game->steps = 0;
}

// 1. calculate next head position
// 2. validate wall
// 3. check self-collision
// 4. commit movement
// 5. increment steps
// 6. handle eating/growth
void game_update(Game* game) {
  if (!game->alive) {
    return;
  }

  Snake* snake = &game->snake;
  Vec2 old_tail = snake->body[snake->length - 1];
  Vec2 new_head = snake->body[0];
  int old_distance = distance_to_food(game, new_head);

  Vec2 dhead = game_direction_delta(snake->direction);
  new_head.x += dhead.x;
  new_head.y += dhead.y;

  // check if the snake has hit the wall
  if (new_head.x < 0 || new_head.x >= GRID_WIDTH || new_head.y < 0 || new_head.y >= GRID_HEIGHT) {
    game->alive = false;
    return;
  }

  bool will_eat_food = vec2_equals(new_head, game->food);

  int collision_length = will_eat_food ? snake->length : snake->length - 1;
  for (int i = 0; i < collision_length; i++) {
    if (vec2_equals(snake->body[i], new_head)) {
      game->alive = false;
      return;
    }
  }

  for (int i = snake->length - 1; i > 0; i--) {
    snake->body[i] = snake->body[i - 1];
  }

  snake->body[0] = new_head;

  game->steps++;
  game->steps_since_food++;

  int new_distance = distance_to_food(game, new_head);
  if (new_distance < old_distance) {
    game->distance_reward++;
  } else if (new_distance > old_distance) {
    game->distance_reward--;
  }

  if (will_eat_food) {
    game->score++;
    if (snake->length < MAX_SNAKE_LENGTH) {
      snake->body[snake->length] = old_tail;
      snake->length++;
    }
    game->steps_since_food = 0;
    spawn_food(game);
  }
}

void game_set_direction(Game* game, Direction direction) {
  Direction current = game->snake.direction;

  if (current == UP && direction == DOWN) {
    return;
  }

  if (current == DOWN && direction == UP) {
    return;
  }

  if (current == LEFT && direction == RIGHT) {
    return;
  }

  if (current == RIGHT && direction == LEFT) {
    return;
  }

  game->snake.direction = direction;
}