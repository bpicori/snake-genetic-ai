#include "game.h"
#include <stdlib.h>

static bool vec2_equals(Vec2 a, Vec2 b) { return a.x == b.x && a.y == b.y; }

static bool snake_contains_position(Snake *snake, Vec2 position) {
  for (int i = 0; i < snake->length; i++) {
    if (vec2_equals(snake->body[i], position)) {
      return true;
    }
  }
  return false;
}

static void spawn_food(Game *game) {
  Vec2 position;
  do {
    position = (Vec2){rand() % GRID_WIDTH, rand() % GRID_HEIGHT};
  } while (snake_contains_position(&game->snake, position));

  game->food = position;
}

void game_init(Game *game) {
  game->snake.length = 1;
  game->snake.body[0] = (Vec2){10, 10};
  game->snake.direction = RIGHT;

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
void game_update(Game *game) {
  if (!game->alive) {
    return;
  }

  Snake *snake = &game->snake;
  Vec2 old_tail = snake->body[snake->length - 1];
  Vec2 new_head = snake->body[0];

  // update the snake's head position based on its direction
  switch (snake->direction) {
  case UP:
    new_head.y -= 1;
    break;
  case DOWN:
    new_head.y += 1;
    break;
  case LEFT:
    new_head.x -= 1;
    break;
  case RIGHT:
    new_head.x += 1;
    break;
  }

  // check if the snake has hit the wall
  if (new_head.x < 0 || new_head.x >= GRID_WIDTH || new_head.y < 0 ||
      new_head.y >= GRID_HEIGHT) {
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

  if (will_eat_food) {
    game->score++;
    if (snake->length < MAX_SNAKE_LENGTH) {
      snake->body[snake->length] = old_tail;
      snake->length++;
    }

    spawn_food(game);
  }
}

void game_set_direction(Game *game, Direction direction) {
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