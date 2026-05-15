#ifndef GAME_H
#define GAME_H

#include <stdbool.h>

#define GRID_WIDTH 20
#define GRID_HEIGHT 20
#define MAX_SNAKE_LENGTH (GRID_WIDTH * GRID_HEIGHT)

typedef struct {
  int x;
  int y;
} Vec2;

typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

typedef struct {
  int length;
  Vec2 body[MAX_SNAKE_LENGTH];
  Direction direction;
} Snake;

typedef struct {
  Snake snake;
  Vec2 food;
  int score;
  bool alive;
  int steps;
  int steps_since_food;
  int distance_reward;
} Game;

void game_init(Game* game);
void game_update(Game* game);
void game_set_direction(Game* game, Direction direction);
bool game_is_direction_safe(const Game* game, Direction direction);

/* Grid / movement helpers shared by game simulation and brain sensors. */
Vec2 game_direction_delta(Direction direction);
Direction game_turn_left(Direction direction);
Direction game_turn_right(Direction direction);

#endif