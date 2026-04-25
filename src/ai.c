#include "ai.h"

#include "game.h"

Direction choose_direction_toward_food(const Game* game) {
  Vec2 head = game->snake.body[0];

  Direction preferred[6];
  int count = 0;

  if (game->food.x > head.x) {
    preferred[count++] = RIGHT;
  } else if (game->food.x < head.x) {
    preferred[count++] = LEFT;
  }
  if (game->food.y > head.y) {
    preferred[count++] = DOWN;
  } else if (game->food.y < head.y) {
    preferred[count++] = UP;
  }

  preferred[count++] = UP;
  preferred[count++] = RIGHT;
  preferred[count++] = DOWN;
  preferred[count++] = LEFT;

  for (int i = 0; i < count; i++) {
    if (game_is_direction_safe(game, preferred[i])) {
      return preferred[i];
    }
  }

  return game->snake.direction;
}