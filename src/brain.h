#ifndef BRAIN_H
#define BRAIN_H

#include <stdbool.h>

#include "game.h"
#include "tensor.h"

/*
 * Neural network input layout.
 *
 * The brain receives a compact snapshot of the current game state as 22 float
 * values. Some values are booleans represented as 0.0f or 1.0f; the rest are
 * normalized distances.
 *
 * Inputs 0-2 describe immediate danger relative to the snake's current
 * direction:
 *   0: danger straight
 *   1: danger left
 *   2: danger right
 *
 * Inputs 3-6 encode the snake's current movement direction:
 *   3: moving up
 *   4: moving down
 *   5: moving left
 *   6: moving right
 *
 * Inputs 7-10 encode the food position relative to the snake's head:
 *   7: food is above
 *   8: food is below
 *   9: food is left
 *  10: food is right
 *  11: normalized food dx
 *  12: normalized food dy
 *  13: normalized wall distance straight
 *  14: normalized wall distance left
 *  15: normalized wall distance right
 *  16: normalized body distance straight
 *  17: normalized body distance left
 *  18: normalized body distance right
 *  19: reachable space straight
 *  20: reachable space left
 *  21: reachable space right
 *
 * The network outputs 3 scores:
 *   0: turn left
 *   1: go straight
 *   2: turn right
 *
 * The highest-scoring output becomes the chosen action.
 */
#define BRAIN_INPUTS 22
#define BRAIN_HIDDEN 32
#define BRAIN_OUTPUTS 3

typedef struct {
  Tensor* w1; /* [BRAIN_INPUTS, BRAIN_HIDDEN] */
  Tensor* b1; /* [BRAIN_HIDDEN] */
  Tensor* w2; /* [BRAIN_HIDDEN, BRAIN_OUTPUTS] */
  Tensor* b2; /* [BRAIN_OUTPUTS] */
} Brain;

void brain_init(Brain* brain);
void brain_destroy(Brain* brain);

void brain_randomize(Brain* brain);
Direction brain_choose_direction(const Brain* brain, const Game* game);
void brain_copy(Brain* dest, const Brain* src);
void brain_mutate(Brain* brain, float mutation_rate, float mutation_strength);
void brain_crossover(Brain* child, const Brain* parent_a, const Brain* parent_b);
bool brain_save(const Brain* brain, const char* path);
bool brain_load(Brain* brain, const char* path);

#endif
