#include "brain.h"
#include <stdlib.h>

static float random_weight(void) {
  return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
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