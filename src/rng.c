#include "rng.h"

#include <assert.h>
#include <stdlib.h>

#include "tensor.h"

void rng_seed(unsigned int seed) { tensor_seed(seed); }

float rng_uniform(float min, float max) {
  assert(min < max);
  float span = max - min;
  float u = (float)rand() / ((float)RAND_MAX + 1.0f);
  return min + span * u;
}

int rng_int(int upper_exclusive) {
  assert(upper_exclusive > 0);
  return rand() % upper_exclusive;
}

bool rng_bool(void) { return rand() % 2 == 0; }
