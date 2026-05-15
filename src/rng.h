#ifndef RNG_H
#define RNG_H

#include <stdbool.h>

/* Seed libc rand/srand and zero-tensor's tensor_seed (same PRNG state). */
void rng_seed(unsigned int seed);

/* Uniform in [min, max), half-open — matches tensor_fill_uniform. */
float rng_uniform(float min, float max);

/* Integer in [0, upper_exclusive). Requires upper_exclusive > 0. */
int rng_int(int upper_exclusive);

/* 50/50 coin flip. */
bool rng_bool(void);

#endif
