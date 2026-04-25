# Snake Genetic AI

A small C project where Snake learns through a neural network and a genetic algorithm.

It uses SDL2 for rendering.

Build:

```sh
make build
```

Train without rendering:

```sh
./out/snake-ai --train --no-render --strategy v4 --generations 1000 --brain out/v4.brain
```

Strategies:

- `v1`: copy the best snake and mutate the rest from it.
- `v2`: keep the top snakes and breed from a small parent pool.
- `v3`: mix two parent brains with crossover, then mutate.
- `v4`: pick parents with tournament selection before crossover.
- `adaptive`: raise or lower mutation as training improves or stalls.
- `adaptive-conservative`: like adaptive, but with smaller mutation swings.

Replay the saved brain:

```sh
./out/snake-ai --replay --brain out/v4.brain
```
