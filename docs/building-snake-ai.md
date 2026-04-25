# Building Snake AI With C, SDL2, Neural Networks, and Genetic Algorithms

This project started as a simple Snake game in C using SDL2, then evolved into
an AI training environment where snakes are controlled by neural-network brains
and improved using genetic algorithms.

The goal was not only to build the final program, but to understand each piece:
the game rules, rendering, AI decision making, fitness scoring, population
selection, mutation, crossover, and replay.

## 1. Basic SDL2 Setup

The first step was creating a minimal SDL2 application.

It included:

- initializing SDL
- creating a window
- creating a renderer
- handling quit events
- drawing a simple shape
- cleaning up SDL resources

This confirmed that the project could open a window and render before adding
Snake logic.

At this point, `main.c` owned everything. That was fine at the beginning because
the project was still small.

## 2. Snake Game Logic

The Snake rules were moved into `game.c` and `game.h`.

The game module owns:

- grid size
- snake body
- snake direction
- food position
- score
- alive/dead state
- movement
- wall collision
- self collision
- food eating
- reset logic

The important design decision was keeping `game.c` independent from SDL.

That matters because training needs to run many games quickly without rendering
anything. If the game rules depended on SDL, every AI simulation would be tied
to the window.

The game became a pure simulation:

```text
Game state -> direction -> updated Game state
```

## 3. Grid Coordinates

The game uses grid coordinates, not pixels.

For example:

```text
snake head = (5, 5)
food       = (12, 8)
```

SDL only turns those grid positions into pixels when drawing.

This separation keeps the game logic simple. The snake does not know about
window size, colors, rectangles, or rendering.

## 4. Snake Movement

Each update calculates the next head position based on direction.

The safe update order became:

```text
1. calculate next head position
2. check wall collision
3. check self collision
4. move body
5. place new head
6. update score/growth if food was eaten
```

This fixed the issue where the snake head could visually move outside the wall
before dying.

## 5. Food And Growth

Food spawning was improved so food does not appear inside the snake.

Growth was also made explicit.

Normal movement means:

```text
head moves forward
tail disappears
```

Eating food means:

```text
head moves forward
old tail stays
snake length increases
```

So the code stores the old tail before moving, then appends it back if food was
eaten.

## 6. Manual Control

Keyboard input was added so the game could be played manually.

Arrow keys change direction. The game prevents instant reverse movement:

```text
moving right -> cannot immediately move left
moving up    -> cannot immediately move down
```

This matches normal Snake rules and avoids impossible turns.

## 7. Rendering

Rendering started in `main.c`, then moved into `render.c` and `render.h`.

The renderer is responsible only for drawing:

- background
- food
- snake cells

It does not update the game and does not decide movement.

The responsibilities became:

```text
game.c   = rules and state
render.c = drawing
main.c   = app loop and coordination
```

## 8. Simple Rule-Based AI

Before neural networks, a simple AI was added that moves toward food.

It was intentionally dumb. Its purpose was to prove that the game could be
controlled by code instead of keyboard input.

The important architecture was:

```text
observe game state -> choose direction -> update game
```

That is the same shape later used by the neural network.

## 9. Neural Network Brain

The `Brain` struct was introduced to control the snake.

The network shape is:

```text
11 inputs -> 16 hidden neurons -> 3 outputs
```

The inputs describe:

- danger straight
- danger left
- danger right
- current movement direction
- food direction relative to the snake head

The outputs are:

```text
0 = turn left
1 = go straight
2 = turn right
```

The highest output score becomes the chosen action.

## 10. Brain Weights And Biases

The brain stores:

```c
w1: input -> hidden weights
b1: hidden biases

w2: hidden -> output weights
b2: output biases
```

`w1` connects every input to every hidden neuron.

With 11 inputs and 16 hidden neurons:

```text
11 * 16 = 176 input-to-hidden weights
```

`w2` connects every hidden neuron to every output.

With 16 hidden neurons and 3 outputs:

```text
16 * 3 = 48 hidden-to-output weights
```

Biases are default tendencies for neurons. They let a neuron or output have a
baseline value before input signals affect it.

Simple summary:

```text
weights = how strongly signals matter
biases  = default preferences
```

## 11. Agent

An `Agent` wraps a brain with training metadata.

An agent stores:

- `Brain brain`
- `fitness`
- `score`
- `steps`
- `distance_reward`

The brain makes decisions. The agent stores how well that brain performed after
a game.

## 12. Fitness Function

Fitness is the score used by the genetic algorithm to rank agents.

The current idea is:

```text
fitness = score * 1000 + steps + distance_reward * 5
```

Food is weighted heavily because eating is more important than merely surviving.

Example:

```text
score = 3, steps = 120
fitness = 3 * 1000 + 120 = 3120

score = 0, steps = 300
fitness = 0 * 1000 + 300 = 300
```

This ensures a snake that eats food ranks higher than a snake that survives
without eating.

The distance reward gives smaller feedback for moving closer to food. It helps
when agents have not eaten yet but are behaving better than random movement.

## 13. Population

A `Population` contains many agents.

Each generation:

```text
1. evaluate every agent
2. assign fitness
3. find the best agent
4. create the next generation
```

Evaluation runs games without rendering so training can happen quickly.

## 14. Multi-Game Evaluation

At first, each agent played one game.

That was noisy because food placement is random:

```text
agent A might get easy food
agent B might get hard food
```

To reduce luck, each agent is evaluated multiple times and fitness is averaged.

This makes selection more stable because a good agent has to perform well across
several games, not just one lucky game.

## 15. Genetic Algorithm V1: Best Only

The first evolution strategy was:

```text
best agent survives unchanged
all other agents become mutated copies of the best agent
```

This is easy to understand, but it loses diversity quickly. The population can
become many small variations of one lucky brain.

## 16. Genetic Algorithm V2: Elites And Parent Pool

The second strategy added elitism and a parent pool.

Example with:

```text
POPULATION_SIZE = 100
ELITE_COUNT = 5
PARENT_POOL_SIZE = 20
```

The next generation is:

```text
agents 0-4:
  copied unchanged from the top 5 agents

agents 5-99:
  each chooses a random parent from agents 0-19
  copies that parent's brain
  mutates the copy
```

This preserves the best agents while keeping more variety than V1.

## 17. Genetic Algorithm V3: Crossover

The third strategy added crossover.

Instead of creating a child from one parent:

```text
child = copy parent
```

it creates a child from two parents:

```text
child = mix parent A and parent B
```

For each weight and bias, the child randomly inherits from either parent A or
parent B.

Then mutation is applied.

This can combine useful traits from different agents.

## 18. Adaptive Mutation

Adaptive mutation changes mutation settings based on progress.

If fitness improves:

```text
reduce mutation slightly
```

This helps fine-tune a good solution.

If fitness stalls for several generations:

```text
increase mutation
```

This helps the population explore more aggressively.

Example:

```text
mutation_rate = 0.05
mutation_strength = 0.20

new best fitness:
  mutation_rate = 0.05 * 0.95 = 0.0475
  mutation_strength = 0.20 * 0.95 = 0.19

stuck for too long:
  mutation_rate = 0.0475 * 1.25 = 0.059
  mutation_strength = 0.19 * 1.25 = 0.2375
```

Mutation is clamped between minimum and maximum values so it never becomes too
small to explore or too large to preserve useful behavior.

## 19. Saving And Loading Brains

The best brain can be saved to disk:

```text
out/best.brain
```

This allows:

- keeping the best result after closing the program
- continuing training from a previous best brain
- replaying a trained brain without retraining

The save/load format is binary and stores the `Brain` struct directly.

## 20. Replay Mode

Replay mode loads `out/best.brain` and watches it play without training.

Training mode:

```text
load saved brain into the population if available
train generations
save new best brains
```

Replay mode:

```text
load saved brain
watch it play
restart game when it dies
do not train
do not mutate
```

This makes it easy to compare training and replay behavior.

## 21. Window Score Display

Because plain SDL2 does not draw text by itself, the current score is shown in
the window title.

Drawing text inside the SDL window would require either:

- `SDL2_ttf`
- a custom bitmap font
- hand-drawn digit shapes

The window title keeps the project dependency-free.

## 22. Current Architecture

Current module responsibilities:

```text
main.c
  SDL setup and cleanup
  main loop
  input handling
  training/replay coordination

game.c / game.h
  snake rules
  game state
  movement
  collision
  food

render.c / render.h
  SDL drawing

brain.c / brain.h
  neural network
  decision making
  mutation
  crossover
  save/load

agent.c / agent.h
  brain plus fitness metadata

genetic.c / genetic.h
  population evaluation
  selection
  crossover
  mutation strategies
```

## 23. Why The Score Plateaus

If score plateaus around a value like 40, likely causes are:

- the brain input representation is too limited
- the network is too small
- the fitness function rewards local behavior too much
- max step or hunger limits stop long runs
- mutation settings are not exploring enough

This is normal in genetic algorithms. Plateaus are part of the search process.

## 24. Current Optimizations

Implemented optimization experiments:

- adaptive mutation fields are initialized during population setup
- normalized food distance inputs were added:

```text
food_dx = (food.x - head.x) / GRID_WIDTH
food_dy = (food.y - head.y) / GRID_HEIGHT
```

- hidden neurons were increased from 16 to 32
- `MAX_GAME_STEPS` and `MAX_STEPS_WITHOUT_FOOD` were increased
- fitness now uses squared score reward: `score * score * 1000`
- brain inputs include normalized wall and body distances for straight, left,
  and right movement
- command-line flags select training, replay, no-render training, generation
  count, brain file, CSV log file, and genetic strategy
- logs include strategy name, best fitness, average fitness, score, steps,
  distance reward, and mutation settings

Useful commands:

```text
./out/snake-ai --train
./out/snake-ai --replay
./out/snake-ai --train --strategy v1
./out/snake-ai --train --strategy v2
./out/snake-ai --train --strategy v3
./out/snake-ai --train --strategy v4
./out/snake-ai --train --strategy adaptive
./out/snake-ai --train --strategy adaptive-conservative
./out/snake-ai --no-render --generations 1000 --strategy adaptive
./out/snake-ai --no-render --generations 1000 --strategy adaptive-conservative --brain out/adaptive-conservative.brain --csv out/adaptive-conservative.csv
./out/snake-ai --no-render --generations 1000 --strategy v3 --brain out/v3.brain --csv out/v3.csv
```

Good next experiments:

- compare V1, V2, V3, V4, adaptive mutation, and conservative adaptive mutation
  over the same number of generations
- try parallel population evaluation after the strategy comparison is stable
- plot CSV logs to compare strategies visually
