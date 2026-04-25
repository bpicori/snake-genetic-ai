#include "config.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void app_config_print_usage(const char* program_name) {
  printf(
      "Usage: %s [--train|--replay] [--no-render] [--generations count] "
      "[--brain path] [--strategy v1|v2|v3|v4|adaptive|adaptive-conservative] [--seed value]\n",
      program_name);
  printf("  --train              train and periodically replay the best brain (default)\n");
  printf("  --replay             load the saved brain and replay without training\n");
  printf("  --no-render          train without opening an SDL window\n");
  printf("  --generations <n>    generations to train per batch, or total in --no-render mode\n");
  printf("  --brain <path>       brain file to load/save (default: out/best.brain)\n");
  printf("  --strategy <name>    choose v1, v2, v3, v4, adaptive, or adaptive-conservative\n");
  printf("  --seed <value>       use a fixed random seed for reproducible runs\n");
}

AppConfigParseResult app_config_parse(int argc, char* argv[], AppConfig* config) {
  config->replay_only = false;
  config->no_render = false;
  config->generations = DEFAULT_GENERATIONS;
  config->brain_path = DEFAULT_BRAIN_PATH;
  config->strategy = POPULATION_STRATEGY_ADAPTIVE_MUTATION;
  config->seed_provided = false;
  config->seed = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--train") == 0) {
      config->replay_only = false;
    } else if (strcmp(argv[i], "--replay") == 0) {
      config->replay_only = true;
    } else if (strcmp(argv[i], "--no-render") == 0) {
      config->no_render = true;
    } else if (strcmp(argv[i], "--generations") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing generation count.\n");
        app_config_print_usage(argv[0]);
        return APP_CONFIG_PARSE_ERROR;
      }

      config->generations = atoi(argv[i + 1]);
      if (config->generations <= 0) {
        fprintf(stderr, "Generation count must be greater than zero.\n");
        app_config_print_usage(argv[0]);
        return APP_CONFIG_PARSE_ERROR;
      }
      i++;
    } else if (strcmp(argv[i], "--brain") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing brain path.\n");
        app_config_print_usage(argv[0]);
        return APP_CONFIG_PARSE_ERROR;
      }

      config->brain_path = argv[i + 1];
      i++;
    } else if (strcmp(argv[i], "--strategy") == 0) {
      if (i + 1 >= argc || !population_strategy_from_name(argv[i + 1], &config->strategy)) {
        fprintf(stderr, "Invalid or missing strategy.\n");
        app_config_print_usage(argv[0]);
        return APP_CONFIG_PARSE_ERROR;
      }
      i++;
    } else if (strcmp(argv[i], "--seed") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing seed value.\n");
        app_config_print_usage(argv[0]);
        return APP_CONFIG_PARSE_ERROR;
      }

      char* end = NULL;
      errno = 0;
      unsigned long seed = strtoul(argv[i + 1], &end, 10);
      if (errno != 0 || end == argv[i + 1] || *end != '\0' || seed > UINT_MAX) {
        fprintf(stderr, "Seed must be an unsigned integer.\n");
        app_config_print_usage(argv[0]);
        return APP_CONFIG_PARSE_ERROR;
      }

      config->seed_provided = true;
      config->seed = (unsigned int)seed;
      i++;
    } else if (strcmp(argv[i], "--help") == 0) {
      app_config_print_usage(argv[0]);
      return APP_CONFIG_PARSE_HELP;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      app_config_print_usage(argv[0]);
      return APP_CONFIG_PARSE_ERROR;
    }
  }

  return APP_CONFIG_PARSE_OK;
}

void app_config_seed_random(const AppConfig* config) {
  unsigned int seed = config->seed_provided ? config->seed : (unsigned int)time(NULL);
  srand(seed);
  printf("Random seed: %u%s\n", seed, config->seed_provided ? " (fixed)" : "");
}

void app_config_print_summary(const AppConfig* config) {
  printf(
      "Mode: %s | strategy: %s | generations: %d | brain: %s\n", config->replay_only ? "replay" : "train",
      population_strategy_name(config->strategy), config->generations, config->brain_path);
}
