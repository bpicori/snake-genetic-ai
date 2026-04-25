#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>

#include "genetic.h"

#define DEFAULT_GENERATIONS 500
#define DEFAULT_BRAIN_PATH "out/best.brain"

typedef struct {
  bool replay_only;
  bool no_render;
  int generations;
  const char* brain_path;
  PopulationStrategy strategy;
  bool seed_provided;
  unsigned int seed;
} AppConfig;

typedef enum {
  APP_CONFIG_PARSE_OK,
  APP_CONFIG_PARSE_HELP,
  APP_CONFIG_PARSE_ERROR
} AppConfigParseResult;

void app_config_print_usage(const char* program_name);
AppConfigParseResult app_config_parse(int argc, char* argv[], AppConfig* config);
void app_config_seed_random(const AppConfig* config);
void app_config_print_summary(const AppConfig* config);

#endif
