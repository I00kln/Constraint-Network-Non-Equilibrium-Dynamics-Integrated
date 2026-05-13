#ifndef NULL_MODELS_H
#define NULL_MODELS_H

#include "causal_graph_v2.h"

typedef enum {
    FULL_MODEL,
    NULL_MODEL_A,
    NULL_MODEL_B,
    NULL_MODEL_C
} ModelType;

typedef struct {
    ModelType type;
    int disable_macro_micro_decomposition;
    int random_rewiring;
    int static_topology;
    char description[256];
} NullModelConfig;

void init_null_model_config(NullModelConfig *config, ModelType type);
const char* model_type_name(ModelType type);

typedef struct {
    double rho[MAX_NODES];
    double macro[MAX_NODES];
    double micro[MAX_NODES];
    int num_nodes;
    double total_info;
    double total_macro;
    double total_micro;
} InformationStateV2;

void init_info_state_v2(InformationStateV2 *state, int num_nodes);
void set_uniform_density_v2(InformationStateV2 *state, double total);
void compute_totals_v2(InformationStateV2 *state);

#endif
