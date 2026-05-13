#ifndef UNIT_TESTS_V2_H
#define UNIT_TESTS_V2_H

#include "causal_graph_v2.h"
#include "null_models.h"
#include "dynamics_v2.h"
#include "wilson_loop.h"

typedef struct {
    int passed;
    int failed;
    int total;
    char test_names[100][256];
    int test_results[100];
    char error_messages[100][512];
} TestResults;

void init_test_results(TestResults *results);
void add_test_result(TestResults *results, const char *name, int passed, const char *error_msg);
void print_test_summary(TestResults *results);

int test_graph_initialization(TestResults *results);
int test_u1_phase_conservation(TestResults *results);
int test_edge_add_remove(TestResults *results);
int test_info_conservation(TestResults *results);
int test_phase_propagation(TestResults *results);
int test_spectral_decomposition(TestResults *results);
int test_null_models(TestResults *results);
int test_wilson_loop(TestResults *results);
int test_ricci_curvature(TestResults *results);

int run_all_unit_tests(TestResults *results);

#endif
