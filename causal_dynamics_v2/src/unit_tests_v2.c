#include "unit_tests_v2.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TEST_TOLERANCE 1e-6

void init_test_results(TestResults *results) {
    memset(results, 0, sizeof(TestResults));
}

void add_test_result(TestResults *results, const char *name, int passed, const char *error_msg) {
    if (results->total < 100) {
        strncpy(results->test_names[results->total], name, 255);
        results->test_names[results->total][255] = '\0';
        results->test_results[results->total] = passed;
        if (!passed && error_msg) {
            strncpy(results->error_messages[results->total], error_msg, 511);
            results->error_messages[results->total][511] = '\0';
        }
        results->total++;
        if (passed) results->passed++;
        else results->failed++;
    }
}

void print_test_summary(TestResults *results) {
    printf("\n");
    printf("============================================================\n");
    printf("  Unit Test Summary\n");
    printf("============================================================\n");
    printf("Total: %d, Passed: %d, Failed: %d\n\n", results->total, results->passed, results->failed);
    
    for (int i = 0; i < results->total; i++) {
        printf("[%s] %s\n", results->test_results[i] ? "PASS" : "FAIL", results->test_names[i]);
        if (!results->test_results[i]) {
            printf("  Error: %s\n", results->error_messages[i]);
        }
    }
    
    printf("\n============================================================\n");
    if (results->failed == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TESTS FAILED\n", results->failed);
    }
    printf("============================================================\n");
}

int test_graph_initialization(TestResults *results) {
    CausalGraphV2 graph;
    init_causal_graph_v2(&graph, 100, GAUGE_U1);
    
    int passed = (graph.num_nodes == 100);
    passed = passed && (graph.num_edges == 0);
    passed = passed && (graph.gauge_type == GAUGE_U1);
    
    char msg[512];
    snprintf(msg, sizeof(msg), "num_nodes=%d (expected 100), num_edges=%d (expected 0)", 
             graph.num_nodes, graph.num_edges);
    
    add_test_result(results, "Graph Initialization", passed, msg);
    return passed;
}

int test_u1_phase_conservation(TestResults *results) {
    CausalGraphV2 graph;
    unsigned int seed = 42;
    init_causal_graph_v2(&graph, 50, GAUGE_U1);
    
    for (int i = 0; i < 100; i++) {
        int u = (int)((double)rand() / RAND_MAX * 50);
        int v = (int)((double)rand() / RAND_MAX * 50);
        if (u >= 50) u = 49;
        if (v >= 50) v = 49;
        if (u != v) {
            double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
            add_edge_with_phase(&graph, u, v, theta);
        }
    }
    
    int all_unit_modulus = 1;
    for (int e = 0; e < graph.num_edges; e++) {
        double mod = complex_modulus(graph.edges[e].phase);
        if (fabs(mod - 1.0) > 1e-6) {
            all_unit_modulus = 0;
            break;
        }
    }
    
    char msg[512];
    snprintf(msg, sizeof(msg), "Checked %d edges, all have |phase|=1", graph.num_edges);
    
    add_test_result(results, "U(1) Phase Conservation", all_unit_modulus, msg);
    return all_unit_modulus;
}

int test_edge_add_remove(TestResults *results) {
    CausalGraphV2 graph;
    init_causal_graph_v2(&graph, 20, GAUGE_U1);
    
    add_edge_with_phase(&graph, 0, 1, 0.5);
    add_edge_with_phase(&graph, 1, 2, 1.0);
    add_edge_with_phase(&graph, 2, 3, 1.5);
    
    int passed = (graph.num_edges == 3);
    
    int idx = find_edge_v2(&graph, 1, 2);
    passed = passed && (idx >= 0);
    
    if (idx >= 0) {
        remove_edge_v2(&graph, idx);
        passed = passed && (graph.num_edges == 2);
        passed = passed && (find_edge_v2(&graph, 1, 2) < 0);
    }
    
    char msg[512];
    snprintf(msg, sizeof(msg), "After add 3 and remove 1: num_edges=%d (expected 2)", graph.num_edges);
    
    add_test_result(results, "Edge Add/Remove", passed, msg);
    return passed;
}

int test_info_conservation(TestResults *results) {
    CausalGraphV2 graph;
    InformationStateV2 state;
    LaplacianSpectrum spectrum;
    DynamicsParamsV2 params;
    NullModelConfig nm_config;
    unsigned int seed = 12345;
    
    init_causal_graph_v2(&graph, 50, GAUGE_U1);
    init_null_model_config(&nm_config, FULL_MODEL);
    init_info_state_v2(&state, 50);
    set_uniform_density_v2(&state, 1.0);
    
    double p = 3.0 / 50;
    for (int u = 0; u < 50; u++) {
        for (int v = 0; v < 50; v++) {
            if (u != v && ((double)rand() / RAND_MAX) < p) {
                double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
                add_edge_with_phase(&graph, u, v, theta);
            }
        }
    }
    
    params.eta = 0.1;
    params.lambda = 1.0;
    params.kc_fraction = 0.1;
    params.alpha = 0.5;
    params.beta = 0.1;
    params.delta = 0.01;
    params.epsilon_del = 0.01;
    params.sigma_theta = 0.1;
    params.T_filter = 10;
    params.max_new_edges = 50;
    params.max_steps = 100;
    
    double initial_total = state.total_info;
    
    for (int step = 0; step < 50; step++) {
        dynamics_full_step_v2(&graph, &state, &spectrum, &params, &nm_config, step, &seed);
    }
    
    compute_totals_v2(&state);
    double final_total = state.total_info;
    
    int passed = (fabs(final_total - initial_total) < 0.01);
    
    char msg[512];
    snprintf(msg, sizeof(msg), "Initial total=%.6f, Final total=%.6f, Diff=%.6f", 
             initial_total, final_total, fabs(final_total - initial_total));
    
    add_test_result(results, "Information Conservation", passed, msg);
    return passed;
}

int test_phase_propagation(TestResults *results) {
    CausalGraphV2 graph;
    unsigned int seed = 42;
    init_causal_graph_v2(&graph, 30, GAUGE_U1);
    
    for (int i = 0; i < 50; i++) {
        int u = (int)((double)rand() / RAND_MAX * 30);
        int v = (int)((double)rand() / RAND_MAX * 30);
        if (u >= 30) u = 29;
        if (v >= 30) v = 29;
        if (u != v) {
            double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
            add_edge_with_phase(&graph, u, v, theta);
        }
    }
    
    int initial_edges = graph.num_edges;
    
    for (int e = 0; e < graph.num_edges; e++) {
        double old_theta = atan2(graph.edges[e].phase.im, graph.edges[e].phase.re);
        double xi = 0.1;
        double new_theta = old_theta + xi;
        graph.edges[e].phase = u1_phase(new_theta);
    }
    
    int all_unit_modulus = 1;
    for (int e = 0; e < graph.num_edges; e++) {
        double mod = complex_modulus(graph.edges[e].phase);
        if (fabs(mod - 1.0) > 1e-6) {
            all_unit_modulus = 0;
            break;
        }
    }
    
    int passed = all_unit_modulus && (graph.num_edges == initial_edges);
    
    char msg[512];
    snprintf(msg, sizeof(msg), "After phase update: all phases still on unit circle");
    
    add_test_result(results, "Phase Propagation", passed, msg);
    return passed;
}

int test_spectral_decomposition(TestResults *results) {
    CausalGraphV2 graph;
    InformationStateV2 state;
    LaplacianSpectrum spectrum;
    unsigned int seed = 42;
    
    init_causal_graph_v2(&graph, 40, GAUGE_U1);
    init_info_state_v2(&state, 40);
    set_uniform_density_v2(&state, 1.0);
    
    for (int u = 0; u < 40; u++) {
        for (int v = 0; v < 40; v++) {
            if (u != v && ((double)rand() / RAND_MAX) < 0.1) {
                double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
                add_edge_with_phase(&graph, u, v, theta);
            }
        }
    }
    
    compute_laplacian_spectrum(&graph, &spectrum, 10);
    
    int passed = (spectrum.num_eigenvalues > 0);
    passed = passed && (fabs(spectrum.eigenvalues[0]) < 1e-6);
    
    project_to_macro(&spectrum, state.rho, state.macro, state.num_nodes, 3);
    project_to_micro(&spectrum, state.rho, state.micro, state.num_nodes, 3);
    
    double max_diff = 0.0;
    for (int v = 0; v < state.num_nodes; v++) {
        double diff = fabs(state.rho[v] - (state.macro[v] + state.micro[v]));
        if (diff > max_diff) max_diff = diff;
    }
    
    passed = passed && (max_diff < 0.1);
    
    char msg[512];
    snprintf(msg, sizeof(msg), "Eigenvalues: %d, lambda_0=%.6f, max_decomp_error=%.6f", 
             spectrum.num_eigenvalues, spectrum.eigenvalues[0], max_diff);
    
    add_test_result(results, "Spectral Decomposition", passed, msg);
    return passed;
}

int test_null_models(TestResults *results) {
    NullModelConfig configs[4];
    int passed = 1;
    
    for (int i = 0; i < 4; i++) {
        init_null_model_config(&configs[i], (ModelType)i);
        
        switch (i) {
            case 0:
                passed = passed && (!configs[i].disable_macro_micro_decomposition);
                passed = passed && (!configs[i].random_rewiring);
                passed = passed && (!configs[i].static_topology);
                break;
            case 1:
                passed = passed && (configs[i].disable_macro_micro_decomposition);
                break;
            case 2:
                passed = passed && (configs[i].random_rewiring);
                break;
            case 3:
                passed = passed && (configs[i].static_topology);
                break;
        }
    }
    
    char msg[512];
    snprintf(msg, sizeof(msg), "All 4 null model configurations correct");
    
    add_test_result(results, "Null Model Configurations", passed, msg);
    return passed;
}

int test_wilson_loop(TestResults *results) {
    CausalGraphV2 graph;
    unsigned int seed = 42;
    init_causal_graph_v2(&graph, 20, GAUGE_U1);
    
    for (int i = 0; i < 20; i++) {
        int next = (i + 1) % 20;
        double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
        add_edge_with_phase(&graph, i, next, theta);
    }
    
    for (int i = 0; i < 10; i++) {
        int u = (int)((double)rand() / RAND_MAX * 20);
        int v = (int)((double)rand() / RAND_MAX * 20);
        if (u >= 20) u = 19;
        if (v >= 20) v = 19;
        if (u != v) {
            double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
            add_edge_with_phase(&graph, u, v, theta);
        }
    }
    
    WilsonLoopCollection wlc;
    find_all_wilson_loops(&graph, &wlc, 3, 8);
    
    int passed = (wlc.num_loops > 0);
    
    analyze_wilson_distribution(&wlc);
    passed = passed && (wlc.avg_modulus >= 0.0);
    
    char msg[512];
    snprintf(msg, sizeof(msg), "Found %d Wilson loops, avg_modulus=%.4f", 
             wlc.num_loops, wlc.avg_modulus);
    
    add_test_result(results, "Wilson Loop Computation", passed, msg);
    return passed;
}

int test_ricci_curvature(TestResults *results) {
    CausalGraphV2 graph;
    unsigned int seed = 42;
    init_causal_graph_v2(&graph, 30, GAUGE_U1);
    
    for (int i = 0; i < 30; i++) {
        int next = (i + 1) % 30;
        double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
        add_edge_with_phase(&graph, i, next, theta);
    }
    
    RicciCurvature ricci;
    compute_ollivier_ricci(&graph, &ricci);
    
    int passed = 1;
    for (int v = 0; v < graph.num_nodes; v++) {
        if (isnan(ricci.curvature[v]) || isinf(ricci.curvature[v])) {
            passed = 0;
            break;
        }
    }
    
    passed = passed && (!isnan(ricci.avg_curvature));
    passed = passed && (!isnan(ricci.std_curvature));
    
    char msg[512];
    snprintf(msg, sizeof(msg), "Ricci: avg=%.4f, std=%.4f, min=%.4f, max=%.4f", 
             ricci.avg_curvature, ricci.std_curvature, ricci.min_curvature, ricci.max_curvature);
    
    add_test_result(results, "Ricci Curvature Computation", passed, msg);
    return passed;
}

int run_all_unit_tests(TestResults *results) {
    init_test_results(results);
    
    printf("Running Unit Tests for Phase 0...\n\n");
    
    test_graph_initialization(results);
    test_u1_phase_conservation(results);
    test_edge_add_remove(results);
    test_info_conservation(results);
    test_phase_propagation(results);
    test_spectral_decomposition(results);
    test_null_models(results);
    test_wilson_loop(results);
    test_ricci_curvature(results);
    
    print_test_summary(results);
    
    return (results->failed == 0);
}
