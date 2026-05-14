#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "causal_graph_v2.h"
#include "null_models.h"
#include "dynamics_v2.h"
#include "wilson_loop.h"
#include "geometry_emergence.h"
#include "finite_size_scaling.h"
#include "unit_tests_v2.h"

static void print_banner(void) {
    printf("============================================================\n");
    printf("  Causal Information Dynamics Simulator V2.0\n");
    printf("  Spectral Coarse-Graining + Gauge Structure\n");
    printf("  Version 2.0 - May 2026\n");
    printf("============================================================\n\n");
}

static void print_usage(void) {
    printf("Usage: causal_sim_v2 <mode> [options]\n\n");
    printf("Modes:\n");
    printf("  single    Run single simulation\n");
    printf("  null      Run with null model comparison\n");
    printf("  phase     Generate phase diagram\n");
    printf("  scale     Finite size scaling analysis\n");
    printf("  test      Run unit tests\n\n");
    printf("Single mode options:\n");
    printf("  -N <num>        Number of nodes (default: 500)\n");
    printf("  -steps <num>    Max steps (default: 1000)\n");
    printf("  -eta <val>      Propagation rate (default: 0.1)\n");
    printf("  -lambda <val>   Micro compression (default: 1.0)\n");
    printf("  -kc <val>       Cutoff fraction (default: 0.1)\n");
    printf("  -alpha <val>    Rewire coeff (default: 0.5)\n");
    printf("  -sigma <val>    Phase noise (default: 0.1)\n");
    printf("  -seed <num>     Random seed (default: 12345)\n");
    printf("  -model <type>   Model type: 0=Full, 1=NM-A, 2=NM-B, 3=NM-C\n");
}

static int run_single_mode(int argc, char *argv[]) {
    int N = 500;
    int max_steps = 1000;
    double eta = 0.1;
    double lambda = 1.0;
    double kc = 0.1;
    double alpha = 0.5;
    double sigma_theta = 0.1;
    unsigned int seed = 12345;
    int model_type = 0;
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i+1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-steps") == 0 && i+1 < argc) max_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-eta") == 0 && i+1 < argc) eta = atof(argv[++i]);
        else if (strcmp(argv[i], "-lambda") == 0 && i+1 < argc) lambda = atof(argv[++i]);
        else if (strcmp(argv[i], "-kc") == 0 && i+1 < argc) kc = atof(argv[++i]);
        else if (strcmp(argv[i], "-alpha") == 0 && i+1 < argc) alpha = atof(argv[++i]);
        else if (strcmp(argv[i], "-sigma") == 0 && i+1 < argc) sigma_theta = atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-model") == 0 && i+1 < argc) model_type = atoi(argv[++i]);
    }
    
    printf("=== Single Simulation ===\n");
    printf("N = %d, eta = %.4f, lambda = %.4f, kc = %.4f\n", N, eta, lambda, kc);
    printf("Model type: %s\n\n", model_type_name((ModelType)model_type));
    
    CausalGraphV2 graph;
    InformationStateV2 state;
    LaplacianSpectrum spectrum;
    DynamicsParamsV2 params;
    NullModelConfig nm_config;
    
    init_causal_graph_v2(&graph, N, GAUGE_U1);
    init_null_model_config(&nm_config, (ModelType)model_type);
    init_info_state_v2(&state, N);
    set_uniform_density_v2(&state, 1.0);
    
    params.eta = eta;
    params.lambda = lambda;
    params.kc_fraction = kc;
    params.alpha = alpha;
    params.beta = 0.1;
    params.delta = 0.02;
    params.epsilon_del = 0.01;
    params.sigma_theta = sigma_theta;
    params.T_filter = 10;
    params.max_new_edges = 50;
    params.max_steps = max_steps;
    
    double p = 2.0 / N;
    generate_er_graph_v2(&graph, p, &seed);
    
    printf("Initial graph: %d nodes, %d edges\n", graph.num_nodes, graph.num_edges);
    printf("Starting dynamics...\n\n");
    
    static ObservablesV2 obs_history[2000];
    int history_len = 0;
    
    for (int step = 0; step < max_steps; step++) {
        dynamics_full_step_v2(&graph, &state, &spectrum, &params, &nm_config, step, &seed);
        
        if (step % 100 == 0 || step == max_steps - 1) {
            ObservablesV2 obs;
            compute_observables_v2(&graph, &state, &spectrum, &obs);
            obs_history[history_len++] = obs;
            
            printf("Step %d: entropy=%.6f, total=%.6f, edges=%d, clusters=%d\n",
                   step, obs.entropy, obs.total_info, obs.num_edges, obs.num_spectral_clusters);
        }
    }
    
    printf("\n=== Final Analysis ===\n");
    
    WilsonLoopCollection wlc;
    find_all_wilson_loops(&graph, &wlc, 3, 8);
    analyze_wilson_distribution(&wlc);
    int num_peaks = detect_gauge_peaks(&wlc, 0.3);
    
    printf("Wilson loops: %d found, avg_mod=%.4f, peaks=%d\n",
           wlc.num_loops, wlc.avg_modulus, num_peaks);
    
    RicciCurvature ricci;
    compute_ollivier_ricci(&graph, &ricci);
    printf("Ricci curvature: avg=%.4f, std=%.4f\n", ricci.avg_curvature, ricci.std_curvature);
    
    EinsteinRelation einstein;
    compute_einstein_relation(&ricci, &state, &einstein);
    printf("Einstein relation: slope=%.4f, R^2=%.4f\n",
           einstein.fit_result.slope, einstein.fit_result.r_squared);
    
    SpectralClusterResult sr;
    analyze_eigenvalue_clusters(&spectrum, &sr, 2.0);
    printf("Spectral clusters: %d, three_gen=%s\n",
           sr.num_clusters, sr.three_generation_detected ? "Yes" : "No");
    
    // 递归深度分析（三代结构）
    ThreeGenerationResult tg_result;
    analyze_three_generation_structure(&graph, &spectrum, &tg_result);
    printf("\n=== Three Generation Analysis ===\n");
    printf("Gen1 (depth=0): %d nodes (%.2f%%)\n", tg_result.gen1_nodes, tg_result.gen1_fraction * 100);
    printf("Gen2 (depth=1): %d nodes (%.2f%%)\n", tg_result.gen2_nodes, tg_result.gen2_fraction * 100);
    printf("Gen3 (depth=2): %d nodes (%.2f%%)\n", tg_result.gen3_nodes, tg_result.gen3_fraction * 100);
    printf("Three-gen detected: %s\n", tg_result.three_gen_detected ? "Yes" : "No");
    if (tg_result.three_gen_detected) {
        printf("Mass ratios: G2/G1=%.3f, G3/G2=%.3f\n", tg_result.mass_ratio_12, tg_result.mass_ratio_23);
    }
    
    // IPR分析（本征模局域化）
    double ipr_values[MAX_EIGENVECTORS];
    int localized_modes[MAX_EIGENVECTORS];
    analyze_eigenmode_localization(&spectrum, ipr_values, localized_modes, 0.1);
    printf("\n=== Eigenmode Localization (IPR) ===\n");
    int num_localized = 0;
    while (localized_modes[num_localized] >= 0) num_localized++;
    printf("Localized modes (IPR>0.1): %d/%d\n", num_localized, spectrum.num_eigenvalues);
    printf("Avg IPR: %.4f\n", ipr_values[0]);
    
    int converged = check_convergence(obs_history, history_len, 1e-4);
    printf("\nConverged: %s\n", converged ? "Yes" : "No");
    
    return 0;
}

static int run_null_comparison(int argc, char *argv[]) {
    printf("=== Null Model Comparison ===\n\n");
    
    int N = 500;
    int max_steps = 1000;
    unsigned int seed = 12345;
    
    double results[4][4];
    const char *model_names[] = {"Full Model", "NM-A", "NM-B", "NM-C"};
    
    for (int m = 0; m < 4; m++) {
        printf("Running %s...\n", model_names[m]);
        
        CausalGraphV2 graph;
        InformationStateV2 state;
        LaplacianSpectrum spectrum;
        DynamicsParamsV2 params;
        NullModelConfig nm_config;
        
        init_causal_graph_v2(&graph, N, GAUGE_U1);
        init_null_model_config(&nm_config, (ModelType)m);
        init_info_state_v2(&state, N);
        set_uniform_density_v2(&state, 1.0);
        
        params.eta = 0.1;
        params.lambda = 1.0;
        params.kc_fraction = 0.1;
        params.alpha = 0.5;
        params.beta = 0.1;
        params.delta = 0.02;
        params.epsilon_del = 0.01;
        params.sigma_theta = 0.1;
        params.T_filter = 10;
        params.max_new_edges = 50;
        params.max_steps = max_steps;
        
        generate_er_graph_v2(&graph, 2.0/N, &seed);
        
        for (int step = 0; step < max_steps; step++) {
            dynamics_full_step_v2(&graph, &state, &spectrum, &params, &nm_config, step, &seed);
        }
        
        ObservablesV2 obs;
        compute_observables_v2(&graph, &state, &spectrum, &obs);
        
        WilsonLoopCollection wlc;
        find_all_wilson_loops(&graph, &wlc, 3, 8);
        analyze_wilson_distribution(&wlc);
        int peaks = detect_gauge_peaks(&wlc, 0.3);
        
        SpectralClusterResult sr;
        analyze_eigenvalue_clusters(&spectrum, &sr, 2.0);
        
        results[m][0] = obs.num_spectral_clusters;
        results[m][1] = (double)peaks;
        results[m][2] = wlc.avg_modulus;
        results[m][3] = sr.three_generation_detected ? 1.0 : 0.0;
    }
    
    printf("\n=== Comparison Results ===\n");
    printf("%-12s %10s %10s %10s %10s\n", "Model", "Clusters", "W-Peaks", "W-Mod", "3-Gen");
    printf("%-12s %10s %10s %10s %10s\n", "------", "--------", "-------", "-----", "-----");
    
    for (int m = 0; m < 4; m++) {
        printf("%-12s %10.1f %10.1f %10.4f %10.1f\n",
               model_names[m], results[m][0], results[m][1], results[m][2], results[m][3]);
    }
    
    printf("\n=== Signal Validation ===\n");
    
    SignalValidation sv1, sv2, sv3;
    validate_signal(&sv1, "Spectral Clusters", results[0][0], results[1][0], results[2][0], results[3][0], 2.0);
    validate_signal(&sv2, "Wilson Peaks", results[0][1], results[1][1], results[2][1], results[3][1], 1.0);
    validate_signal(&sv3, "Three Generations", results[0][3], results[1][3], results[2][3], results[3][3], 0.5);
    
    print_validation_summary(&sv1);
    printf("\n");
    print_validation_summary(&sv2);
    printf("\n");
    print_validation_summary(&sv3);
    
    return 0;
}

static int run_test_mode(void) {
    printf("=== Running Unit Tests (Phase 0) ===\n\n");
    
    TestResults results;
    int all_passed = run_all_unit_tests(&results);
    
    return all_passed ? 0 : 1;
}

int main(int argc, char *argv[]) {
    print_banner();
    
    if (argc < 2) {
        print_usage();
        return 0;
    }
    
    if (strcmp(argv[1], "single") == 0) {
        return run_single_mode(argc, argv);
    } else if (strcmp(argv[1], "null") == 0) {
        return run_null_comparison(argc, argv);
    } else if (strcmp(argv[1], "test") == 0) {
        return run_test_mode();
    } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage();
        return 0;
    } else {
        printf("Unknown mode: %s\n", argv[1]);
        print_usage();
        return 1;
    }
    
    return 0;
}
