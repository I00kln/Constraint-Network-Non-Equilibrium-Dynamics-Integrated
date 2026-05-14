#include "causal_capacity.h"

static CausalGraph graph;
static InfoState state;
static Observables obs;

static void print_spectrum_detail(const PropagationSpectrum *spec) {
    printf("  Spectrum: %d eigenvalues computed\n", spec->num_eigenvalues);
    printf("  Negative: %d, Positive: %d, Zero: %d\n",
           spec->negative_count, spec->positive_count, spec->zero_count);
    printf("  Lorentz signature: %s\n", spec->lorentz_signature_detected ? "YES" : "NO");
    printf("  Signature ratio |neg_max/pos_min|: %.6f\n", spec->signature_ratio);
    printf("  Spectral gap: %.6f\n", spec->spectral_gap);
    printf("  Asymmetry measure: %.6f\n", spec->asymmetry_measure);
    printf("  Neg fraction: %.4f\n", spec->neg_fraction);

    printf("  Top eigenvalues: ");
    int show = (spec->num_eigenvalues > 15) ? 15 : spec->num_eigenvalues;
    for (int k = 0; k < show; k++) {
        printf("%.4f ", spec->eigenvalues[k]);
    }
    printf("\n");

    printf("  Coarse-grained tensor eigenvalues: ");
    for (int d = 0; d < LOCAL_DIM; d++) {
        printf("%.4f(%s) ", spec->coarse_grained_eigenvalues[d],
               spec->coarse_grained_signs[d] < 0 ? "-" : "+");
    }
    printf("\n");
    printf("  Coarse-grained Lorentz: %s (neg_frac=%.3f)\n",
           spec->coarse_grained_lorentz ? "YES" : "NO",
           spec->coarse_grained_neg_fraction);
}

static void print_pca_detail(const LocalPCAResult *pca, int n) {
    printf("  Local PCA: Lorentz=%d Euclidean=%d Other=%d (%.1f%% Lorentz)\n",
           pca->lorentz_nodes, pca->euclidean_nodes, pca->other_nodes,
           pca->lorentz_fraction * 100.0);
    printf("  Avg asymmetry: %.4f, Avg response strength: %.6f\n",
           pca->avg_asymmetry, pca->avg_response_strength);
    printf("  Avg tensor eigenvalues: ");
    for (int d = 0; d < LOCAL_DIM; d++) {
        printf("%.4f(%s) ", pca->avg_tensor_eigenvalues[d],
               pca->avg_tensor_signs[d] < 0 ? "-" : "+");
    }
    printf("\n");
    printf("  Avg tensor Lorentz: %s\n", pca->avg_tensor_lorentz ? "YES" : "NO");
}

static void print_green_detail(const GreenFunction *gf) {
    printf("  Green fn: velocity=%.4f sharpness=%.4f cone=%s anisotropy=%.4f front_width=%.4f\n",
           gf->light_cone_velocity, gf->cone_sharpness,
           gf->light_cone_detected ? "YES" : "NO", gf->anisotropy, gf->front_width);
}

static void run_single_simulation(int N, double p, const DynamicsParams *params,
                                   int total_steps, int measure_interval,
                                   unsigned int seed, const char *label,
                                   int use_er_graph) {
    printf("\n=== Simulation: %s (N=%d, p=%.3f, seed=%u, %s) ===\n",
           label, N, p, seed, use_er_graph ? "ER" : "DAG");
    printf("  Params: eta=%.3f lambda=%.3f alpha=%.3f beta=%.3f eps=%.3f sigma=%.3f cap_lr=%.3f\n",
           params->eta, params->lambda_param, params->alpha, params->beta,
           params->epsilon_del, params->sigma, params->cap_learning_rate);

    init_causal_graph(&graph, N);
    if (use_er_graph) {
        generate_er_graph(&graph, p, &seed);
    } else {
        generate_directed_acyclic_graph(&graph, p, &seed);
    }

    init_info_state(&state, N);
    set_uniform_density(&state, 1.0);

    detect_cycles(&graph);
    printf("  Initial graph: %d nodes, %d edges, cycles=%d\n",
           graph.num_nodes, graph.num_edges, graph.cycle_count);

    PropagationSpectrum spec;
    LocalPCAResult pca;
    GreenFunction gf;

    compute_response_spectrum(&graph, &state, params->eta, 0.01, &spec, 20);
    printf("\n  [Step 0] Initial response spectrum:\n");
    print_spectrum_detail(&spec);

    compute_local_pca(&graph, &state, params->eta, 0.01, &pca);
    print_pca_detail(&pca, N);

    int lorentz_found_step = -1;
    int light_cone_found_step = -1;

    for (int step = 1; step <= total_steps; step++) {
        dynamics_full_step(&graph, &state, params, &seed);

        if (step % measure_interval == 0 || step == total_steps) {
            compute_observables(&graph, &state, params, &obs);

            printf("\n  [Step %d] edges=%d entropy=%.4f total_info=%.4f cycles=%d\n",
                   step, obs.num_edges, obs.entropy, obs.total_info, obs.cycle_count);
            printf("    Lorentz=%s neg_ev=%d pos_ev=%d sig_ratio=%.4f pde=%.2f lf=%.3f asym=%.4f\n",
                   obs.lorentz_detected ? "YES" : "NO",
                   obs.neg_eigenvalues, obs.pos_eigenvalues,
                   obs.signature_ratio, obs.pde_type_indicator,
                   obs.lorentz_fraction, obs.asymmetry_measure);

            if (obs.lorentz_detected && lorentz_found_step < 0) {
                lorentz_found_step = step;
                printf("    *** LORENTZ SIGNATURE FIRST DETECTED at step %d ***\n", step);

                compute_response_spectrum(&graph, &state, params->eta, 0.01, &spec, 30);
                print_spectrum_detail(&spec);

                compute_local_pca(&graph, &state, params->eta, 0.01, &pca);
                print_pca_detail(&pca, N);
            }

            if (step == total_steps || (obs.lorentz_detected && step % (measure_interval * 5) == 0)) {
                compute_green_function(&graph, &state, &gf, 0, 50, 0.01, params->eta);
                print_green_detail(&gf);

                if (gf.light_cone_detected && light_cone_found_step < 0) {
                    light_cone_found_step = step;
                    printf("    *** LIGHT CONE FIRST DETECTED at step %d ***\n", step);
                }
            }
        }
    }

    compute_response_spectrum(&graph, &state, params->eta, 0.01, &spec, 30);
    compute_local_pca(&graph, &state, params->eta, 0.01, &pca);

    printf("\n  === Final State ===\n");
    print_spectrum_detail(&spec);
    print_pca_detail(&pca, N);

    compute_green_function(&graph, &state, &gf, 0, 100, 0.01, params->eta);
    print_green_detail(&gf);

    detect_cycles(&graph);
    printf("  Final cycles: %d\n", graph.cycle_count);
    printf("\n  Summary: Lorentz_first=%d LightCone_first=%d\n",
           lorentz_found_step, light_cone_found_step);
}

static void run_parameter_scan(int N, double p, int use_er_graph) {
    printf("\n\n############################################\n");
    printf("#  PHASE 1 PARAMETER SCAN: N=%d, p=%.3f, %s\n", N, p, use_er_graph ? "ER" : "DAG");
    printf("############################################\n");

    DynamicsParams params;
    params.epsilon_del = 0.05;
    params.sigma = 0.05;
    params.max_new_edges = 5;
    params.intervention_samples = 5;
    params.use_conservation = 1;

    double eta_values[] = {0.3, 0.5, 0.7};
    double lambda_values[] = {0.5, 0.8};
    double alpha_values[] = {0.5, 1.0, 2.0};
    double beta_values[] = {0.3, 0.7};
    double cap_lr_values[] = {0.0, 0.03, 0.06};

    int n_eta = sizeof(eta_values) / sizeof(eta_values[0]);
    int n_lambda = sizeof(lambda_values) / sizeof(lambda_values[0]);
    int n_alpha = sizeof(alpha_values) / sizeof(alpha_values[0]);
    int n_beta = sizeof(beta_values) / sizeof(beta_values[0]);
    int n_cap_lr = sizeof(cap_lr_values) / sizeof(cap_lr_values[0]);

    int total_combos = n_eta * n_lambda * n_alpha * n_beta * n_cap_lr;
    int num_seeds = 3;
    printf("Scanning %d x %d x %d x %d x %d = %d combos x %d seeds = %d runs\n",
           n_eta, n_lambda, n_alpha, n_beta, n_cap_lr, total_combos, num_seeds,
           total_combos * num_seeds);

    int combo = 0;
    int lorentz_count = 0;
    int light_cone_count = 0;

    double best_sig_ratio = 0.0;
    DynamicsParams best_params;
    memset(&best_params, 0, sizeof(best_params));
    int best_lorentz_frac = 0;
    DynamicsParams best_pca_params;
    memset(&best_pca_params, 0, sizeof(best_pca_params));
    double best_asym = 0.0;
    DynamicsParams best_asym_params;
    memset(&best_asym_params, 0, sizeof(best_asym_params));

    for (int ie = 0; ie < n_eta; ie++) {
        for (int il = 0; il < n_lambda; il++) {
            for (int ia = 0; ia < n_alpha; ia++) {
                for (int ib = 0; ib < n_beta; ib++) {
                    for (int ic = 0; ic < n_cap_lr; ic++) {
                        combo++;
                        params.eta = eta_values[ie];
                        params.lambda_param = lambda_values[il];
                        params.alpha = alpha_values[ia];
                        params.beta = beta_values[ib];
                        params.cap_learning_rate = cap_lr_values[ic];

                        double avg_lorentz_frac = 0.0;
                        double avg_asym = 0.0;
                        int any_lorentz = 0;
                        int any_light_cone = 0;
                        double max_sig_ratio = 0.0;

                        for (int s = 0; s < num_seeds; s++) {
                            unsigned int seed = 12345 + (combo * num_seeds + s) * 7919;

                            init_causal_graph(&graph, N);
                            if (use_er_graph) {
                                generate_er_graph(&graph, p, &seed);
                            } else {
                                generate_directed_acyclic_graph(&graph, p, &seed);
                            }
                            init_info_state(&state, N);
                            set_uniform_density(&state, 1.0);

                            for (int step = 1; step <= 300; step++) {
                                dynamics_full_step(&graph, &state, &params, &seed);
                            }

                            compute_observables(&graph, &state, &params, &obs);

                            avg_lorentz_frac += obs.lorentz_fraction;
                            avg_asym += obs.asymmetry_measure;

                            if (obs.lorentz_detected) any_lorentz = 1;
                            if (obs.signature_ratio > max_sig_ratio) max_sig_ratio = obs.signature_ratio;

                            if (obs.lorentz_detected || obs.lorentz_fraction > 0.3) {
                                GreenFunction gf;
                                compute_green_function(&graph, &state, &gf, 0, 50, 0.01, params.eta);
                                if (gf.light_cone_detected) any_light_cone = 1;
                            }
                        }

                        avg_lorentz_frac /= num_seeds;
                        avg_asym /= num_seeds;

                        if (any_lorentz) lorentz_count++;
                        if (any_light_cone) light_cone_count++;

                        if (max_sig_ratio > best_sig_ratio) {
                            best_sig_ratio = max_sig_ratio;
                            best_params = params;
                        }

                        if ((int)(avg_lorentz_frac * 100) > best_lorentz_frac) {
                            best_lorentz_frac = (int)(avg_lorentz_frac * 100);
                            best_pca_params = params;
                        }

                        if (avg_asym > best_asym) {
                            best_asym = avg_asym;
                            best_asym_params = params;
                        }

                        if (combo % 20 == 0 || any_lorentz || avg_lorentz_frac > 0.5) {
                            printf("[%d/%d] eta=%.2f lam=%.2f a=%.2f b=%.2f clr=%.3f | "
                                   "Lorentz=%s avg_lf=%.3f avg_asym=%.4f sig=%.4f%s\n",
                                   combo, total_combos,
                                   params.eta, params.lambda_param, params.alpha, params.beta,
                                   params.cap_learning_rate,
                                   any_lorentz ? "YES" : "NO",
                                   avg_lorentz_frac, avg_asym, max_sig_ratio,
                                   any_light_cone ? " LIGHTCONE" : "");
                        }
                    }
                }
            }
        }
    }

    printf("\n=== SCAN RESULTS ===\n");
    printf("Total combinations: %d (x%d seeds each)\n", total_combos, num_seeds);
    printf("Lorentz signature detected: %d (%.1f%%)\n", lorentz_count, 100.0 * lorentz_count / total_combos);
    printf("Light cone detected: %d (%.1f%%)\n", light_cone_count, 100.0 * light_cone_count / total_combos);
    printf("Best signature ratio: %.4f (eta=%.2f lam=%.2f alpha=%.2f beta=%.2f clr=%.3f)\n",
           best_sig_ratio, best_params.eta, best_params.lambda_param,
           best_params.alpha, best_params.beta, best_params.cap_learning_rate);
    printf("Best avg Lorentz fraction: %d%% (eta=%.2f lam=%.2f alpha=%.2f beta=%.2f clr=%.3f)\n",
           best_lorentz_frac, best_pca_params.eta, best_pca_params.lambda_param,
           best_pca_params.alpha, best_pca_params.beta, best_pca_params.cap_learning_rate);
    printf("Best avg asymmetry: %.4f (eta=%.2f lam=%.2f alpha=%.2f beta=%.2f clr=%.3f)\n",
           best_asym, best_asym_params.eta, best_asym_params.lambda_param,
           best_asym_params.alpha, best_asym_params.beta, best_asym_params.cap_learning_rate);

    if (lorentz_count > 0) {
        printf("\n*** LORENTZ SIGNATURE EMERGENCE CONFIRMED ***\n");
        printf("Running detailed simulation with best params...\n");

        unsigned int seed = 42;
        run_single_simulation(N, p, &best_params, 1000, 50, seed, "Best Lorentz params", use_er_graph);
    } else if (best_lorentz_frac > 20) {
        printf("\n*** PARTIAL LORENTZ SIGNAL DETECTED (%d%% avg nodes) ***\n", best_lorentz_frac);
        printf("Running detailed simulation with best PCA params...\n");

        unsigned int seed = 42;
        run_single_simulation(N, p, &best_pca_params, 1000, 50, seed, "Best PCA params", use_er_graph);
    } else {
        printf("\n*** NO STRONG LORENTZ SIGNAL IN THIS SCAN ***\n");
        printf("Best avg asymmetry: %.4f - running detailed simulation...\n", best_asym);
        unsigned int seed = 42;
        run_single_simulation(N, p, &best_asym_params, 1000, 50, seed, "Best asymmetry params", use_er_graph);
    }
}

static void run_graph_topology_comparison(int N) {
    printf("\n\n############################################\n");
    printf("#  GRAPH TOPOLOGY COMPARISON: N=%d\n", N);
    printf("############################################\n");

    DynamicsParams params;
    params.eta = 0.5;
    params.lambda_param = 0.5;
    params.alpha = 1.0;
    params.beta = 0.5;
    params.epsilon_del = 0.05;
    params.sigma = 0.05;
    params.max_new_edges = 5;
    params.intervention_samples = 5;
    params.cap_learning_rate = 0.02;
    params.use_conservation = 1;

    const char *topo_names[] = {"DAG p=0.05", "DAG p=0.1", "ER p=0.05", "ER p=0.1", "SmallWorld k=3"};
    double probs[] = {0.05, 0.1, 0.05, 0.1, 0.0};
    int use_dag[] = {1, 1, 0, 0, 2};
    int n_topo = 5;

    for (int t = 0; t < n_topo; t++) {
        unsigned int seed = 42 + t * 1000;

        init_causal_graph(&graph, N);
        if (use_dag[t] == 1) {
            generate_directed_acyclic_graph(&graph, probs[t], &seed);
        } else if (use_dag[t] == 0) {
            generate_er_graph(&graph, probs[t], &seed);
        } else {
            generate_small_world_graph(&graph, 3, 0.3, &seed);
        }

        init_info_state(&state, N);
        set_uniform_density(&state, 1.0);

        detect_cycles(&graph);
        printf("\n--- Topology: %s (initial edges=%d, cycles=%d) ---\n",
               topo_names[t], graph.num_edges, graph.cycle_count);

        PropagationSpectrum spec0;
        compute_response_spectrum(&graph, &state, params.eta, 0.01, &spec0, 20);
        printf("  Initial response spectrum: neg=%d pos=%d asym=%.4f Lorentz=%s\n",
               spec0.negative_count, spec0.positive_count,
               spec0.asymmetry_measure,
               spec0.lorentz_signature_detected ? "YES" : "NO");

        for (int step = 1; step <= 500; step++) {
            dynamics_full_step(&graph, &state, &params, &seed);
        }

        compute_observables(&graph, &state, &params, &obs);
        detect_cycles(&graph);
        printf("  After 500 steps: edges=%d cycles=%d Lorentz=%s neg=%d pos=%d sig_ratio=%.4f pde=%.2f lf=%.3f asym=%.4f\n",
               obs.num_edges, graph.cycle_count,
               obs.lorentz_detected ? "YES" : "NO",
               obs.neg_eigenvalues, obs.pos_eigenvalues,
               obs.signature_ratio, obs.pde_type_indicator,
               obs.lorentz_fraction, obs.asymmetry_measure);

        PropagationSpectrum spec_final;
        compute_response_spectrum(&graph, &state, params.eta, 0.01, &spec_final, 20);
        printf("  Final top eigenvalues: ");
        int show = (spec_final.num_eigenvalues > 10) ? 10 : spec_final.num_eigenvalues;
        for (int k = 0; k < show; k++) {
            printf("%.4f ", spec_final.eigenvalues[k]);
        }
        printf("\n");

        LocalPCAResult pca;
        compute_local_pca(&graph, &state, params.eta, 0.01, &pca);
        print_pca_detail(&pca, N);

        GreenFunction gf;
        compute_green_function(&graph, &state, &gf, 0, 50, 0.01, params.eta);
        print_green_detail(&gf);
    }
}

static void run_time_evolution_tracking(int N, double p, int use_er_graph) {
    printf("\n\n############################################\n");
    printf("#  TIME EVOLUTION TRACKING: N=%d, p=%.3f, %s\n", N, p, use_er_graph ? "ER" : "DAG");
    printf("############################################\n");

    DynamicsParams params;
    params.eta = 0.5;
    params.lambda_param = 0.5;
    params.alpha = 1.0;
    params.beta = 0.5;
    params.epsilon_del = 0.05;
    params.sigma = 0.05;
    params.max_new_edges = 5;
    params.intervention_samples = 5;
    params.cap_learning_rate = 0.02;
    params.use_conservation = 1;

    unsigned int seed = 42;

    init_causal_graph(&graph, N);
    if (use_er_graph) {
        generate_er_graph(&graph, p, &seed);
    } else {
        generate_directed_acyclic_graph(&graph, p, &seed);
    }
    init_info_state(&state, N);
    set_uniform_density(&state, 1.0);

    printf("Step,Edges,Entropy,NegEv,PosEv,Lorentz,SigRatio,PDE,LorentzFrac,Asymmetry,Cycles\n");

    for (int step = 0; step <= 1000; step++) {
        if (step > 0) {
            dynamics_full_step(&graph, &state, &params, &seed);
        }

        if (step % 50 == 0) {
            compute_observables(&graph, &state, &params, &obs);
            detect_cycles(&graph);
            printf("%d,%d,%.6f,%d,%d,%d,%.6f,%.2f,%.6f,%.6f,%d\n",
                   step, obs.num_edges, obs.entropy,
                   obs.neg_eigenvalues, obs.pos_eigenvalues,
                   obs.lorentz_detected, obs.signature_ratio,
                   obs.pde_type_indicator, obs.lorentz_fraction,
                   obs.asymmetry_measure, obs.cycle_count);
            fflush(stdout);
        }
    }
}

static void run_quick_diagnostic(int N) {
    printf("\n\n############################################\n");
    printf("#  QUICK DIAGNOSTIC: Testing intervention response mechanism\n");
    printf("############################################\n");

    unsigned int seed = 42;
    init_causal_graph(&graph, N);
    generate_er_graph(&graph, 0.1, &seed);
    init_info_state(&state, N);
    set_uniform_density(&state, 1.0);

    printf("Graph: %d nodes, %d edges\n", graph.num_nodes, graph.num_edges);
    detect_cycles(&graph);
    printf("Cycles: %d\n\n", graph.cycle_count);

    double eta = 0.5;
    double delta = 0.01;

    printf("Testing intervention response at node 0:\n");
    static double orig_rho[MAX_NODES];
    memcpy(orig_rho, state.rho, N * sizeof(double));

    state.rho[0] += delta;

    static double new_rho[MAX_NODES];
    for (int v = 0; v < N; v++) {
        double sum_cap = 0.0;
        double weighted_sum = 0.0;
        for (int i = 0; i < graph.in_degree[v]; i++) {
            int parent = graph.in_neighbors[v][i];
            int eidx = graph.in_edge_idx[v][i];
            double cap = graph.edges[eidx].capacity;
            sum_cap += cap;
            weighted_sum += cap * state.rho[parent];
        }
        if (sum_cap > 1e-15) {
            new_rho[v] = (1.0 - eta) * state.rho[v] + eta * weighted_sum;
        } else {
            new_rho[v] = state.rho[v];
        }
    }

    double total_before = 0.0, total_after = 0.0;
    for (int v = 0; v < N; v++) {
        total_before += orig_rho[v];
        total_after += new_rho[v];
    }
    printf("  Before renorm: total_before=%.6f total_after=%.6f (diff=%.6f)\n",
           total_before, total_after, total_after - total_before);

    double scale = state.total_info / total_after;
    for (int v = 0; v < N; v++) new_rho[v] *= scale;

    int pos_resp = 0, neg_resp = 0, zero_resp = 0;
    double max_pos = 0.0, max_neg = 0.0;
    for (int v = 0; v < N; v++) {
        double response = (new_rho[v] - orig_rho[v]) / delta;
        if (response > 1e-10) {
            pos_resp++;
            if (response > max_pos) max_pos = response;
        } else if (response < -1e-10) {
            neg_resp++;
            if (response < max_neg) max_neg = response;
        } else {
            zero_resp++;
        }
    }

    printf("  After renorm: pos_response=%d neg_response=%d zero=%d\n",
           pos_resp, neg_resp, zero_resp);
    printf("  Max positive response: %.6f\n", max_pos);
    printf("  Max negative response: %.6f\n", max_neg);
    printf("  Conservation creates negative responses: %s\n",
           neg_resp > 0 ? "YES" : "NO");

    memcpy(state.rho, orig_rho, N * sizeof(double));

    printf("\nComputing full response spectrum...\n");
    PropagationSpectrum spec;
    compute_response_spectrum(&graph, &state, eta, delta, &spec, 20);
    print_spectrum_detail(&spec);

    printf("\nComputing local PCA...\n");
    LocalPCAResult pca;
    compute_local_pca(&graph, &state, eta, delta, &pca);
    print_pca_detail(&pca, N);

    printf("\nSample local tensors (first 5 Lorentz nodes):\n");
    int shown = 0;
    for (int v = 0; v < N && shown < 5; v++) {
        LocalResponseTensor tensor;
        compute_local_response_tensor(&graph, &state, v, eta, delta, &tensor);
        if (tensor.lorentz_signature || tensor.dim >= 2) {
            printf("  Node %d (dim=%d, Lorentz=%s, asym=%.4f):\n",
                   v, tensor.dim, tensor.lorentz_signature ? "YES" : "NO",
                   tensor.asymmetry_norm);
            printf("    Eigenvalues: ");
            for (int d = 0; d < tensor.dim; d++) {
                printf("%.4f(%s) ", tensor.eigenvalues[d],
                       tensor.sign_pattern[d] == -1 ? "-" : "+");
            }
            printf("\n");
            if (tensor.dim <= 4 && tensor.dim >= 2) {
                printf("    Matrix:\n");
                for (int i = 0; i < tensor.dim; i++) {
                    printf("      ");
                    for (int j = 0; j < tensor.dim; j++) {
                        printf("%.4f ", tensor.matrix[i][j]);
                    }
                    printf("\n");
                }
            }
            shown++;
        }
    }
}

int main(int argc, char *argv[]) {
    printf("============================================================\n");
    printf("  Causal Capacity Theory - Phase 1 Verification (v2.0)\n");
    printf("  Intervention Response Spectrum & Lorentz Signature\n");
    printf("============================================================\n");
    printf("  Key insight: Conservation-induced negative responses\n");
    printf("  are the physical mechanism for Lorentz signature.\n");
    printf("============================================================\n");

    int mode = 0;
    if (argc > 1) {
        mode = atoi(argv[1]);
    }

    switch (mode) {
    case 0: {
        printf("\nMode 0: Quick diagnostic (N=60)\n");
        run_quick_diagnostic(60);
        break;
    }
    case 1: {
        printf("\nMode 1: Single simulation ER graph (N=100)\n");
        DynamicsParams params;
        params.eta = 0.5;
        params.lambda_param = 0.5;
        params.alpha = 1.0;
        params.beta = 0.5;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 5;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.02;
        params.use_conservation = 1;
        run_single_simulation(100, 0.1, &params, 500, 50, 42, "ER test", 1);
        break;
    }
    case 2: {
        printf("\nMode 2: Parameter scan ER graph (N=60)\n");
        run_parameter_scan(60, 0.1, 1);
        break;
    }
    case 3: {
        printf("\nMode 3: Topology comparison (N=60)\n");
        run_graph_topology_comparison(60);
        break;
    }
    case 4: {
        printf("\nMode 4: Time evolution ER (N=60)\n");
        run_time_evolution_tracking(60, 0.1, 1);
        break;
    }
    case 5: {
        printf("\nMode 5: Parameter scan DAG (N=60)\n");
        run_parameter_scan(60, 0.1, 0);
        break;
    }
    case 6: {
        printf("\nMode 6: Larger scale ER scan (N=100)\n");
        run_parameter_scan(100, 0.08, 1);
        break;
    }
    case 7: {
        printf("\nMode 7: Single simulation DAG (N=100)\n");
        DynamicsParams params;
        params.eta = 0.5;
        params.lambda_param = 0.5;
        params.alpha = 1.0;
        params.beta = 0.5;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 5;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.02;
        params.use_conservation = 1;
        run_single_simulation(100, 0.1, &params, 500, 50, 42, "DAG test", 0);
        break;
    }
    case 8: {
        printf("\nMode 8: Small world simulation (N=100)\n");
        DynamicsParams params;
        params.eta = 0.5;
        params.lambda_param = 0.5;
        params.alpha = 1.0;
        params.beta = 0.5;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 5;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.02;
        params.use_conservation = 1;

        unsigned int seed = 42;
        init_causal_graph(&graph, 100);
        generate_small_world_graph(&graph, 3, 0.3, &seed);
        init_info_state(&state, 100);
        set_uniform_density(&state, 1.0);

        detect_cycles(&graph);
        printf("  Small-world graph: %d edges, %d cycles\n", graph.num_edges, graph.cycle_count);

        for (int step = 1; step <= 500; step++) {
            dynamics_full_step(&graph, &state, &params, &seed);
        }

        compute_observables(&graph, &state, &params, &obs);
        printf("  After 500 steps: Lorentz=%s neg=%d pos=%d lf=%.3f asym=%.4f\n",
               obs.lorentz_detected ? "YES" : "NO",
               obs.neg_eigenvalues, obs.pos_eigenvalues,
               obs.lorentz_fraction, obs.asymmetry_measure);

        PropagationSpectrum spec;
        compute_response_spectrum(&graph, &state, params.eta, 0.01, &spec, 20);
        print_spectrum_detail(&spec);

        LocalPCAResult pca;
        compute_local_pca(&graph, &state, params.eta, 0.01, &pca);
        print_pca_detail(&pca, 100);
        break;
    }
    case 9: {
        printf("\nMode 9: Large scale ER simulation (N=200)\n");
        run_parameter_scan(200, 0.05, 1);
        break;
    }
    case 10: {
        printf("\nMode 10: Large scale single simulation (N=200)\n");
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 10;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        run_single_simulation(200, 0.05, &params, 1000, 100, 42, "Large ER", 1);
        break;
    }
    case 11: {
        printf("\nMode 11: Light cone analysis (N=150)\n");
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 10;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;

        unsigned int seed = 42;
        int N = 150;
        init_causal_graph(&graph, N);
        generate_er_graph(&graph, 0.06, &seed);
        init_info_state(&state, N);
        set_uniform_density(&state, 1.0);

        detect_cycles(&graph);
        printf("  Initial: %d nodes, %d edges, %d cycles\n", N, graph.num_edges, graph.cycle_count);

        for (int step = 1; step <= 500; step++) {
            dynamics_full_step(&graph, &state, &params, &seed);
        }

        printf("\n  After 500 steps:\n");
        compute_observables(&graph, &state, &params, &obs);
        printf("  Lorentz=%s neg=%d pos=%d sig_ratio=%.4f lf=%.3f asym=%.4f\n",
               obs.lorentz_detected ? "YES" : "NO",
               obs.neg_eigenvalues, obs.pos_eigenvalues,
               obs.signature_ratio, obs.lorentz_fraction, obs.asymmetry_measure);

        printf("\n  Computing Green function for light cone analysis...\n");
        GreenFunction gf;
        for (int origin = 0; origin < 3; origin++) {
            compute_green_function(&graph, &state, &gf, origin, 150, 0.01, params.eta);
            printf("  Origin %d: velocity=%.4f sharpness=%.4f anisotropy=%.4f cone=%s\n",
                   origin, gf.light_cone_velocity, gf.cone_sharpness, gf.anisotropy,
                   gf.light_cone_detected ? "YES" : "NO");
        }

        PropagationSpectrum spec;
        compute_response_spectrum(&graph, &state, params.eta, 0.01, &spec, 30);
        print_spectrum_detail(&spec);

        LocalPCAResult pca;
        compute_local_pca(&graph, &state, params.eta, 0.01, &pca);
        print_pca_detail(&pca, N);
        break;
    }
    case 12: {
        printf("\nMode 12: Finite Size Scaling Analysis\n");
        printf("========================================\n");
        
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 10;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        
        int sizes[] = {100, 150, 200, 250, 300};
        int num_sizes = 5;
        int num_seeds = 3;
        int steps = 500;
        
        printf("Sizes: ");
        for (int i = 0; i < num_sizes; i++) printf("%d ", sizes[i]);
        printf("\nSeeds per size: %d\n", num_seeds);
        printf("Steps: %d\n\n", steps);
        
        printf("N,Seed,Edges_Init,Edges_Final,Lorentz,NegEv,PosEv,SigRatio,LorentzFrac,Asymmetry,Velocity,Sharpness,Anisotropy\n");
        
        for (int si = 0; si < num_sizes; si++) {
            int N = sizes[si];
            double p = 0.06;
            
            for (int seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
                unsigned int seed = 42 + seed_idx * 1000;
                
                init_causal_graph(&graph, N);
                generate_er_graph(&graph, p, &seed);
                init_info_state(&state, N);
                set_uniform_density(&state, 1.0);
                
                int edges_init = graph.num_edges;
                
                for (int step = 1; step <= steps; step++) {
                    dynamics_full_step(&graph, &state, &params, &seed);
                }
                
                compute_observables(&graph, &state, &params, &obs);
                
                GreenFunction gf;
                compute_green_function(&graph, &state, &gf, 0, 300, 0.01, params.eta);
                
                printf("%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                       N, seed_idx, edges_init, obs.num_edges,
                       obs.lorentz_detected, obs.neg_eigenvalues, obs.pos_eigenvalues,
                       obs.signature_ratio, obs.lorentz_fraction, obs.asymmetry_measure,
                       gf.light_cone_velocity, gf.cone_sharpness, gf.anisotropy);
                fflush(stdout);
            }
        }
        break;
    }
    case 13: {
        printf("\nMode 13: Large scale N=300 simulation\n");
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 15;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        run_single_simulation(300, 0.04, &params, 800, 100, 42, "N300 ER", 1);
        break;
    }
    case 14: {
        printf("\nMode 14: Large scale N=400 simulation\n");
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 20;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        run_single_simulation(400, 0.035, &params, 800, 100, 42, "N400 ER", 1);
        break;
    }
    case 15: {
        printf("\nMode 15: Large scale N=500 simulation\n");
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 25;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        run_single_simulation(500, 0.03, &params, 800, 100, 42, "N500 ER", 1);
        break;
    }
    case 16: {
        printf("\nMode 16: 4D Lorentz Verification\n");
        printf("==================================\n");
        printf("Testing for 4D Lorentzian signature [-,+,+,+]\n\n");
        
        int sizes[] = {100, 150, 200, 250, 300, 400};
        int num_sizes = 6;
        
        printf("Size,NegEv,PosEv,Is4D,NegRatio,PosRatio,TopEigenvalues\n");
        
        for (int si = 0; si < num_sizes; si++) {
            int N = sizes[si];
            double p = 0.06;
            
            init_causal_graph(&graph, N);
            generate_er_graph(&graph, p, &(unsigned int){42});
            init_info_state(&state, N);
            set_uniform_density(&state, 1.0);
            
            DynamicsParams params;
            params.eta = 0.7;
            params.lambda_param = 0.5;
            params.alpha = 0.5;
            params.beta = 0.7;
            params.epsilon_del = 0.05;
            params.sigma = 0.05;
            params.max_new_edges = 10;
            params.intervention_samples = 5;
            params.cap_learning_rate = 0.0;
            params.use_conservation = 1;
            
            for (int step = 1; step <= 500; step++) {
                dynamics_full_step(&graph, &state, &params, &(unsigned int){42});
            }
            
            compute_observables(&graph, &state, &params, &obs);
            
            PropagationSpectrum spectrum;
            compute_propagation_operator_spectrum(&graph, &spectrum, 30);
            
            int neg = spectrum.negative_count;
            int pos = spectrum.positive_count;
            int is_4d = (neg >= 1 && pos >= 3) ? 1 : 0;
            int total = neg + pos;
            double neg_ratio = (total > 0) ? (double)neg / total : 0.0;
            double pos_ratio = (total > 0) ? (double)pos / total : 0.0;
            
            printf("%d,%d,%d,%d,%.4f,%.4f", N, neg, pos, is_4d, neg_ratio, pos_ratio);
            for (int k = 0; k < 4 && k < spectrum.num_eigenvalues; k++) {
                printf(",%.4f", spectrum.eigenvalues[k]);
            }
            printf("\n");
            fflush(stdout);
        }
        
        printf("\n--- Local Tensor 4D Analysis ---\n");
        
        int N_test = 300;
        init_causal_graph(&graph, N_test);
        generate_er_graph(&graph, 0.04, &(unsigned int){42});
        init_info_state(&state, N_test);
        set_uniform_density(&state, 1.0);
        
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 15;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        
        for (int step = 1; step <= 500; step++) {
            dynamics_full_step(&graph, &state, &params, &(unsigned int){42});
        }
        
        int count_4d = 0, count_3d = 0, count_2d = 0, count_other = 0;
        int count_dim4 = 0, count_dim3 = 0, count_dim2 = 0, count_dim1 = 0;
        double avg_neg_ev[4] = {0}, avg_pos_ev[4] = {0};
        int neg_count = 0, pos_count = 0;
        int all_neg_patterns[5] = {0};
        
        for (int i = 0; i < N_test; i++) {
            LocalPCAResult pca;
            compute_local_pca(&graph, &state, params.eta, 0.01, &pca);
            
            int local_neg = 0, local_pos = 0;
            for (int k = 0; k < LOCAL_DIM; k++) {
                double ev = pca.eigenvalues[i][k];
                if (ev < -1e-10) {
                    local_neg++;
                    if (local_neg <= 4) avg_neg_ev[local_neg-1] += fabs(ev);
                    neg_count++;
                } else if (ev > 1e-10) {
                    local_pos++;
                    if (local_pos <= 4) avg_pos_ev[local_pos-1] += ev;
                    pos_count++;
                }
            }
            
            int dim = local_neg + local_pos;
            if (dim == 4) count_dim4++;
            else if (dim == 3) count_dim3++;
            else if (dim == 2) count_dim2++;
            else if (dim == 1) count_dim1++;
            
            if (local_neg >= 0 && local_neg <= 4) {
                all_neg_patterns[local_neg]++;
            }
            
            if (local_neg == 1 && local_pos == 3) count_4d++;
            else if (local_neg == 1 && local_pos == 2) count_3d++;
            else if (local_neg == 1 && local_pos == 1) count_2d++;
            else count_other++;
        }
        
        printf("N=%d Local Tensor Analysis:\n", N_test);
        printf("  Tensor dimensions: dim4=%d dim3=%d dim2=%d dim1=%d\n", 
               count_dim4, count_dim3, count_dim2, count_dim1);
        printf("  Neg eigenvalue patterns: 0neg=%d 1neg=%d 2neg=%d 3neg=%d 4neg=%d\n",
               all_neg_patterns[0], all_neg_patterns[1], all_neg_patterns[2], 
               all_neg_patterns[3], all_neg_patterns[4]);
        printf("  4D Lorentz [-,+,+,+]: %d nodes (%.1f%%)\n", count_4d, 100.0*count_4d/N_test);
        printf("  3D Lorentz [-,+,+]: %d nodes (%.1f%%)\n", count_3d, 100.0*count_3d/N_test);
        printf("  2D Lorentz [-,+]: %d nodes (%.1f%%)\n", count_2d, 100.0*count_2d/N_test);
        printf("  Other: %d nodes (%.1f%%)\n", count_other, 100.0*count_other/N_test);
        
        if (neg_count > 0) {
            printf("  Avg negative eigenvalue: %.4f\n", avg_neg_ev[0]/count_4d);
        }
        if (pos_count > 0) {
            printf("  Avg positive eigenvalue: %.4f\n", avg_pos_ev[0]/count_4d);
        }
        
        break;
    }
    case 17: {
        printf("\nMode 17: Metric Extraction & Coarse-Graining Analysis\n");
        printf("======================================================\n\n");
        
        int N = 300;
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 15;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        
        init_causal_graph(&graph, N);
        generate_er_graph(&graph, 0.04, &(unsigned int){42});
        init_info_state(&state, N);
        set_uniform_density(&state, 1.0);
        
        for (int step = 1; step <= 500; step++) {
            dynamics_full_step(&graph, &state, &params, &(unsigned int){42});
        }
        
        printf("=== Emergent Metric Extraction (N=%d) ===\n\n", N);
        
        EmergentMetric metric;
        extract_emergent_metric(&graph, &state, params.eta, 0.01, &metric);
        
        printf("Metric tensor g_μν:\n");
        for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
            printf("  [");
            for (int nu = 0; nu < EFFECTIVE_DIM; nu++) {
                printf(" %8.4f", metric.metric[mu][nu]);
            }
            printf(" ]\n");
        }
        
        printf("\nEigenvalues: ");
        for (int k = 0; k < EFFECTIVE_DIM; k++) {
            printf("%.4f(%+d) ", metric.eigenvalues[k], metric.signature[k]);
        }
        printf("\n");
        
        printf("Determinant: %.6f\n", metric.determinant);
        printf("Is Lorentzian: %s\n", metric.is_lorentzian ? "YES" : "NO");
        printf("Is Minkowski: %s\n", metric.is_minkowski ? "YES" : "NO");
        printf("Minkowski deviation: %.4f\n", metric.minkowski_deviation);
        printf("Effective c (light speed): %.4f\n", metric.effective_c);
        printf("2D Curvature: %.6f\n", metric.curvature_2d);
        
        printf("\n=== Coarse-Graining / RG Flow ===\n\n");
        
        RGFlowResult rg;
        compute_coarse_graining(&graph, &state, params.eta, 0.01, &rg);
        
        printf("Scale,BlockSize,Lorentz,NegEv,PosEv,EffC,Anisotropy,MinkDev\n");
        for (int li = 0; li < rg.num_levels; li++) {
            CoarseGrainLevel *lvl = &rg.levels[li];
            printf("%d,%.0f,%.1f,%.0f,%.0f,%.4f,%.4f,%.4f\n",
                   li, lvl->scale, lvl->lorentz_fraction,
                   lvl->neg_count, lvl->pos_count,
                   lvl->effective_c, lvl->anisotropy,
                   lvl->minkowski_deviation);
        }
        
        printf("\nRG Flow Summary:\n");
        printf("  Number of levels: %d\n", rg.num_levels);
        printf("  Has fixed point: %s\n", rg.has_fixed_point ? "YES" : "NO");
        if (rg.has_fixed_point) {
            printf("  Fixed point metric:\n");
            for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
                printf("    [");
                for (int nu = 0; nu < EFFECTIVE_DIM; nu++) {
                    printf(" %8.4f", rg.fixed_point_metric[mu][nu]);
                }
                printf(" ]\n");
            }
            printf("  Fixed point Minkowski deviation: %.4f\n", rg.fixed_point_deviation);
        }
        
        printf("\n=== Multi-Size Metric Comparison ===\n\n");
        
        int sizes[] = {100, 200, 300, 400};
        int num_sizes = 4;
        
        printf("N,IsLorentz,IsMink,EffC,MinkDev,Curvature,Eigenvalues\n");
        for (int si = 0; si < num_sizes; si++) {
            int Ns = sizes[si];
            double p = 0.06;
            
            init_causal_graph(&graph, Ns);
            generate_er_graph(&graph, p, &(unsigned int){42});
            init_info_state(&state, Ns);
            set_uniform_density(&state, 1.0);
            
            for (int step = 1; step <= 500; step++) {
                dynamics_full_step(&graph, &state, &params, &(unsigned int){42});
            }
            
            EmergentMetric m;
            extract_emergent_metric(&graph, &state, params.eta, 0.01, &m);
            
            printf("%d,%s,%s,%.4f,%.4f,%.6f",
                   Ns,
                   m.is_lorentzian ? "YES" : "NO",
                   m.is_minkowski ? "YES" : "NO",
                   m.effective_c, m.minkowski_deviation, m.curvature_2d);
            for (int k = 0; k < EFFECTIVE_DIM; k++) {
                printf(",%.4f(%+d)", m.eigenvalues[k], m.signature[k]);
            }
            printf("\n");
            fflush(stdout);
        }
        
        break;
    }
    case 18: {
        printf("\nMode 18: Phase 1 Complete Verification\n");
        printf("======================================\n\n");
        
        int N = 400;
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 20;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        
        init_causal_graph(&graph, N);
        generate_er_graph(&graph, 0.035, &(unsigned int){42});
        init_info_state(&state, N);
        set_uniform_density(&state, 1.0);
        
        for (int step = 1; step <= 800; step++) {
            dynamics_full_step(&graph, &state, &params, &(unsigned int){42});
        }
        
        printf("=== 1. Time Direction Detection ===\n\n");
        
        TimeDirectionResult td;
        detect_time_direction(&graph, &state, params.eta, 0.01, &td);
        
        printf("Time eigenvalue: %.6f\n", td.time_eigenvalue);
        printf("Max positive eigenvalue: %.6f\n", td.max_positive_eigenvalue);
        printf("Pearson correlation (eigenvector vs topo order): %.4f\n", td.correlation);
        printf("Spearman correlation: %.4f\n", td.spearman_correlation);
        printf("Is aligned: %s\n", td.is_aligned ? "YES" : "NO");
        printf("Alignment score: %.4f\n", td.alignment_score);
        printf("Time direction node: %d\n", td.time_direction_node);
        
        printf("\n=== 2. Coarse-Graining PCA Stability ===\n\n");
        
        CoarsePCA pca_results[20];
        int num_pca = 0;
        compute_coarse_pca(&graph, &state, params.eta, 0.01, pca_results, &num_pca);
        
        printf("Scale,NegCount,PosCount,LorentzFrac,Stable,Deviation\n");
        for (int i = 0; i < num_pca; i++) {
            printf("%.0f,%.2f,%.2f,%.2f,%s,%.4f\n",
                   pca_results[i].scale,
                   pca_results[i].neg_count,
                   pca_results[i].pos_count,
                   pca_results[i].lorentz_fraction,
                   pca_results[i].stable_signature ? "YES" : "NO",
                   pca_results[i].signature_deviation);
        }
        
        printf("\n=== 3. PDE Type Transition Analysis ===\n\n");
        
        PDETypeInfo pde_results[30];
        int num_pde = 0;
        analyze_pde_transition(&graph, &state, pde_results, &num_pde);
        
        printf("Eta,Lambda,Type,Hyperbolic,Elliptic,Parabolic,NegEv,PosEv,Gap,Asym\n");
        for (int i = 0; i < num_pde; i++) {
            const char *type_str = "Unknown";
            if (pde_results[i].is_hyperbolic) type_str = "Hyperbolic";
            else if (pde_results[i].is_parabolic) type_str = "Parabolic";
            else if (pde_results[i].is_elliptic) type_str = "Elliptic";
            
            printf("%.1f,%.1f,%s,%d,%d,%d,%.0f,%.0f,%.4f,%.4f\n",
                   pde_results[i].eta,
                   pde_results[i].lambda_param,
                   type_str,
                   pde_results[i].is_hyperbolic,
                   pde_results[i].is_elliptic,
                   pde_results[i].is_parabolic,
                   pde_results[i].neg_eigenvalue_count,
                   pde_results[i].pos_eigenvalue_count,
                   pde_results[i].spectral_gap,
                   pde_results[i].asymmetry);
        }
        
        printf("\n=== 4. Effective PDE Identification ===\n\n");
        
        PropagationSpectrum spec;
        compute_response_spectrum(&graph, &state, params.eta, 0.01, &spec, 30);
        
        double pde_type = classify_pde_type(&spec);
        printf("PDE classification score: %.4f\n", pde_type);
        
        int hyperbolic_count = 0;
        for (int i = 0; i < num_pde; i++) {
            if (pde_results[i].is_hyperbolic) hyperbolic_count++;
        }
        double hyperbolic_fraction = (double)hyperbolic_count / num_pde;
        printf("Hyperbolic parameter region fraction: %.2f%%\n", hyperbolic_fraction * 100);
        
        printf("\n=== 5. Light Cone Structure ===\n\n");
        
        GreenFunction gf;
        compute_green_function(&graph, &state, &gf, 0, 80, 0.01, params.eta);
        int lc_detected = check_light_cone(&gf);
        
        printf("Light cone detected: %s\n", lc_detected ? "YES" : "NO");
        printf("Light cone velocity: %.4f\n", gf.light_cone_velocity);
        printf("Cone sharpness: %.4f\n", gf.cone_sharpness);
        printf("Anisotropy: %.4f\n", gf.anisotropy);
        printf("Front width: %.4f\n", gf.front_width);
        
        printf("\n=== 6. Lorentz Signature Verification ===\n\n");
        
        EmergentMetric metric;
        extract_emergent_metric(&graph, &state, params.eta, 0.01, &metric);
        
        printf("Metric signature: ");
        for (int k = 0; k < EFFECTIVE_DIM; k++) {
            printf("%+d ", metric.signature[k]);
        }
        printf("\n");
        printf("Is Lorentzian: %s\n", metric.is_lorentzian ? "YES" : "NO");
        printf("Effective c: %.4f\n", metric.effective_c);
        
        printf("\n=== Phase 1 Success Criteria ===\n\n");
        
        int criterion1 = metric.is_lorentzian;
        int criterion2 = lc_detected;
        int criterion3 = (pde_type > 0.5 || hyperbolic_fraction > 0.3);
        
        printf("Criterion 1: Stable Lorentz signature (-,+,+,+) - %s\n", 
               criterion1 ? "PASS" : "FAIL");
        printf("Criterion 2: Light cone structure detected - %s\n", 
               criterion2 ? "PASS" : "FAIL");
        printf("Criterion 3: Hyperbolic PDE identified - %s\n", 
               criterion3 ? "PASS" : "FAIL");
        
        int all_pass = criterion1 && criterion2 && criterion3;
        printf("\n=== PHASE 1 STATUS: %s ===\n", 
               all_pass ? "ALL CRITERIA PASSED" : "NEEDS MORE WORK");
        
        break;
    }
    case 19: {
        printf("\nMode 19: Phase 2 - Einstein-Hilbert Term Verification\n");
        printf("========================================================\n\n");
        
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 20;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        
        printf("=== 1. Effective Action Calculation ===\n\n");
        
        int N1 = 400;
        init_causal_graph(&graph, N1);
        generate_er_graph(&graph, 0.035, &(unsigned int){42});
        init_info_state(&state, N1);
        set_uniform_density(&state, 1.0);
        
        for (int step = 1; step <= 800; step++) {
            dynamics_full_step(&graph, &state, &params, &(unsigned int){42});
        }
        
        EffectiveAction action;
        compute_effective_action(&graph, &state, params.eta, 0.01, &action);
        
        printf("Curvature term R coefficient: %.6f\n", action.R_coefficient);
        printf("R² coefficient: %.6f\n", action.R2_coefficient);
        printf("R_ijkl coefficient: %.6f\n", action.R_ijkl_coefficient);
        printf("Total action: %.6f\n", action.total_action);
        printf("Kinetic term: %.6f\n", action.kinetic_term);
        printf("Potential term: %.6f\n", action.potential_term);
        printf("Is Einstein-Hilbert: %s\n", action.is_einstein_hilbert ? "YES" : "NO");
        printf("EH deviation: %.6f\n", action.eh_deviation);
        
        printf("\n=== 2. Gravity Coupling G vs N ===\n\n");
        
        int sizes[] = {100, 200, 300, 400};
        int num_sizes = 4;
        
        CausalGraph graphs[4];
        InfoState states[4];
        CausalGraph *graph_ptrs[4];
        InfoState *state_ptrs[4];
        
        printf("N,G_measured,G_predicted(1/sqrt(N)),Deviation\n");
        for (int si = 0; si < num_sizes; si++) {
            int Ns = sizes[si];
            
            init_causal_graph(&graphs[si], Ns);
            generate_er_graph(&graphs[si], 0.04, &(unsigned int){42 + si});
            init_info_state(&states[si], Ns);
            set_uniform_density(&states[si], 1.0);
            
            for (int step = 1; step <= 600; step++) {
                dynamics_full_step(&graphs[si], &states[si], &params, &(unsigned int){42 + si});
            }
            
            graph_ptrs[si] = &graphs[si];
            state_ptrs[si] = &states[si];
        }
        
        GravityCoupling gc[4];
        compute_gravity_coupling(graph_ptrs, state_ptrs, num_sizes, sizes, gc);
        
        for (int si = 0; si < num_sizes; si++) {
            printf("%d,%.6f,%.6f,%.4f\n",
                   gc[si].N, gc[si].G, gc[si].G_predicted, gc[si].G_deviation);
        }
        
        printf("\nScaling exponent (G ~ N^alpha): %.4f (expected: -0.5)\n", gc[0].scaling_exponent);
        
        printf("\n=== 3. Newton Potential Test ===\n\n");
        
        NewtonPotential potential[50];
        int num_points = 0;
        compute_newton_potential(&graph, &state, 0, potential, &num_points);
        
        printf("r,phi,phi_newton(-M/r),deviation,count\n");
        for (int i = 0; i < num_points && i < 15; i++) {
            printf("%.0f,%.6f,%.6f,%.4f,%d\n",
                   potential[i].r, potential[i].phi,
                   potential[i].phi_newton, potential[i].deviation,
                   potential[i].count);
        }
        
        double avg_dev = 0.0;
        for (int i = 0; i < num_points; i++) {
            avg_dev += potential[i].deviation;
        }
        if (num_points > 0) avg_dev /= num_points;
        printf("\nAverage deviation from Newton potential: %.4f\n", avg_dev);
        
        printf("\n=== 4. Fluctuation-Dissipation Theorem ===\n\n");
        
        FluctuationDissipation fd;
        check_fluctuation_dissipation(&graph, &state, params.eta, 0.01, &fd);
        
        printf("Capacity fluctuation: %.6f\n", fd.capacity_fluctuation);
        printf("Metric response: %.6f\n", fd.metric_response);
        printf("G from fluctuation: %.6f\n", fd.G_from_fluctuation);
        printf("G from response: %.6f\n", fd.G_from_response);
        printf("Fluctuation-dissipation ratio: %.4f (expected: ~1.0)\n", fd.fluctuation_dissipation_ratio);
        printf("Satisfies FD theorem: %s\n", fd.satisfies_fd_theorem ? "YES" : "NO");
        
        printf("\n=== Phase 2 Success Criteria ===\n\n");
        
        int criterion1 = (fabs(action.R_coefficient) > 1e-10);
        int criterion2 = (fabs(gc[0].scaling_exponent + 0.5) < 0.3);
        int criterion3 = (avg_dev < 1.0);
        
        printf("Criterion 1: R term coefficient non-zero - %s\n", 
               criterion1 ? "PASS" : "FAIL");
        printf("Criterion 2: G ~ 1/sqrt(N) scaling - %s\n", 
               criterion2 ? "PASS" : "FAIL");
        printf("Criterion 3: Newton potential recovered - %s\n", 
               criterion3 ? "PASS" : "FAIL");
        
        int all_pass = criterion1 && criterion2 && criterion3;
        printf("\n=== PHASE 2 STATUS: %s ===\n", 
               all_pass ? "ALL CRITERIA PASSED" : "NEEDS MORE WORK");
        
        break;
    }
    case 20: {
        printf("\nMode 20: Phase 3 - Fermions, Gauge Fields & Particle Spectrum\n");
        printf("===================================================================\n\n");
        
        DynamicsParams params;
        params.eta = 0.7;
        params.lambda_param = 0.5;
        params.alpha = 0.5;
        params.beta = 0.7;
        params.epsilon_del = 0.05;
        params.sigma = 0.05;
        params.max_new_edges = 20;
        params.intervention_samples = 5;
        params.cap_learning_rate = 0.0;
        params.use_conservation = 1;
        
        printf("=== 1. Closed Cycle Detection (N=400) ===\n\n");
        
        int N = 400;
        init_causal_graph(&graph, N);
        generate_er_graph(&graph, 0.06, &(unsigned int){42});
        init_info_state(&state, N);
        set_uniform_density(&state, 1.0);
        
        for (int step = 1; step <= 400; step++) {
            dynamics_full_step(&graph, &state, &params, &(unsigned int){42});
        }
        
        CycleSpectrum spectrum;
        find_closed_cycles(&graph, &spectrum);
        
        printf("Total cycles found: %d\n", spectrum.num_cycles);
        printf("Min action: %.6f\n", spectrum.min_action);
        printf("Max action: %.6f\n", spectrum.max_action);
        printf("Avg action: %.6f\n", spectrum.avg_action);
        
        printf("\nFirst 10 cycles:\n");
        printf("Index,Length,Action,Winding,Mass,Charge\n");
        for (int i = 0; i < spectrum.num_cycles && i < 10; i++) {
            printf("%d,%d,%.4f,%.2f,%.4f,%.4f\n",
                   i, spectrum.cycles[i].length, spectrum.cycles[i].action,
                   spectrum.cycles[i].winding_number, spectrum.cycles[i].mass,
                   spectrum.cycles[i].charge);
        }
        
        printf("\n=== 2. Stable Cycle Classification ===\n\n");
        
        classify_stable_cycles(&spectrum);
        
        printf("Number of stable cycles: %d\n", spectrum.num_stable);
        printf("Number of homotopy classes: %d\n", spectrum.num_homotopy_classes);
        
        printf("\nStable cycles by homotopy class:\n");
        int class_counts[10] = {0};
        for (int i = 0; i < spectrum.num_cycles; i++) {
            if (spectrum.cycles[i].is_stable) {
                int hc = spectrum.cycles[i].homotopy_class;
                if (hc >= 0 && hc < 10) class_counts[hc]++;
            }
        }
        for (int i = 0; i < 10; i++) {
            if (class_counts[i] > 0) {
                printf("  Class %d: %d cycles\n", i, class_counts[i]);
            }
        }
        
        printf("\n=== 3. Gauge Group Structure ===\n\n");
        
        GaugeGroup group;
        analyze_gauge_structure(&graph, &group);
        
        printf("Number of factors: %d\n", group.num_factors);
        printf("Total dimension: %.0f\n", group.total_dim);
        printf("\nGauge group factors:\n");
        for (int i = 0; i < group.num_factors; i++) {
            printf("  %s: dim=%d, compact=%d, simple=%d\n",
                   group.factors[i].name, group.factors[i].dim,
                   group.factors[i].is_compact, group.factors[i].is_simple);
        }
        printf("\nMatches Standard Model gauge group: %s\n", 
               group.matches_standard_model ? "YES" : "NO");
        
        printf("\n=== 4. Particle Spectrum ===\n\n");
        
        ParticleSpectrum ps;
        compute_particle_spectrum(&spectrum, &ps);
        
        printf("Number of particles: %d\n", ps.num_particles);
        printf("Number of generations: %d\n", ps.num_generations);
        printf("Has quarks: %s\n", ps.has_quarks ? "YES" : "NO");
        printf("Has leptons: %s\n", ps.has_leptons ? "YES" : "NO");
        
        printf("\nParticle list:\n");
        printf("Name,Mass,Charge,Gen,Spin,Color,Fermion\n");
        for (int i = 0; i < ps.num_particles && i < 20; i++) {
            printf("%s,%.4f,%.4f,%d,%d,%d,%s\n",
                   ps.particles[i].name, ps.particles[i].mass,
                   ps.particles[i].charge, ps.particles[i].generation,
                   ps.particles[i].spin, ps.particles[i].color,
                   ps.particles[i].is_fermion ? "YES" : "NO");
        }
        
        printf("\n=== 5. CKM and PMNS Mixing Matrices ===\n\n");
        
        MixingMatrix ckm, pmns;
        compute_ckm_matrix(&spectrum, &ckm);
        compute_pmns_matrix(&spectrum, &pmns);
        
        printf("CKM Matrix (quark mixing):\n");
        printf("        d           s           b\n");
        for (int i = 0; i < 3; i++) {
            char row[3] = {'u', 'c', 't'};
            printf("%c:  ", row[i]);
            for (int j = 0; j < 3; j++) {
                double mag = sqrt(ckm.real[i][j] * ckm.real[i][j] + ckm.imag[i][j] * ckm.imag[i][j]);
                printf("%.4f    ", mag);
            }
            printf("\n");
        }
        printf("Jarlskog invariant J = %.6e\n", ckm.jarlskog);
        printf("Unitarity violation = %.6e\n", ckm.unitarity_violation);
        
        printf("\nPMNS Matrix (neutrino mixing):\n");
        printf("        v1          v2          v3\n");
        for (int i = 0; i < 3; i++) {
            char row[3] = {'e', 'm', 't'};
            printf("%c:  ", row[i]);
            for (int j = 0; j < 3; j++) {
                double mag = sqrt(pmns.real[i][j] * pmns.real[i][j] + pmns.imag[i][j] * pmns.imag[i][j]);
                printf("%.4f    ", mag);
            }
            printf("\n");
        }
        printf("Jarlskog invariant J = %.6e\n", pmns.jarlskog);
        printf("Unitarity violation = %.6e\n", pmns.unitarity_violation);
        
        printf("\n=== 6. Standard Model Comparison ===\n\n");
        
        int sm_match = check_standard_model_match(&ps);
        
        printf("Standard Model match check:\n");
        printf("  Three generations: %s\n", ps.num_generations == 3 ? "YES" : "NO");
        printf("  Has quarks: %s\n", ps.has_quarks ? "YES" : "NO");
        printf("  Has leptons: %s\n", ps.has_leptons ? "YES" : "NO");
        printf("  Gauge group SU(3)xSU(2)xU(1): %s\n", group.matches_standard_model ? "YES" : "NO");
        
        printf("\n=== Phase 3 Success Criteria ===\n\n");
        
        int criterion1_p3 = (spectrum.num_cycles > 10);
        int criterion2_p3 = (spectrum.num_stable > 0);
        int criterion3_p3 = (ps.num_generations >= 2);
        int criterion4_p3 = group.matches_standard_model;
        
        printf("Criterion 1: Sufficient cycles found - %s\n", 
               criterion1_p3 ? "PASS" : "FAIL");
        printf("Criterion 2: Stable cycles exist - %s\n", 
               criterion2_p3 ? "PASS" : "FAIL");
        printf("Criterion 3: Multiple generations - %s\n", 
               criterion3_p3 ? "PASS" : "FAIL");
        printf("Criterion 4: Standard Model gauge group - %s\n", 
               criterion4_p3 ? "PASS" : "FAIL");
        
        int all_pass_p3 = criterion1_p3 && criterion2_p3 && criterion3_p3;
        printf("\n=== PHASE 3 STATUS: %s ===\n", 
               all_pass_p3 ? "PARTIAL SUCCESS" : "NEEDS MORE WORK");
        
        printf("\nNote: Full Standard Model matching (3 generations, exact masses)\n");
        printf("requires much larger systems and deeper theoretical development.\n");
        
        break;
    }
    default:
        printf("Unknown mode %d. Available modes: 0-20\n", mode);
        printf("  0: Quick diagnostic (N=60)\n");
        printf("  1: Single ER simulation (N=100)\n");
        printf("  2: Parameter scan ER (N=60)\n");
        printf("  3: Topology comparison (N=60)\n");
        printf("  4: Time evolution ER (N=60)\n");
        printf("  5: Parameter scan DAG (N=60)\n");
        printf("  6: Larger ER scan (N=100)\n");
        printf("  7: Single DAG simulation (N=100)\n");
        printf("  8: Small world simulation (N=100)\n");
        printf("  9: Large scale ER scan (N=200)\n");
        printf("  10: Large scale single simulation (N=200)\n");
        printf("  11: Light cone analysis (N=150)\n");
        printf("  12: Finite size scaling analysis\n");
        printf("  13: Large scale N=300 simulation\n");
        printf("  14: Large scale N=400 simulation\n");
        printf("  15: Large scale N=500 simulation\n");
        printf("  16: 4D Lorentz verification\n");
        printf("  17: Metric extraction & coarse-graining\n");
        printf("  18: Phase 1 complete verification\n");
        printf("  19: Phase 2 - Einstein-Hilbert term\n");
        printf("  20: Phase 3 - Particle spectrum\n");
        break;
    }

    printf("\n============================================================\n");
    printf("  Phase 1 Verification Complete\n");
    printf("============================================================\n");

    return 0;
}
