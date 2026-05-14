#include "dynamics_v2.h"
#include "wilson_loop.h"
#include "geometry_emergence.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double rand_uniform(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (double)(*seed % RAND_MAX) / RAND_MAX;
}

void init_null_model_config(NullModelConfig *config, ModelType type) {
    memset(config, 0, sizeof(NullModelConfig));
    config->type = type;
    
    switch (type) {
        case FULL_MODEL:
            strcpy(config->description, "Full model with all mechanisms");
            break;
        case NULL_MODEL_A:
            config->disable_macro_micro_decomposition = 1;
            strcpy(config->description, "Null Model A: No macro-micro decomposition");
            break;
        case NULL_MODEL_B:
            config->random_rewiring = 1;
            strcpy(config->description, "Null Model B: Random rewiring");
            break;
        case NULL_MODEL_C:
            config->static_topology = 1;
            strcpy(config->description, "Null Model C: Static topology");
            break;
    }
}

const char* model_type_name(ModelType type) {
    switch (type) {
        case FULL_MODEL: return "Full Model";
        case NULL_MODEL_A: return "Null Model A";
        case NULL_MODEL_B: return "Null Model B";
        case NULL_MODEL_C: return "Null Model C";
        default: return "Unknown";
    }
}

void init_info_state_v2(InformationStateV2 *state, int num_nodes) {
    memset(state, 0, sizeof(InformationStateV2));
    state->num_nodes = num_nodes;
}

void set_uniform_density_v2(InformationStateV2 *state, double total) {
    double density = total / state->num_nodes;
    for (int v = 0; v < state->num_nodes; v++) {
        state->rho[v] = density;
        state->macro[v] = density;
        state->micro[v] = 0.0;
    }
    state->total_info = total;
    state->total_macro = total;
    state->total_micro = 0.0;
}

void compute_totals_v2(InformationStateV2 *state) {
    double total_info = 0.0;
    double total_macro = 0.0;
    double total_micro = 0.0;
    
    #pragma omp parallel for reduction(+:total_info,total_macro,total_micro)
    for (int v = 0; v < state->num_nodes; v++) {
        total_info += state->rho[v];
        total_macro += state->macro[v];
        total_micro += state->micro[v];
    }
    
    state->total_info = total_info;
    state->total_macro = total_macro;
    state->total_micro = total_micro;
}

void step1_causal_propagation_v2(CausalGraphV2 *graph, InformationStateV2 *state, double eta) {
    static double new_rho[MAX_NODES];
    
    for (int v = 0; v < state->num_nodes; v++) {
        new_rho[v] = (1.0 - eta) * state->rho[v];
        
        int in_deg = graph->in_degree[v];
        if (in_deg > MAX_NEIGHBORS) in_deg = MAX_NEIGHBORS;
        
        if (in_deg > 0) {
            for (int i = 0; i < in_deg; i++) {
                int u = graph->in_neighbors[v][i];
                if (u < 0 || u >= state->num_nodes) continue;
                
                int edge_idx = graph->in_edge_indices[v][i];
                double weight = graph->edges[edge_idx].weight;
                double phase_mod_sq = 1.0;
                
                new_rho[v] += eta * state->rho[u] * weight * phase_mod_sq / in_deg;
            }
        }
        
        if (new_rho[v] < 1e-15) new_rho[v] = 1e-15;
    }
    
    for (int v = 0; v < state->num_nodes; v++) {
        state->rho[v] = new_rho[v];
    }
}

void step2_spectral_decomposition(const CausalGraphV2 *graph, InformationStateV2 *state,
                                   LaplacianSpectrum *spectrum, double kc_fraction) {
    int num_eigenvalues = (int)(0.5 * state->num_nodes);
    if (num_eigenvalues < 20) num_eigenvalues = 20;
    if (num_eigenvalues > MAX_EIGENVECTORS) num_eigenvalues = MAX_EIGENVECTORS;
    
    compute_laplacian_spectrum(graph, spectrum, num_eigenvalues);
    
    int cutoff = 1;
    double lambda_threshold = 0.1;
    for (int i = 0; i < spectrum->num_eigenvalues; i++) {
        if (spectrum->eigenvalues[i] < lambda_threshold) {
            cutoff = i + 1;
        }
    }
    if (cutoff < 2) cutoff = 2;
    
    project_to_macro(spectrum, state->rho, state->macro, state->num_nodes, cutoff);
    project_to_micro(spectrum, state->rho, state->micro, state->num_nodes, cutoff);
    
    spectrum->cutoff_index = cutoff;
    
    for (int v = 0; v < state->num_nodes; v++) {
        if (state->macro[v] < 1e-15) state->macro[v] = 1e-15;
    }
}

void step3_total_redistribution_v2(InformationStateV2 *state, double lambda, double target_total) {
    compute_totals_v2(state);
    
    double M = state->total_macro;
    double m = state->total_micro;
    
    double target = target_total - M;
    
    if (fabs(m) > 1e-10 && fabs(lambda) > 1e-10) {
        double scale = target / (lambda * m);
        
        if (scale > 10.0) scale = 10.0;
        if (scale < -10.0) scale = -10.0;
        
        for (int v = 0; v < state->num_nodes; v++) {
            state->micro[v] *= scale;
        }
        
        for (int v = 0; v < state->num_nodes; v++) {
            state->rho[v] = state->macro[v] + lambda * state->micro[v];
            
            if (state->rho[v] < 1e-15) {
                state->rho[v] = 1e-15;
            }
        }
    } else {
        double deficit = target_total - state->total_info;
        double correction = deficit / state->num_nodes;
        
        for (int v = 0; v < state->num_nodes; v++) {
            state->rho[v] += correction;
            if (state->rho[v] < 1e-15) state->rho[v] = 1e-15;
        }
    }
    
    compute_totals_v2(state);
    
    if (fabs(state->total_info - target_total) > 1e-6) {
        double renorm = target_total / state->total_info;
        for (int v = 0; v < state->num_nodes; v++) {
            state->rho[v] *= renorm;
        }
        compute_totals_v2(state);
    }
}

void step4_topology_phase_rewiring(CausalGraphV2 *graph, const InformationStateV2 *state,
                                    double alpha, double beta, double delta, double epsilon_del,
                                    double sigma_theta, int max_new_edges, unsigned int *seed) {
    static int edges_to_remove[MAX_EDGES];
    int num_to_remove = 0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        double new_weight = graph->edges[e].weight - delta;
        if (new_weight < epsilon_del || new_weight < 0.01) {
            if (num_to_remove < MAX_EDGES) {
                edges_to_remove[num_to_remove++] = e;
            }
        } else {
            graph->edges[e].weight = new_weight;
            if (graph->edges[e].weight < 0.01) graph->edges[e].weight = 0.01;
        }
    }
    
    for (int i = num_to_remove - 1; i >= 0; i--) {
        remove_edge_v2(graph, edges_to_remove[i]);
    }
    
    double avg_weight = 0.0;
    if (graph->num_edges > 0) {
        for (int e = 0; e < graph->num_edges; e++) {
            avg_weight += graph->edges[e].weight;
        }
        avg_weight /= graph->num_edges;
    } else {
        avg_weight = 1.0;
    }
    
    int edges_added = 0;
    int attempts = 0;
    int max_attempts = max_new_edges * 10;
    
    while (edges_added < max_new_edges && attempts < max_attempts) {
        double ru = rand_uniform(seed);
        double rv = rand_uniform(seed);
        int u = (int)(ru * graph->num_nodes);
        int v = (int)(rv * graph->num_nodes);
        
        if (u >= graph->num_nodes) u = graph->num_nodes - 1;
        if (v >= graph->num_nodes) v = graph->num_nodes - 1;
        if (u < 0) u = 0;
        if (v < 0) v = 0;
        
        if (u != v && find_edge_v2(graph, u, v) < 0) {
            double rho_u = state->macro[u];
            double rho_v = state->macro[v];
            
            if (rho_u < 1e-15) rho_u = 1e-15;
            if (rho_v < 1e-15) rho_v = 1e-15;
            
            double score = alpha * rho_u * rho_v + beta * fabs(1.0 - avg_weight);
            
            if (score > 20.0) score = 20.0;
            if (score < -20.0) score = -20.0;
            
            double prob = 1.0 / (1.0 + exp(-score));
            
            if (rand_uniform(seed) < prob) {
                double theta = rand_uniform(seed) * 2.0 * M_PI;
                add_edge_with_phase(graph, u, v, theta);
                edges_added++;
            }
        }
        attempts++;
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        double xi = (rand_uniform(seed) - 0.5) * sigma_theta;
        double old_theta = atan2(graph->edges[e].phase.im, graph->edges[e].phase.re);
        double new_theta = old_theta + xi;
        graph->edges[e].phase = u1_phase(new_theta);
    }
}

void compute_observables_v2(const CausalGraphV2 *graph, const InformationStateV2 *state,
                             const LaplacianSpectrum *spectrum, ObservablesV2 *obs) {
    memset(obs, 0, sizeof(ObservablesV2));
    
    double sum = 0.0;
    for (int v = 0; v < state->num_nodes; v++) {
        sum += state->rho[v];
    }
    
    if (sum > 1e-15) {
        for (int v = 0; v < state->num_nodes; v++) {
            double p = state->rho[v] / sum;
            if (p > 1e-15) {
                obs->entropy -= p * log(p);
            }
        }
    }
    
    obs->total_info = sum;
    obs->num_edges = graph->num_edges;
    
    if (graph->num_nodes > 0) {
        obs->avg_degree = (double)graph->num_edges / graph->num_nodes;
    }
    
    WilsonLoopCollection wlc;
    find_all_wilson_loops(graph, &wlc, 3, 8);
    analyze_wilson_distribution(&wlc);
    obs->wilson_avg_mod = wlc.avg_modulus;
    
    SpectralClusterResult sr;
    analyze_eigenvalue_clusters(spectrum, &sr, 2.0);
    obs->num_spectral_clusters = sr.num_clusters;
}

int check_convergence(const ObservablesV2 *obs_history, int history_len, double threshold) {
    if (history_len < 100) return 0;
    
    double sum_diff = 0.0;
    for (int i = history_len - 100; i < history_len - 1; i++) {
        double diff = fabs(obs_history[i+1].entropy - obs_history[i].entropy);
        sum_diff += diff;
    }
    
    double avg_diff = sum_diff / 99.0;
    return (avg_diff < threshold);
}

void dynamics_full_step_v2(CausalGraphV2 *graph, InformationStateV2 *state,
                           LaplacianSpectrum *spectrum, const DynamicsParamsV2 *params,
                           const NullModelConfig *nm_config, int step, unsigned int *seed) {
    step1_causal_propagation_v2(graph, state, params->eta);
    
    if (!nm_config->disable_macro_micro_decomposition) {
        step2_spectral_decomposition(graph, state, spectrum, params->kc_fraction);
        step3_total_redistribution_v2(state, params->lambda, 1.0);
    }
    
    if (!nm_config->static_topology) {
        double effective_alpha = nm_config->random_rewiring ? 0.0 : params->alpha;
        double effective_beta = nm_config->random_rewiring ? 0.0 : params->beta;
        
        step4_topology_phase_rewiring(graph, state, effective_alpha, effective_beta,
                                       params->delta, params->epsilon_del,
                                       params->sigma_theta, params->max_new_edges, seed);
    }
}
