#include "rg_flow.h"
#include "phase_space.h"
#include "observables.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void init_coarse_graining_map(CoarseGrainingMap *cg_map, int num_blocks) {
    cg_map->num_blocks = num_blocks;
    for (int i = 0; i < num_blocks; i++) {
        cg_map->block_size[i] = 0;
    }
    for (int i = 0; i < MAX_NODES; i++) {
        cg_map->node_to_block[i] = -1;
    }
}

void assign_node_to_block(CoarseGrainingMap *cg_map, int node, int block) {
    cg_map->node_to_block[node] = block;
    int idx = cg_map->block_size[block];
    cg_map->block_nodes[block][idx] = node;
    cg_map->block_size[block]++;
}

void create_block_partition_tetrahedron(CoarseGrainingMap *cg_map) {
    init_coarse_graining_map(cg_map, 2);
    
    assign_node_to_block(cg_map, 0, 0);
    assign_node_to_block(cg_map, 1, 0);
    assign_node_to_block(cg_map, 2, 1);
    assign_node_to_block(cg_map, 3, 1);
}

void coarse_grain_graph(const Graph *fine_graph, const CoarseGrainingMap *cg_map,
                        Graph *coarse_graph) {
    coarse_graph->num_nodes = cg_map->num_blocks;
    coarse_graph->num_edges = 0;
    coarse_graph->num_faces = 0;
    
    for (int b1 = 0; b1 < cg_map->num_blocks; b1++) {
        for (int b2 = b1 + 1; b2 < cg_map->num_blocks; b2++) {
            int edge_exists = 0;
            
            for (int i = 0; i < cg_map->block_size[b1]; i++) {
                int node1 = cg_map->block_nodes[b1][i];
                
                for (int j = 0; j < cg_map->block_size[b2]; j++) {
                    int node2 = cg_map->block_nodes[b2][j];
                    
                    for (int e = 0; e < fine_graph->num_edges; e++) {
                        int from = fine_graph->edges[e].i;
                        int to = fine_graph->edges[e].j;
                        
                        if ((from == node1 && to == node2) || 
                            (from == node2 && to == node1)) {
                            edge_exists = 1;
                            break;
                        }
                    }
                    if (edge_exists) break;
                }
                if (edge_exists) break;
            }
            
            if (edge_exists) {
                int e = coarse_graph->num_edges;
                coarse_graph->edges[e].i = b1;
                coarse_graph->edges[e].j = b2;
                coarse_graph->num_edges++;
            }
        }
    }
    
    for (int b1 = 0; b1 < cg_map->num_blocks; b1++) {
        coarse_graph->node_edge_count[b1] = 0;
        for (int e = 0; e < coarse_graph->num_edges; e++) {
            if (coarse_graph->edges[e].i == b1 || coarse_graph->edges[e].j == b1) {
                coarse_graph->node_edge_count[b1]++;
            }
        }
    }
}

void coarse_grain_phase_space(const PhaseSpace *fine_ps, const Graph *fine_graph,
                               const CoarseGrainingMap *cg_map, PhaseSpace *coarse_ps) {
    for (int e = 0; e < cg_map->num_blocks * (cg_map->num_blocks - 1) / 2; e++) {
        coarse_ps->A[e] = 0.0;
        coarse_ps->E[e] = 0.0;
    }
    
    int coarse_edge_idx = 0;
    for (int b1 = 0; b1 < cg_map->num_blocks; b1++) {
        for (int b2 = b1 + 1; b2 < cg_map->num_blocks; b2++) {
            double A_sum = 0.0;
            double E_sum = 0.0;
            int count = 0;
            
            for (int i = 0; i < cg_map->block_size[b1]; i++) {
                int node1 = cg_map->block_nodes[b1][i];
                
                for (int j = 0; j < cg_map->block_size[b2]; j++) {
                    int node2 = cg_map->block_nodes[b2][j];
                    
                    for (int e = 0; e < fine_graph->num_edges; e++) {
                        int from = fine_graph->edges[e].i;
                        int to = fine_graph->edges[e].j;
                        
                        if ((from == node1 && to == node2) || 
                            (from == node2 && to == node1)) {
                            A_sum += fine_ps->A[e];
                            E_sum += fine_ps->E[e];
                            count++;
                        }
                    }
                }
            }
            
            if (count > 0) {
                coarse_ps->A[coarse_edge_idx] = atan2(sin(A_sum), cos(A_sum));
                coarse_ps->E[coarse_edge_idx] = E_sum / count;
            }
            coarse_edge_idx++;
        }
    }
}

void init_monte_carlo_ensemble(MonteCarloEnsemble *ensemble) {
    ensemble->num_configs = 0;
    ensemble->total_weight = 0.0;
    ensemble->average_energy = 0.0;
    ensemble->energy_variance = 0.0;
}

void generate_random_configuration(MonteCarloConfiguration *config, const Graph *graph, 
                                   unsigned int *seed) {
    for (int e = 0; e < graph->num_edges; e++) {
        config->A[e] = 2.0 * M_PI * ((double)rand() / RAND_MAX) - M_PI;
        config->E[e] = 0.1 * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
    }
}

double compute_configuration_energy(const MonteCarloConfiguration *config, 
                                    const Graph *graph, double beta) {
    double H_kin = 0.0;
    for (int e = 0; e < graph->num_edges; e++) {
        H_kin += 0.5 * config->E[e] * config->E[e];
    }
    
    double H_curv = 0.0;
    for (int f = 0; f < graph->num_faces; f++) {
        double F = 0.0;
        for (int i = 0; i < 3; i++) {
            int n1 = graph->faces[f].nodes[i];
            int n2 = graph->faces[f].nodes[(i + 1) % 3];
            
            int e = -1;
            for (int j = 0; j < graph->num_edges; j++) {
                if ((graph->edges[j].i == n1 && graph->edges[j].j == n2) ||
                    (graph->edges[j].i == n2 && graph->edges[j].j == n1)) {
                    e = j;
                    break;
                }
            }
            
            if (e >= 0) {
                if (graph->edges[e].i == n1) {
                    F += config->A[e];
                } else {
                    F -= config->A[e];
                }
            }
        }
        H_curv -= beta * cos(F);
    }
    
    return H_kin + H_curv;
}

int metropolis_step(MonteCarloConfiguration *config, const Graph *graph, 
                    double beta, double delta, unsigned int *seed) {
    int edge = rand() % graph->num_edges;
    int var_type = rand() % 2;
    
    double old_val, new_val;
    if (var_type == 0) {
        old_val = config->A[edge];
        new_val = old_val + delta * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
        while (new_val > M_PI) new_val -= 2.0 * M_PI;
        while (new_val < -M_PI) new_val += 2.0 * M_PI;
        config->A[edge] = new_val;
    } else {
        old_val = config->E[edge];
        new_val = old_val + 0.1 * delta * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
        if (new_val > 2.0) new_val = 2.0;
        if (new_val < -2.0) new_val = -2.0;
        config->E[edge] = new_val;
    }
    
    double E_old = compute_configuration_energy(config, graph, beta);
    
    if (var_type == 0) {
        config->A[edge] = old_val;
    } else {
        config->E[edge] = old_val;
    }
    
    double E_new = compute_configuration_energy(config, graph, beta);
    
    double dE = E_new - E_old;
    double accept_prob = exp(-beta * dE);
    
    if (dE < 0 || ((double)rand() / RAND_MAX) < accept_prob) {
        if (var_type == 0) {
            config->A[edge] = new_val;
        } else {
            config->E[edge] = new_val;
        }
        return 1;
    }
    
    return 0;
}

void run_monte_carlo_sampling(const Graph *graph, double beta, 
                              MonteCarloEnsemble *ensemble, int num_steps,
                              unsigned int seed) {
    init_monte_carlo_ensemble(ensemble);
    
    MonteCarloConfiguration current_config;
    generate_random_configuration(&current_config, graph, &seed);
    
    double delta = 0.5;
    int accept_count = 0;
    
    for (int step = 0; step < THERMALIZATION_STEPS; step++) {
        accept_count += metropolis_step(&current_config, graph, beta, delta, &seed);
        
        if (step > 0 && step % 100 == 0) {
            double accept_rate = (double)accept_count / 100.0;
            if (accept_rate < 0.3) delta *= 0.9;
            else if (accept_rate > 0.6) delta *= 1.1;
            accept_count = 0;
        }
    }
    
    double energy_sum = 0.0;
    double energy_sq_sum = 0.0;
    
    for (int step = 0; step < num_steps; step++) {
        for (int sub = 0; sub < 10; sub++) {
            metropolis_step(&current_config, graph, beta, delta, &seed);
        }
        
        double energy = compute_configuration_energy(&current_config, graph, beta);
        
        if (ensemble->num_configs < MAX_ITERATIONS) {
            int idx = ensemble->num_configs;
            memcpy(ensemble->configs[idx].A, current_config.A, 
                   graph->num_edges * sizeof(double));
            memcpy(ensemble->configs[idx].E, current_config.E, 
                   graph->num_edges * sizeof(double));
            ensemble->configs[idx].energy = energy;
            ensemble->num_configs++;
        }
        
        energy_sum += energy;
        energy_sq_sum += energy * energy;
    }
    
    ensemble->average_energy = energy_sum / num_steps;
    ensemble->energy_variance = energy_sq_sum / num_steps - 
                                 ensemble->average_energy * ensemble->average_energy;
}

double compute_effective_beta(const MonteCarloEnsemble *ensemble, 
                              const Graph *coarse_graph) {
    if (ensemble->num_configs == 0) return 0.0;
    
    double E_mean = ensemble->average_energy;
    double E_var = ensemble->energy_variance;
    
    if (E_var < 1e-10) return 1.0;
    
    int num_dof = 2 * coarse_graph->num_edges;
    double beta_eff = num_dof / (2.0 * E_var);
    
    if (beta_eff > 10.0) beta_eff = 10.0;
    if (beta_eff < 0.01) beta_eff = 0.01;
    
    return beta_eff;
}

void compute_effective_hamiltonian(const Graph *fine_graph, 
                                   const MonteCarloEnsemble *ensemble,
                                   CoarseGrainedSystem *cg_system) {
    cg_system->effective_beta = compute_effective_beta(ensemble, &cg_system->coarse_graph);
    
    double coupling_sum = 0.0;
    for (int e = 0; e < cg_system->coarse_graph.num_edges; e++) {
        double E_sq_sum = 0.0;
        for (int c = 0; c < ensemble->num_configs; c++) {
            E_sq_sum += ensemble->configs[c].E[e] * ensemble->configs[c].E[e];
        }
        coupling_sum += E_sq_sum / ensemble->num_configs;
    }
    
    cg_system->effective_coupling = coupling_sum / cg_system->coarse_graph.num_edges;
}

void init_rg_flow_data(RGFlowData *rg_data) {
    rg_data->num_points = 0;
    rg_data->beta_star = 0.0;
    rg_data->fixed_point_found = 0;
}

void compute_rg_flow_step(const Graph *graph, double beta_in, 
                          double *beta_out, unsigned int seed) {
    MonteCarloEnsemble ensemble;
    run_monte_carlo_sampling(graph, beta_in, &ensemble, MEASUREMENT_STEPS, seed);
    
    *beta_out = compute_effective_beta(&ensemble, graph);
}

void compute_rg_flow(const Graph *graph, double beta_initial, 
                     RGFlowData *rg_data, int max_steps, unsigned int seed) {
    init_rg_flow_data(rg_data);
    
    double beta_current = beta_initial;
    
    for (int step = 0; step < max_steps; step++) {
        double beta_next;
        compute_rg_flow_step(graph, beta_current, &beta_next, seed + step);
        
        rg_data->beta_values[step] = beta_current;
        rg_data->energy_values[step] = 0.0;
        rg_data->num_points = step + 1;
        
        double diff = fabs(beta_next - beta_current);
        if (diff < 1e-6) {
            rg_data->beta_star = beta_current;
            rg_data->fixed_point_found = 1;
            printf("Fixed point found at step %d: beta* = %.6f\n", step, beta_current);
            return;
        }
        
        beta_current = beta_next;
    }
    
    rg_data->beta_star = beta_current;
    rg_data->fixed_point_found = 0;
}

double find_fixed_point(const RGFlowData *rg_data) {
    if (rg_data->num_points < 2) return 0.0;
    
    for (int i = 1; i < rg_data->num_points; i++) {
        double diff = fabs(rg_data->beta_values[i] - rg_data->beta_values[i-1]);
        if (diff < 1e-5) {
            return rg_data->beta_values[i];
        }
    }
    
    return rg_data->beta_values[rg_data->num_points - 1];
}

double find_fixed_point_bisection(const Graph *graph, double beta_low, 
                                   double beta_high, double tolerance,
                                   unsigned int seed) {
    double beta_out_low, beta_out_high;
    compute_rg_flow_step(graph, beta_low, &beta_out_low, seed);
    compute_rg_flow_step(graph, beta_high, &beta_out_high, seed + 1);
    
    double f_low = beta_out_low - beta_low;
    double f_high = beta_out_high - beta_high;
    
    if (f_low * f_high > 0) {
        printf("Warning: No fixed point in interval [%.3f, %.3f]\n", beta_low, beta_high);
        return (beta_low + beta_high) / 2.0;
    }
    
    for (int iter = 0; iter < 50; iter++) {
        double beta_mid = (beta_low + beta_high) / 2.0;
        double beta_out_mid;
        compute_rg_flow_step(graph, beta_mid, &beta_out_mid, seed + iter + 2);
        
        double f_mid = beta_out_mid - beta_mid;
        
        if (fabs(f_mid) < tolerance) {
            printf("Fixed point found (bisection): beta* = %.6f after %d iterations\n", 
                   beta_mid, iter + 1);
            return beta_mid;
        }
        
        if (f_low * f_mid < 0) {
            beta_high = beta_mid;
            f_high = f_mid;
        } else {
            beta_low = beta_mid;
            f_low = f_mid;
        }
    }
    
    return (beta_low + beta_high) / 2.0;
}

int verify_fixed_point(const Graph *graph, double beta_candidate, 
                       double tolerance, unsigned int seed) {
    double beta_out;
    compute_rg_flow_step(graph, beta_candidate, &beta_out, seed);
    
    double diff = fabs(beta_out - beta_candidate);
    return diff < tolerance;
}

void compute_correlation_function(const Graph *graph, 
                                  const MonteCarloEnsemble *ensemble,
                                  CorrelationFunction *corr) {
    for (int i = 0; i < graph->num_nodes; i++) {
        for (int j = 0; j < graph->num_nodes; j++) {
            double corr_sum = 0.0;
            
            for (int c = 0; c < ensemble->num_configs; c++) {
                double cos_A_i = 0.0, cos_A_j = 0.0;
                int count_i = 0, count_j = 0;
                
                for (int e = 0; e < graph->num_edges; e++) {
                    if (graph->edges[e].i == i || graph->edges[e].j == i) {
                        cos_A_i += cos(ensemble->configs[c].A[e]);
                        count_i++;
                    }
                    if (graph->edges[e].i == j || graph->edges[e].j == j) {
                        cos_A_j += cos(ensemble->configs[c].A[e]);
                        count_j++;
                    }
                }
                
                if (count_i > 0 && count_j > 0) {
                    corr_sum += (cos_A_i / count_i) * (cos_A_j / count_j);
                }
            }
            
            corr->correlation[i][j] = corr_sum / ensemble->num_configs;
        }
    }
    
    corr->max_distance = graph->num_nodes;
}

double compute_correlation_length(const CorrelationFunction *corr, 
                                  const Graph *graph) {
    double xi_sum = 0.0;
    int count = 0;
    
    for (int i = 0; i < graph->num_nodes; i++) {
        for (int j = i + 1; j < graph->num_nodes; j++) {
            double c_ij = corr->correlation[i][j];
            double c_ii = corr->correlation[i][i];
            double c_jj = corr->correlation[j][j];
            
            if (c_ii > 0 && c_jj > 0 && fabs(c_ij) > 1e-10) {
                double norm_corr = c_ij / sqrt(c_ii * c_jj);
                if (norm_corr > 0 && norm_corr < 1) {
                    double distance = 1.0;
                    double xi = -distance / log(norm_corr);
                    if (xi > 0 && xi < 1000) {
                        xi_sum += xi;
                        count++;
                    }
                }
            }
        }
    }
    
    return count > 0 ? xi_sum / count : 0.0;
}

void measure_correlation_length_vs_beta(const Graph *graph, 
                                        const double *beta_values, int num_betas,
                                        double *correlation_lengths,
                                        unsigned int seed) {
    for (int i = 0; i < num_betas; i++) {
        MonteCarloEnsemble ensemble;
        run_monte_carlo_sampling(graph, beta_values[i], &ensemble, 
                                MEASUREMENT_STEPS, seed + i * 1000);
        
        CorrelationFunction corr;
        compute_correlation_function(graph, &ensemble, &corr);
        
        correlation_lengths[i] = compute_correlation_length(&corr, graph);
    }
}

double compute_specific_heat(const MonteCarloEnsemble *ensemble) {
    if (ensemble->num_configs == 0) return 0.0;
    
    return ensemble->energy_variance;
}

double compute_susceptibility(const Graph *graph, 
                              const MonteCarloEnsemble *ensemble) {
    double chi = 0.0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        double E_mean = 0.0;
        double E_sq_mean = 0.0;
        
        for (int c = 0; c < ensemble->num_configs; c++) {
            E_mean += ensemble->configs[c].E[e];
            E_sq_mean += ensemble->configs[c].E[e] * ensemble->configs[c].E[e];
        }
        
        E_mean /= ensemble->num_configs;
        E_sq_mean /= ensemble->num_configs;
        
        chi += E_sq_mean - E_mean * E_mean;
    }
    
    return chi;
}

void compute_thermodynamic_quantities(const Graph *graph, double beta,
                                      double *specific_heat, double *susceptibility,
                                      unsigned int seed) {
    MonteCarloEnsemble ensemble;
    run_monte_carlo_sampling(graph, beta, &ensemble, MEASUREMENT_STEPS, seed);
    
    *specific_heat = compute_specific_heat(&ensemble);
    *susceptibility = compute_susceptibility(graph, &ensemble);
}

void print_coarse_graining_map(const CoarseGrainingMap *cg_map) {
    printf("=== Coarse Graining Map ===\n");
    printf("Number of blocks: %d\n", cg_map->num_blocks);
    for (int b = 0; b < cg_map->num_blocks; b++) {
        printf("Block %d: ", b);
        for (int i = 0; i < cg_map->block_size[b]; i++) {
            printf("%d ", cg_map->block_nodes[b][i]);
        }
        printf("\n");
    }
}

void print_rg_flow_data(const RGFlowData *rg_data) {
    printf("\n=== RG Flow Data ===\n");
    printf("Number of steps: %d\n", rg_data->num_points);
    printf("Fixed point found: %s\n", rg_data->fixed_point_found ? "Yes" : "No");
    printf("Beta*: %.6f\n", rg_data->beta_star);
    
    printf("\nBeta flow:\n");
    for (int i = 0; i < rg_data->num_points && i < 20; i++) {
        printf("  Step %d: beta = %.6f\n", i, rg_data->beta_values[i]);
    }
}

void print_correlation_function(const CorrelationFunction *corr, int num_nodes) {
    printf("\n=== Correlation Function ===\n");
    printf("        ");
    for (int j = 0; j < num_nodes; j++) {
        printf("%10d ", j);
    }
    printf("\n");
    
    for (int i = 0; i < num_nodes; i++) {
        printf("%3d    ", i);
        for (int j = 0; j < num_nodes; j++) {
            printf("%10.6f ", corr->correlation[i][j]);
        }
        printf("\n");
    }
    
    printf("\nCorrelation length: %.6f\n", corr->correlation_length);
}

void write_rg_flow_data(const RGFlowData *rg_data, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "# RG Flow Data\n");
    fprintf(fp, "# step  beta\n");
    
    for (int i = 0; i < rg_data->num_points; i++) {
        fprintf(fp, "%d  %.10f\n", i, rg_data->beta_values[i]);
    }
    
    fclose(fp);
}

void write_correlation_data(const CorrelationFunction *corr, int num_nodes,
                            const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "# Correlation Function\n");
    fprintf(fp, "# i  j  correlation\n");
    
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            fprintf(fp, "%d  %d  %.10f\n", i, j, corr->correlation[i][j]);
        }
    }
    
    fclose(fp);
}
