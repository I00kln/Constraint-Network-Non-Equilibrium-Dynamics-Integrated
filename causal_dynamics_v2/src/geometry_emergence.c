#include "geometry_emergence.h"
#include "wilson_loop.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void compute_response_function(ResponseFunction *resp, const double *rho_history[MAX_TIME_STEPS],
                                int num_nodes, int num_steps, int origin_node) {
    memset(resp, 0, sizeof(ResponseFunction));
    resp->max_distance = MAX_DISTANCE;
    resp->max_time = num_steps < MAX_TIME_STEPS ? num_steps : MAX_TIME_STEPS;
    
    for (int t = 0; t < resp->max_time; t++) {
        for (int d = 0; d < resp->max_distance; d++) {
            resp->correlation[d][t] = 0.0;
        }
    }
    
    double rho0_mean = 0.0;
    for (int v = 0; v < num_nodes; v++) {
        rho0_mean += rho_history[0][v];
    }
    rho0_mean /= num_nodes;
    
    for (int t = 1; t < resp->max_time; t++) {
        double rho_t_mean = 0.0;
        for (int v = 0; v < num_nodes; v++) {
            rho_t_mean += rho_history[t][v];
        }
        rho_t_mean /= num_nodes;
        
        for (int v = 0; v < num_nodes; v++) {
            int dist = (v - origin_node + num_nodes) % num_nodes;
            if (dist > num_nodes / 2) dist = num_nodes - dist;
            
            if (dist < resp->max_distance) {
                double delta_rho0 = rho_history[0][origin_node] - rho0_mean;
                double delta_rho_t = rho_history[t][v] - rho_t_mean;
                resp->correlation[dist][t] += delta_rho0 * delta_rho_t;
            }
        }
    }
}

double fit_light_cone_velocity(ResponseFunction *resp) {
    int max_t = 0;
    int max_d = 0;
    double max_corr = 0.0;
    
    for (int t = 1; t < resp->max_time; t++) {
        for (int d = 0; d < resp->max_distance; d++) {
            if (fabs(resp->correlation[d][t]) > max_corr) {
                max_corr = fabs(resp->correlation[d][t]);
                max_t = t;
                max_d = d;
            }
        }
    }
    
    if (max_t > 0) {
        resp->light_cone_velocity = (double)max_d / max_t;
        resp->light_cone_detected = 1;
    } else {
        resp->light_cone_velocity = 0.0;
        resp->light_cone_detected = 0;
    }
    
    return resp->light_cone_velocity;
}

double compute_anisotropy(ResponseFunction *resp) {
    double sum_forward = 0.0;
    double sum_backward = 0.0;
    int count = 0;
    
    for (int t = 1; t < resp->max_time; t++) {
        for (int d = 0; d < resp->max_distance / 2; d++) {
            sum_forward += fabs(resp->correlation[d][t]);
            sum_backward += fabs(resp->correlation[resp->max_distance - 1 - d][t]);
            count++;
        }
    }
    
    if (count > 0 && (sum_forward + sum_backward) > 0) {
        resp->anisotropy = fabs(sum_forward - sum_backward) / (sum_forward + sum_backward);
    } else {
        resp->anisotropy = 0.0;
    }
    
    return resp->anisotropy;
}

void compute_einstein_relation(const RicciCurvature *ricci,
                               const InformationStateV2 *state,
                               EinsteinRelation *relation) {
    int n = state->num_nodes;
    
    for (int v = 0; v < n; v++) {
        relation->energy_density[v] = state->rho[v];
        relation->curvature[v] = ricci->curvature[v];
    }
    
    fit_curvature_density_relation(relation, n);
}

void fit_curvature_density_relation(EinsteinRelation *relation, int num_nodes) {
    double sum_T = 0.0, sum_R = 0.0, sum_TT = 0.0, sum_RT = 0.0;
    
    for (int v = 0; v < num_nodes; v++) {
        double T = relation->energy_density[v];
        double R = relation->curvature[v];
        sum_T += T;
        sum_R += R;
        sum_TT += T * T;
        sum_RT += R * T;
    }
    
    double mean_T = sum_T / num_nodes;
    double mean_R = sum_R / num_nodes;
    
    double var_T = sum_TT / num_nodes - mean_T * mean_T;
    double cov_RT = sum_RT / num_nodes - mean_R * mean_T;
    
    if (var_T > 1e-10) {
        relation->fit_result.slope = cov_RT / var_T;
        relation->fit_result.intercept = mean_R - relation->fit_result.slope * mean_T;
        relation->effective_kappa = -relation->fit_result.slope;
    } else {
        relation->fit_result.slope = 0.0;
        relation->fit_result.intercept = mean_R;
        relation->effective_kappa = 0.0;
    }
    
    double ss_tot = 0.0, ss_res = 0.0;
    for (int v = 0; v < num_nodes; v++) {
        double T = relation->energy_density[v];
        double R = relation->curvature[v];
        double R_pred = relation->fit_result.intercept + relation->fit_result.slope * T;
        ss_tot += (R - mean_R) * (R - mean_R);
        ss_res += (R - R_pred) * (R - R_pred);
    }
    
    if (ss_tot > 1e-10) {
        relation->fit_result.r_squared = 1.0 - ss_res / ss_tot;
    } else {
        relation->fit_result.r_squared = 0.0;
    }
    
    relation->fit_result.significant = (relation->fit_result.r_squared > 0.5);
}

void analyze_eigenvalue_clusters(const LaplacianSpectrum *spectrum,
                                  SpectralClusterResult *result,
                                  double gap_threshold) {
    memset(result, 0, sizeof(SpectralClusterResult));
    
    if (spectrum->num_eigenvalues < 2) return;
    
    static double gaps[MAX_EIGENVECTORS];
    int num_gaps = spectrum->num_eigenvalues - 1;
    
    for (int i = 0; i < num_gaps; i++) {
        gaps[i] = spectrum->eigenvalues[i + 1] - spectrum->eigenvalues[i];
    }
    
    double mean_gap = 0.0;
    double std_gap = 0.0;
    for (int i = 0; i < num_gaps; i++) {
        mean_gap += gaps[i];
    }
    mean_gap /= num_gaps;
    
    for (int i = 0; i < num_gaps; i++) {
        std_gap += (gaps[i] - mean_gap) * (gaps[i] - mean_gap);
    }
    std_gap = sqrt(std_gap / num_gaps);
    
    double adaptive_threshold = mean_gap + 2.0 * std_gap;
    
    int cluster_start = 0;
    int cluster_idx = 0;
    
    result->cluster_centers[0] = spectrum->eigenvalues[0];
    result->cluster_counts[0] = 1;
    
    for (int i = 0; i < num_gaps && cluster_idx < MAX_EIGENVECTORS - 1; i++) {
        if (gaps[i] > adaptive_threshold) {
            cluster_idx++;
            result->cluster_centers[cluster_idx] = spectrum->eigenvalues[i + 1];
            result->cluster_counts[cluster_idx] = 1;
            result->cluster_widths[cluster_idx - 1] = spectrum->eigenvalues[i] - result->cluster_centers[cluster_idx - 1];
        } else {
            result->cluster_counts[cluster_idx]++;
        }
    }
    
    result->num_clusters = cluster_idx + 1;
    
    for (int i = 0; i < result->num_clusters; i++) {
        if (result->cluster_counts[i] > 1) {
            int start_idx = 0;
            for (int j = 0; j < i; j++) {
                start_idx += result->cluster_counts[j];
            }
            double sum = 0.0;
            for (int j = 0; j < result->cluster_counts[i]; j++) {
                sum += spectrum->eigenvalues[start_idx + j];
            }
            result->cluster_centers[i] = sum / result->cluster_counts[i];
        }
    }
    
    result->three_generation_detected = detect_three_generations(result);
    
    if (result->three_generation_detected) {
        compute_mass_ratios(result);
    }
}

double compute_ipr(const double *eigenvector, int n) {
    if (n <= 0) return 0.0;
    
    double sum_sq = 0.0;
    double sum_quad = 0.0;
    
    for (int i = 0; i < n; i++) {
        double psi_sq = eigenvector[i] * eigenvector[i];
        sum_sq += psi_sq;
        sum_quad += psi_sq * psi_sq;
    }
    
    if (sum_sq < 1e-15) return 0.0;
    
    return sum_quad / (sum_sq * sum_sq);
}

void analyze_eigenmode_localization(const LaplacianSpectrum *spectrum,
                                     double *ipr_values,
                                     int *localized_modes,
                                     double ipr_threshold) {
    if (spectrum->num_eigenvalues <= 0) return;
    
    int num_localized = 0;
    
    for (int k = 0; k < spectrum->num_eigenvalues; k++) {
        static double eigenvector[MAX_NODES];
        for (int i = 0; i < MAX_NODES; i++) {
            eigenvector[i] = spectrum->eigenvectors[i][k];
        }
        
        ipr_values[k] = compute_ipr(eigenvector, MAX_NODES);
        
        if (ipr_values[k] > ipr_threshold) {
            localized_modes[num_localized++] = k;
        }
    }
    
    localized_modes[num_localized] = -1;
}

static int dfs_recursive_depth(const CausalGraphV2 *graph, int node, int *visited, int depth, int max_depth) {
    if (depth > max_depth) return depth;
    if (visited[node]) return depth;
    
    visited[node] = 1;
    int max_return_depth = 0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        if (graph->edges[e].from == node) {
            int next_node = graph->edges[e].to;
            int return_depth = dfs_recursive_depth(graph, next_node, visited, depth + 1, max_depth);
            if (return_depth > max_return_depth) {
                max_return_depth = return_depth;
            }
        }
    }
    
    visited[node] = 0;
    return max_return_depth;
}

void analyze_recursive_depth_distribution(const CausalGraphV2 *graph,
                                           int *depth_counts,
                                           int max_depth) {
    memset(depth_counts, 0, (max_depth + 1) * sizeof(int));
    
    static int visited[MAX_NODES];
    
    for (int v = 0; v < graph->num_nodes; v++) {
        memset(visited, 0, graph->num_nodes * sizeof(int));
        int depth = dfs_recursive_depth(graph, v, visited, 0, max_depth);
        
        if (depth <= max_depth) {
            depth_counts[depth]++;
        }
    }
}

void analyze_three_generation_structure(const CausalGraphV2 *graph,
                                         const LaplacianSpectrum *spectrum,
                                         ThreeGenerationResult *result) {
    memset(result, 0, sizeof(ThreeGenerationResult));
    
    analyze_recursive_depth_distribution(graph, result->depth_counts, 10);
    
    result->gen1_nodes = result->depth_counts[0];
    result->gen2_nodes = result->depth_counts[1];
    result->gen3_nodes = result->depth_counts[2];
    
    result->total_nodes = graph->num_nodes;
    
    if (result->total_nodes > 0) {
        result->gen1_fraction = (double)result->gen1_nodes / result->total_nodes;
        result->gen2_fraction = (double)result->gen2_nodes / result->total_nodes;
        result->gen3_fraction = (double)result->gen3_nodes / result->total_nodes;
    }
    
    result->three_gen_detected = (
        result->gen1_nodes > 10 &&
        result->gen2_nodes > 10 &&
        result->gen3_nodes > 10
    );
    
    if (result->three_gen_detected) {
        result->mass_ratio_12 = (double)result->gen2_nodes / result->gen1_nodes;
        result->mass_ratio_23 = (double)result->gen3_nodes / result->gen2_nodes;
    }
}

int detect_three_generations(SpectralClusterResult *result) {
    if (result->num_clusters < 3) return 0;
    
    int significant_clusters = 0;
    for (int i = 0; i < result->num_clusters; i++) {
        if (result->cluster_counts[i] >= 2) {
            significant_clusters++;
        }
    }
    
    return (significant_clusters >= 3);
}

void compute_mass_ratios(SpectralClusterResult *result) {
    if (result->num_clusters < 3) return;
    
    double S[3];
    int count = 0;
    for (int i = 0; i < result->num_clusters && count < 3; i++) {
        if (result->cluster_counts[i] >= 2) {
            S[count] = result->cluster_centers[i];
            count++;
        }
    }
    
    if (count >= 3) {
        result->mass_ratios[0] = 1.0;
        result->mass_ratios[1] = exp(S[1] - S[0]);
        result->mass_ratios[2] = exp(S[2] - S[0]);
    }
}

static void kmeans_clustering(const double *data, int n, int k, int *labels, double *centers) {
    static double temp_centers[MAX_EIGENVECTORS];
    static int counts[MAX_EIGENVECTORS];
    
    for (int i = 0; i < k; i++) {
        centers[i] = data[(i * n) / k];
    }
    
    for (int iter = 0; iter < 100; iter++) {
        int changed = 0;
        
        for (int i = 0; i < n; i++) {
            double min_dist = fabs(data[i] - centers[0]);
            int min_idx = 0;
            
            for (int j = 1; j < k; j++) {
                double dist = fabs(data[i] - centers[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            
            if (labels[i] != min_idx) {
                labels[i] = min_idx;
                changed = 1;
            }
        }
        
        if (!changed && iter > 10) break;
        
        memset(temp_centers, 0, k * sizeof(double));
        memset(counts, 0, k * sizeof(int));
        
        for (int i = 0; i < n; i++) {
            temp_centers[labels[i]] += data[i];
            counts[labels[i]]++;
        }
        
        for (int j = 0; j < k; j++) {
            if (counts[j] > 0) {
                centers[j] = temp_centers[j] / counts[j];
            }
        }
    }
}

static double compute_silhouette(const double *data, int n, const int *labels, int k) {
    double total_silhouette = 0.0;
    int valid_count = 0;
    
    for (int i = 0; i < n; i++) {
        int my_cluster = labels[i];
        double a_i = 0.0;
        int a_count = 0;
        
        for (int j = 0; j < n; j++) {
            if (i != j && labels[j] == my_cluster) {
                a_i += fabs(data[i] - data[j]);
                a_count++;
            }
        }
        if (a_count > 0) a_i /= a_count;
        
        double b_i = 1e30;
        for (int c = 0; c < k; c++) {
            if (c == my_cluster) continue;
            
            double dist = 0.0;
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (labels[j] == c) {
                    dist += fabs(data[i] - data[j]);
                    count++;
                }
            }
            if (count > 0) {
                dist /= count;
                if (dist < b_i) b_i = dist;
            }
        }
        
        if (a_count > 0 && b_i < 1e29) {
            double s_i = (b_i - a_i) / (a_i > b_i ? a_i : b_i);
            total_silhouette += s_i;
            valid_count++;
        }
    }
    
    return valid_count > 0 ? total_silhouette / valid_count : 0.0;
}

void analyze_eigenvalue_clusters_kmeans(const LaplacianSpectrum *spectrum,
                                        SpectralClusterResult *result,
                                        int max_k) {
    memset(result, 0, sizeof(SpectralClusterResult));
    
    if (spectrum->num_eigenvalues < 2) return;
    
    int n = spectrum->num_eigenvalues;
    static int labels[MAX_EIGENVECTORS];
    static double centers[MAX_EIGENVECTORS];
    static double best_labels[MAX_EIGENVECTORS];
    static double best_centers[MAX_EIGENVECTORS];
    
    double best_silhouette = -2.0;
    int best_k = 2;
    
    for (int k = 2; k <= max_k && k < n; k++) {
        for (int i = 0; i < n; i++) labels[i] = 0;
        
        kmeans_clustering(spectrum->eigenvalues, n, k, labels, centers);
        double silhouette = compute_silhouette(spectrum->eigenvalues, n, labels, k);
        
        if (silhouette > best_silhouette) {
            best_silhouette = silhouette;
            best_k = k;
            for (int i = 0; i < n; i++) best_labels[i] = labels[i];
            for (int i = 0; i < k; i++) best_centers[i] = centers[i];
        }
    }
    
    result->num_clusters = best_k;
    for (int i = 0; i < best_k; i++) {
        result->cluster_centers[i] = best_centers[i];
        result->cluster_counts[i] = 0;
    }
    
    for (int i = 0; i < n; i++) {
        result->cluster_counts[(int)best_labels[i]]++;
    }
    
    for (int i = 0; i < best_k; i++) {
        if (result->cluster_counts[i] > 0) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if ((int)best_labels[j] == i) {
                    sum += spectrum->eigenvalues[j];
                }
            }
            result->cluster_centers[i] = sum / result->cluster_counts[i];
        }
    }
    
    result->three_generation_detected = detect_three_generations(result);
    
    if (result->three_generation_detected) {
        compute_mass_ratios(result);
    }
}
