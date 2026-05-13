#include "geometry_emergence.h"
#include "wilson_loop.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
    for (int i = 0; i < num_gaps; i++) {
        mean_gap += gaps[i];
    }
    mean_gap /= num_gaps;
    
    int cluster_start = 0;
    int cluster_idx = 0;
    
    result->cluster_centers[0] = spectrum->eigenvalues[0];
    result->cluster_counts[0] = 1;
    
    for (int i = 0; i < num_gaps && cluster_idx < MAX_EIGENVECTORS - 1; i++) {
        if (gaps[i] > gap_threshold * mean_gap) {
            cluster_idx++;
            result->cluster_centers[cluster_idx] = spectrum->eigenvalues[i + 1];
            result->cluster_counts[cluster_idx] = 1;
            result->cluster_widths[cluster_idx - 1] = spectrum->eigenvalues[i] - result->cluster_centers[cluster_idx - 1];
        } else {
            result->cluster_counts[cluster_idx]++;
        }
    }
    
    result->num_clusters = cluster_idx + 1;
    
    result->three_generation_detected = detect_three_generations(result);
    
    if (result->three_generation_detected) {
        compute_mass_ratios(result);
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
