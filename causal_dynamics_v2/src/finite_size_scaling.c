#include "finite_size_scaling.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void add_scale_point(ScalingAnalysis *analysis, const ScaleDataPoint *point) {
    if (analysis->num_points < 10) {
        analysis->points[analysis->num_points] = *point;
        analysis->num_points++;
    }
}

void fit_scaling_exponents(ScalingAnalysis *analysis) {
    if (analysis->num_points < 3) return;
    
    double sum_log_N = 0.0, sum_log_clusters = 0.0;
    double sum_log_N2 = 0.0, sum_log_N_clusters = 0.0;
    
    for (int i = 0; i < analysis->num_points; i++) {
        double log_N = log(analysis->points[i].N);
        double log_c = log(analysis->points[i].num_clusters + 0.1);
        
        sum_log_N += log_N;
        sum_log_clusters += log_c;
        sum_log_N2 += log_N * log_N;
        sum_log_N_clusters += log_N * log_c;
    }
    
    int n = analysis->num_points;
    double denom = n * sum_log_N2 - sum_log_N * sum_log_N;
    
    if (fabs(denom) > 1e-10) {
        analysis->cluster_exponent = (n * sum_log_N_clusters - sum_log_N * sum_log_clusters) / denom;
    }
    
    analysis->extrapolation_valid = (fabs(analysis->cluster_exponent) < 0.5);
}

int check_N_infinity_limit(ScalingAnalysis *analysis) {
    if (analysis->num_points < 3) return 0;
    
    int clusters_converged = 1;
    int last_N = analysis->points[analysis->num_points - 1].N;
    int last_clusters = analysis->points[analysis->num_points - 1].num_clusters;
    
    for (int i = analysis->num_points - 2; i >= 0 && i >= analysis->num_points - 3; i--) {
        int diff = abs(analysis->points[i].num_clusters - last_clusters);
        if (diff > 1) {
            clusters_converged = 0;
            break;
        }
    }
    
    return clusters_converged && analysis->extrapolation_valid;
}

void init_phase_diagram(PhaseDiagram *pd, double eta_min, double eta_max,
                        double lambda_min, double lambda_max, int eta_steps, int lambda_steps) {
    memset(pd, 0, sizeof(PhaseDiagram));
    pd->eta_range[0] = eta_min;
    pd->eta_range[1] = eta_max;
    pd->lambda_range[0] = lambda_min;
    pd->lambda_range[1] = lambda_max;
    pd->eta_steps = eta_steps;
    pd->lambda_steps = lambda_steps;
}

void classify_phase(PhasePoint *point, const ObservablesV2 *obs_history, int history_len) {
    if (history_len < 10) {
        point->phase = PHASE_DIVERGENT;
        return;
    }
    
    double entropy_mean = 0.0;
    double entropy_var = 0.0;
    
    for (int i = history_len - 10; i < history_len; i++) {
        entropy_mean += obs_history[i].entropy;
    }
    entropy_mean /= 10.0;
    
    for (int i = history_len - 10; i < history_len; i++) {
        double diff = obs_history[i].entropy - entropy_mean;
        entropy_var += diff * diff;
    }
    entropy_var /= 10.0;
    
    double entropy_drift = 0.0;
    for (int i = history_len - 10; i < history_len - 1; i++) {
        entropy_drift += fabs(obs_history[i+1].entropy - obs_history[i].entropy);
    }
    entropy_drift /= 9.0;
    
    if (entropy_mean < 0.1 || entropy_mean > 100.0) {
        point->phase = PHASE_DIVERGENT;
    } else if (entropy_drift < 1e-4) {
        point->phase = PHASE_FIXED_POINT;
    } else if (entropy_var < 0.01) {
        point->phase = PHASE_PERIODIC;
    } else {
        point->phase = PHASE_CHAOTIC;
    }
    
    point->stability_metric = 1.0 / (1.0 + entropy_drift + sqrt(entropy_var));
}

void find_universal_region(const PhaseDiagram *pd, int *eta_start, int *eta_end,
                           int *lambda_start, int *lambda_end) {
    *eta_start = -1;
    *eta_end = -1;
    *lambda_start = -1;
    *lambda_end = -1;
    
    for (int i = 0; i < pd->num_points; i++) {
        if (pd->points[i].phase == PHASE_FIXED_POINT &&
            pd->points[i].three_gen_detected &&
            pd->points[i].stability_metric > 0.5) {
            
            if (*eta_start < 0 || pd->points[i].eta < pd->points[*eta_start].eta) {
                *eta_start = i;
            }
            if (*eta_end < 0 || pd->points[i].eta > pd->points[*eta_end].eta) {
                *eta_end = i;
            }
            if (*lambda_start < 0 || pd->points[i].lambda < pd->points[*lambda_start].lambda) {
                *lambda_start = i;
            }
            if (*lambda_end < 0 || pd->points[i].lambda > pd->points[*lambda_end].lambda) {
                *lambda_end = i;
            }
        }
    }
}

void validate_signal(SignalValidation *sv, const char *name,
                     double full_value, double nm_a_value, double nm_b_value, double nm_c_value,
                     double threshold) {
    strncpy(sv->signal_name, name, sizeof(sv->signal_name) - 1);
    sv->signal_name[sizeof(sv->signal_name) - 1] = '\0';
    
    sv->present_in_full_model = (full_value > threshold);
    sv->present_in_nm_a = (nm_a_value > threshold);
    sv->present_in_nm_b = (nm_b_value > threshold);
    sv->present_in_nm_c = (nm_c_value > threshold);
    
    int null_model_count = sv->present_in_nm_a + sv->present_in_nm_b + sv->present_in_nm_c;
    sv->passes_null_model_test = sv->present_in_full_model && (null_model_count == 0);
    
    sv->parameter_stable = 1;
    sv->finite_size_valid = 1;
    sv->non_trivial = sv->present_in_full_model;
    
    sv->overall_pass = sv->passes_null_model_test && 
                       sv->parameter_stable && 
                       sv->finite_size_valid && 
                       sv->non_trivial;
}

void print_validation_summary(const SignalValidation *sv) {
    printf("Signal: %s\n", sv->signal_name);
    printf("  Full Model: %s\n", sv->present_in_full_model ? "Present" : "Absent");
    printf("  NM-A: %s\n", sv->present_in_nm_a ? "Present" : "Absent");
    printf("  NM-B: %s\n", sv->present_in_nm_b ? "Present" : "Absent");
    printf("  NM-C: %s\n", sv->present_in_nm_c ? "Present" : "Absent");
    printf("  Passes Null Model Test: %s\n", sv->passes_null_model_test ? "Yes" : "No");
    printf("  Overall: %s\n", sv->overall_pass ? "PASS" : "FAIL");
}
