#ifndef FINITE_SIZE_SCALING_H
#define FINITE_SIZE_SCALING_H

#include "dynamics_v2.h"
#include "geometry_emergence.h"

typedef struct {
    int N;
    int num_clusters;
    double cluster_spacing;
    double wilson_peak_width;
    double light_cone_velocity;
    double einstein_slope;
    double convergence_time;
} ScaleDataPoint;

typedef struct {
    ScaleDataPoint points[10];
    int num_points;
    double cluster_exponent;
    double spacing_exponent;
    int extrapolation_valid;
} ScalingAnalysis;

void add_scale_point(ScalingAnalysis *analysis, const ScaleDataPoint *point);
void fit_scaling_exponents(ScalingAnalysis *analysis);
int check_N_infinity_limit(ScalingAnalysis *analysis);

typedef enum {
    PHASE_FIXED_POINT,
    PHASE_PERIODIC,
    PHASE_CHAOTIC,
    PHASE_DIVERGENT
} PhaseType;

typedef struct {
    double eta;
    double lambda;
    PhaseType phase;
    int three_gen_detected;
    int geometry_detected;
    int gauge_detected;
    double stability_metric;
} PhasePoint;

typedef struct {
    PhasePoint points[1000];
    int num_points;
    double eta_range[2];
    double lambda_range[2];
    int eta_steps;
    int lambda_steps;
} PhaseDiagram;

void init_phase_diagram(PhaseDiagram *pd, double eta_min, double eta_max, 
                        double lambda_min, double lambda_max, int eta_steps, int lambda_steps);
void classify_phase(PhasePoint *point, const ObservablesV2 *obs_history, int history_len);
void find_universal_region(const PhaseDiagram *pd, int *eta_start, int *eta_end, 
                           int *lambda_start, int *lambda_end);

typedef struct {
    char signal_name[64];
    int present_in_full_model;
    int present_in_nm_a;
    int present_in_nm_b;
    int present_in_nm_c;
    int passes_null_model_test;
    int parameter_stable;
    int finite_size_valid;
    int non_trivial;
    int overall_pass;
} SignalValidation;

void validate_signal(SignalValidation *sv, const char *name,
                     double full_value, double nm_a_value, double nm_b_value, double nm_c_value,
                     double threshold);
void print_validation_summary(const SignalValidation *sv);

#endif
