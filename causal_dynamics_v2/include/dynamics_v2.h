#ifndef DYNAMICS_V2_H
#define DYNAMICS_V2_H

#include "causal_graph_v2.h"
#include "null_models.h"
#include "wilson_loop.h"

typedef struct {
    double eta;
    double lambda;
    double alpha;
    double beta;
    double delta;
    double epsilon_del;
    double sigma_theta;
    double kc_fraction;
    int T_filter;
    int max_new_edges;
    int max_steps;
} DynamicsParamsV2;

typedef struct {
    double entropy;
    double total_info;
    double avg_degree;
    int num_edges;
    double wilson_avg_mod;
    int num_spectral_clusters;
    double light_cone_velocity;
    double einstein_slope;
} ObservablesV2;

void step1_causal_propagation_v2(CausalGraphV2 *graph, InformationStateV2 *state, double eta);
void step2_spectral_decomposition(const CausalGraphV2 *graph, InformationStateV2 *state,
                                   LaplacianSpectrum *spectrum, double kc_fraction);
void step3_total_redistribution_v2(InformationStateV2 *state, double lambda, double target_total);
void step4_topology_phase_rewiring(CausalGraphV2 *graph, const InformationStateV2 *state,
                                    double alpha, double beta, double delta, double epsilon_del,
                                    double sigma_theta, int max_new_edges, unsigned int *seed);
void step5_monitor_output(const CausalGraphV2 *graph, const InformationStateV2 *state,
                           ObservablesV2 *obs);

void dynamics_full_step_v2(CausalGraphV2 *graph, InformationStateV2 *state,
                           LaplacianSpectrum *spectrum, const DynamicsParamsV2 *params,
                           const NullModelConfig *nm_config, int step, unsigned int *seed);

void compute_observables_v2(const CausalGraphV2 *graph, const InformationStateV2 *state,
                             const LaplacianSpectrum *spectrum, ObservablesV2 *obs);

int check_convergence(const ObservablesV2 *obs_history, int history_len, double threshold);

#endif
