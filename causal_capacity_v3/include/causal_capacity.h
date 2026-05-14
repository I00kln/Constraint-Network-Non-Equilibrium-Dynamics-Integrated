#ifndef CAUSAL_CAPACITY_H
#define CAUSAL_CAPACITY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_NODES 500
#define MAX_EDGES 20000
#define MAX_NEIGHBORS 100
#define MAX_EIGENVALUES 50
#define LOCAL_DIM 4
#define MAX_GREEN_STEPS 500
#define EFFECTIVE_DIM 4

typedef struct {
    int from;
    int to;
    double capacity;
    double weight;
    double flow;
} CausalEdge;

typedef struct {
    int num_nodes;
    int num_edges;
    CausalEdge edges[MAX_EDGES];

    int in_neighbors[MAX_NODES][MAX_NEIGHBORS];
    int in_edge_idx[MAX_NODES][MAX_NEIGHBORS];
    int in_degree[MAX_NODES];

    int out_neighbors[MAX_NODES][MAX_NEIGHBORS];
    int out_edge_idx[MAX_NODES][MAX_NEIGHBORS];
    int out_degree[MAX_NODES];

    int has_cycle;
    int cycle_count;
} CausalGraph;

typedef struct {
    double rho[MAX_NODES];
    double rho_macro[MAX_NODES];
    double rho_micro[MAX_NODES];
    double total_info;
    int num_nodes;
} InfoState;

typedef struct {
    double eigenvalues[MAX_EIGENVALUES];
    int num_eigenvalues;
    int negative_count;
    int positive_count;
    int zero_count;
    int lorentz_signature_detected;
    double max_negative_eigenvalue;
    double min_positive_eigenvalue;
    double signature_ratio;
    double spectral_gap;
    double asymmetry_measure;
    double neg_fraction;
    double coarse_grained_eigenvalues[LOCAL_DIM];
    int coarse_grained_signs[LOCAL_DIM];
    int coarse_grained_lorentz;
    double coarse_grained_neg_fraction;
} PropagationSpectrum;

typedef struct {
    double matrix[LOCAL_DIM][LOCAL_DIM];
    double eigenvalues[LOCAL_DIM];
    int sign_pattern[LOCAL_DIM];
    int dim;
    int lorentz_signature;
    double asymmetry_norm;
    double response_strength;
} LocalResponseTensor;

typedef struct {
    double eigenvalues[MAX_NODES][LOCAL_DIM];
    int sign_pattern[MAX_NODES][LOCAL_DIM];
    int lorentz_nodes;
    int euclidean_nodes;
    int other_nodes;
    double lorentz_fraction;
    double avg_asymmetry;
    double avg_response_strength;
    double avg_tensor[LOCAL_DIM][LOCAL_DIM];
    double avg_tensor_eigenvalues[LOCAL_DIM];
    int avg_tensor_signs[LOCAL_DIM];
    int avg_tensor_lorentz;
} LocalPCAResult;

typedef struct {
    double response[MAX_NODES][MAX_GREEN_STEPS];
    int origin_node;
    int max_steps;
    double light_cone_velocity;
    int light_cone_detected;
    double cone_sharpness;
    double anisotropy;
    double front_width;
} GreenFunction;

typedef struct {
    double eta;
    double lambda_param;
    double alpha;
    double beta;
    double epsilon_del;
    double sigma;
    double cap_learning_rate;
    int max_new_edges;
    int intervention_samples;
    int use_conservation;
} DynamicsParams;

typedef struct {
    double entropy;
    double total_info;
    int num_edges;
    int lorentz_detected;
    int light_cone_detected;
    double pde_type_indicator;
    double lorentz_fraction;
    double spectral_gap;
    double signature_ratio;
    int neg_eigenvalues;
    int pos_eigenvalues;
    double asymmetry_measure;
    double avg_response_strength;
    int cycle_count;
} Observables;

void init_causal_graph(CausalGraph *graph, int num_nodes);
int add_edge(CausalGraph *graph, int from, int to, double capacity);
int remove_edge(CausalGraph *graph, int edge_idx);
int find_edge(const CausalGraph *graph, int from, int to);
void generate_er_graph(CausalGraph *graph, double p, unsigned int *seed);
void generate_directed_acyclic_graph(CausalGraph *graph, double p, unsigned int *seed);
void generate_small_world_graph(CausalGraph *graph, int k, double rewire_prob, unsigned int *seed);
int detect_cycles(CausalGraph *graph);

void init_info_state(InfoState *state, int num_nodes);
void set_uniform_density(InfoState *state, double total);

double estimate_causal_capacity(CausalGraph *graph, InfoState *state, int from, int to, double delta, int samples, unsigned int *seed);
void step1_causal_propagation(CausalGraph *graph, InfoState *state, double eta);
void step2_macro_micro_decomposition(CausalGraph *graph, InfoState *state, double lambda_param);
void step3_redistribution(InfoState *state, double lambda_param, double target_total);
void step4_topology_rewiring(CausalGraph *graph, InfoState *state, double alpha, double beta, double epsilon_del, double sigma, int max_new_edges, unsigned int *seed);
void step5_capacity_evolution(CausalGraph *graph, InfoState *state, double learning_rate);
void dynamics_full_step(CausalGraph *graph, InfoState *state, const DynamicsParams *params, unsigned int *seed);

void compute_intervention_response(CausalGraph *graph, InfoState *state, double eta, double delta, double *response_matrix);
void compute_local_response_tensor(CausalGraph *graph, InfoState *state, int node_v, double eta, double delta, LocalResponseTensor *tensor);
void compute_propagation_operator_spectrum(const CausalGraph *graph, PropagationSpectrum *spectrum, int num_eigenvalues);
void compute_response_spectrum(CausalGraph *graph, InfoState *state, double eta, double delta, PropagationSpectrum *spectrum, int num_eigenvalues);
void compute_local_pca(CausalGraph *graph, InfoState *state, double eta, double delta, LocalPCAResult *result);
void compute_green_function(CausalGraph *graph, InfoState *state, GreenFunction *gf, int origin_node, int max_steps, double delta, double eta);
void compute_observables(CausalGraph *graph, InfoState *state, const DynamicsParams *params, Observables *obs);

int check_lorentz_signature(const PropagationSpectrum *spectrum);
int check_light_cone(const GreenFunction *gf);
double classify_pde_type(const PropagationSpectrum *spectrum);

typedef struct {
    double time_eigenvector[MAX_NODES];
    double topo_order[MAX_NODES];
    double correlation;
    double spearman_correlation;
    int time_direction_node;
    double time_eigenvalue;
    double max_positive_eigenvalue;
    int is_aligned;
    double alignment_score;
} TimeDirectionResult;

typedef struct {
    double scale;
    double lorentz_fraction;
    double neg_count;
    double pos_count;
    int stable_signature;
    double signature_deviation;
} CoarsePCA;

typedef struct {
    double eta;
    double lambda_param;
    double pde_type;
    int is_hyperbolic;
    int is_elliptic;
    int is_parabolic;
    double neg_eigenvalue_count;
    double pos_eigenvalue_count;
    double spectral_gap;
    double asymmetry;
} PDETypeInfo;

void detect_time_direction(CausalGraph *graph, InfoState *state, double eta, double delta, TimeDirectionResult *result);
void compute_coarse_pca(CausalGraph *graph, InfoState *state, double eta, double delta, CoarsePCA *results, int *num_results);
void analyze_pde_transition(CausalGraph *graph, InfoState *state, PDETypeInfo *results, int *num_results);
double compute_topological_order_correlation(CausalGraph *graph, const double *eigenvector, int n);

typedef struct {
    double metric[EFFECTIVE_DIM][EFFECTIVE_DIM];
    double inverse_metric[EFFECTIVE_DIM][EFFECTIVE_DIM];
    double eigenvalues[EFFECTIVE_DIM];
    int signature[EFFECTIVE_DIM];
    double determinant;
    double ricci_scalar;
    double curvature_2d;
    int is_minkowski;
    int is_lorentzian;
    double minkowski_deviation;
    double effective_c;
} EmergentMetric;

typedef struct {
    double scale;
    double metric[EFFECTIVE_DIM][EFFECTIVE_DIM];
    double eigenvalues[EFFECTIVE_DIM];
    int signature[EFFECTIVE_DIM];
    double lorentz_fraction;
    double neg_count;
    double pos_count;
    double anisotropy;
    double effective_c;
    double minkowski_deviation;
} CoarseGrainLevel;

typedef struct {
    int num_levels;
    CoarseGrainLevel levels[20];
    double rg_flow_lorentz[20];
    double rg_flow_c[20];
    double rg_flow_anisotropy[20];
    double fixed_point_metric[EFFECTIVE_DIM][EFFECTIVE_DIM];
    int has_fixed_point;
    double fixed_point_deviation;
} RGFlowResult;

void extract_emergent_metric(CausalGraph *graph, InfoState *state, double eta, double delta, EmergentMetric *metric);
void compute_coarse_graining(CausalGraph *graph, InfoState *state, double eta, double delta, RGFlowResult *rg);
double compute_curvature(const EmergentMetric *metric);
double compute_minkowski_deviation(const EmergentMetric *metric);

typedef struct {
    double R_coefficient;
    double R2_coefficient;
    double R_ijkl_coefficient;
    double total_action;
    double kinetic_term;
    double potential_term;
    int is_einstein_hilbert;
    double eh_deviation;
} EffectiveAction;

typedef struct {
    int N;
    double G;
    double G_predicted;
    double G_deviation;
    double scaling_exponent;
    double correlation;
} GravityCoupling;

typedef struct {
    double r;
    double phi;
    double phi_newton;
    double deviation;
    int count;
} NewtonPotential;

typedef struct {
    double capacity_fluctuation;
    double metric_response;
    double G_from_fluctuation;
    double G_from_response;
    double fluctuation_dissipation_ratio;
    int satisfies_fd_theorem;
} FluctuationDissipation;

void compute_effective_action(CausalGraph *graph, InfoState *state, double eta, double delta, EffectiveAction *action);
void compute_gravity_coupling(CausalGraph *graphs[], InfoState *states[], int num_sizes, int sizes[], GravityCoupling *gc);
void compute_newton_potential(CausalGraph *graph, InfoState *state, int source_node, NewtonPotential *potential, int *num_points);
void check_fluctuation_dissipation(CausalGraph *graph, InfoState *state, double eta, double delta, FluctuationDissipation *fd);
double compute_ricci_tensor_component(const EmergentMetric *metric, int mu, int nu);

#define MAX_CYCLE_LENGTH 20
#define MAX_CYCLES 1000

typedef struct {
    int nodes[MAX_CYCLE_LENGTH];
    int length;
    double action;
    double winding_number;
    int is_stable;
    int homotopy_class;
    double mass;
    double charge;
} ClosedCycle;

typedef struct {
    int num_cycles;
    ClosedCycle cycles[MAX_CYCLES];
    double min_action;
    double max_action;
    double avg_action;
    int num_stable;
    int num_homotopy_classes;
} CycleSpectrum;

typedef struct {
    int dim;
    double generators[10][10];
    double structure_constants[10][10][10];
    char name[32];
    int is_compact;
    int is_simple;
} LieAlgebra;

typedef struct {
    int num_factors;
    LieAlgebra factors[5];
    double total_dim;
    int matches_standard_model;
} GaugeGroup;

typedef struct {
    double mass;
    double charge;
    int generation;
    int spin;
    int color;
    int is_fermion;
    char name[32];
} ParticleState;

typedef struct {
    int num_particles;
    ParticleState particles[100];
    int num_generations;
    int has_quarks;
    int has_leptons;
    int matches_standard_model;
} ParticleSpectrum;

typedef struct {
    double real[3][3];
    double imag[3][3];
    double jarlskog;
    double unitarity_violation;
    char name[32];
} MixingMatrix;

void find_closed_cycles(CausalGraph *graph, CycleSpectrum *spectrum);
double compute_cycle_action(CausalGraph *graph, InfoState *state, const ClosedCycle *cycle);
void classify_stable_cycles(CycleSpectrum *spectrum);
void analyze_gauge_structure(CausalGraph *graph, GaugeGroup *group);
void compute_particle_spectrum(CycleSpectrum *spectrum, ParticleSpectrum *ps);
int check_standard_model_match(const ParticleSpectrum *ps);
void compute_ckm_matrix(CycleSpectrum *spectrum, MixingMatrix *ckm);
void compute_pmns_matrix(CycleSpectrum *spectrum, MixingMatrix *pmns);
double compute_jarlskog_invariant(const MixingMatrix *m);

#endif
