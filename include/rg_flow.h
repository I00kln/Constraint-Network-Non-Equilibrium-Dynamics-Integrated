#ifndef RG_FLOW_H
#define RG_FLOW_H

#include "constraint_network.h"

#define MAX_BLOCKS 16
#define MAX_ITERATIONS 2000
#define THERMALIZATION_STEPS 1000
#define MEASUREMENT_STEPS 2000

typedef struct {
    int num_blocks;
    int block_size[MAX_BLOCKS];
    int block_nodes[MAX_BLOCKS][MAX_NODES];
    int node_to_block[MAX_NODES];
} CoarseGrainingMap;

typedef struct {
    Graph coarse_graph;
    PhaseSpace coarse_ps;
    double effective_beta;
    double effective_coupling;
} CoarseGrainedSystem;

typedef struct {
    double beta_values[MAX_ITERATIONS];
    double energy_values[MAX_ITERATIONS];
    int num_points;
    double beta_star;
    int fixed_point_found;
} RGFlowData;

typedef struct {
    double correlation[MAX_NODES][MAX_NODES];
    double correlation_length;
    int max_distance;
} CorrelationFunction;

typedef struct {
    double A[MAX_EDGES];
    double E[MAX_EDGES];
    double energy;
    double weight;
} MonteCarloConfiguration;

typedef struct {
    MonteCarloConfiguration configs[MAX_ITERATIONS];
    int num_configs;
    double total_weight;
    double average_energy;
    double energy_variance;
} MonteCarloEnsemble;

void init_coarse_graining_map(CoarseGrainingMap *cg_map, int num_blocks);
void assign_node_to_block(CoarseGrainingMap *cg_map, int node, int block);
void create_block_partition_tetrahedron(CoarseGrainingMap *cg_map);

void coarse_grain_graph(const Graph *fine_graph, const CoarseGrainingMap *cg_map,
                        Graph *coarse_graph);
void coarse_grain_phase_space(const PhaseSpace *fine_ps, const Graph *fine_graph,
                               const CoarseGrainingMap *cg_map, PhaseSpace *coarse_ps);

void init_monte_carlo_ensemble(MonteCarloEnsemble *ensemble);
void generate_random_configuration(MonteCarloConfiguration *config, const Graph *graph, 
                                   unsigned int *seed);
double compute_configuration_energy(const MonteCarloConfiguration *config, 
                                    const Graph *graph, double beta);
int metropolis_step(MonteCarloConfiguration *config, const Graph *graph, 
                    double beta, double delta, unsigned int *seed);
void run_monte_carlo_sampling(const Graph *graph, double beta, 
                              MonteCarloEnsemble *ensemble, int num_steps,
                              unsigned int seed);

double compute_effective_beta(const MonteCarloEnsemble *ensemble, 
                              const Graph *coarse_graph);
void compute_effective_hamiltonian(const Graph *fine_graph, 
                                   const MonteCarloEnsemble *ensemble,
                                   CoarseGrainedSystem *cg_system);

void init_rg_flow_data(RGFlowData *rg_data);
void compute_rg_flow_step(const Graph *graph, double beta_in, 
                          double *beta_out, unsigned int seed);
void compute_rg_flow(const Graph *graph, double beta_initial, 
                     RGFlowData *rg_data, int max_steps, unsigned int seed);
double find_fixed_point(const RGFlowData *rg_data);
double find_fixed_point_bisection(const Graph *graph, double beta_low, 
                                   double beta_high, double tolerance,
                                   unsigned int seed);
int verify_fixed_point(const Graph *graph, double beta_candidate, 
                       double tolerance, unsigned int seed);

void compute_correlation_function(const Graph *graph, 
                                  const MonteCarloEnsemble *ensemble,
                                  CorrelationFunction *corr);
double compute_correlation_length(const CorrelationFunction *corr, 
                                  const Graph *graph);
void measure_correlation_length_vs_beta(const Graph *graph, 
                                        const double *beta_values, int num_betas,
                                        double *correlation_lengths,
                                        unsigned int seed);

double compute_specific_heat(const MonteCarloEnsemble *ensemble);
double compute_susceptibility(const Graph *graph, 
                              const MonteCarloEnsemble *ensemble);
void compute_thermodynamic_quantities(const Graph *graph, double beta,
                                      double *specific_heat, double *susceptibility,
                                      unsigned int seed);

void print_coarse_graining_map(const CoarseGrainingMap *cg_map);
void print_rg_flow_data(const RGFlowData *rg_data);
void print_correlation_function(const CorrelationFunction *corr, int num_nodes);

void write_rg_flow_data(const RGFlowData *rg_data, const char *filename);
void write_correlation_data(const CorrelationFunction *corr, int num_nodes,
                            const char *filename);

#endif
