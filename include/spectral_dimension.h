#ifndef SPECTRAL_DIMENSION_H
#define SPECTRAL_DIMENSION_H

#include "constraint_network.h"
#include "rg_flow.h"

#define MAX_WALK_STEPS 20000
#define NUM_WALKERS 2000
#define MAX_EIGENVALUES 20

typedef struct {
    int current_node;
    int steps_taken;
    int visited[MAX_NODES];
    int visit_count[MAX_NODES];
} RandomWalker;

typedef struct {
    double return_probability[MAX_WALK_STEPS];
    int num_steps;
    double spectral_dimension;
    double fit_error;
} SpectralDimensionResult;

typedef struct {
    double laplacian[MAX_NODES][MAX_NODES];
    double eigenvalues[MAX_EIGENVALUES];
    double eigenvectors[MAX_EIGENVALUES][MAX_NODES];
    int num_eigenvalues;
} LaplacianSpectrum;

typedef struct {
    double distance_matrix[MAX_NODES][MAX_NODES];
    int graph_diameter;
    double average_distance;
} GraphGeometry;

void init_random_walker(RandomWalker *walker, int start_node);
int random_walk_step(RandomWalker *walker, const Graph *graph);
void run_random_walk(const Graph *graph, int num_steps, RandomWalker *walker);

void compute_return_probability(const Graph *graph, int num_walkers, int max_steps,
                                double *return_prob, unsigned int seed);
double fit_spectral_dimension(const double *return_prob, int num_steps, 
                              int fit_start, int fit_end);
void compute_spectral_dimension(const Graph *graph, int num_walkers, int max_steps,
                                SpectralDimensionResult *result, unsigned int seed);

void compute_graph_laplacian(const Graph *graph, double laplacian[MAX_NODES][MAX_NODES]);
void compute_laplacian_spectrum(const Graph *graph, LaplacianSpectrum *spectrum);
double compute_spectral_dimension_from_spectrum(const LaplacianSpectrum *spectrum, 
                                                double t_min, double t_max);

void compute_graph_geometry(const Graph *graph, GraphGeometry *geometry);
double compute_hausdorff_dimension(const Graph *graph);

void print_spectral_result(const SpectralDimensionResult *result);
void print_laplacian_spectrum(const LaplacianSpectrum *spectrum, int num_nodes);
void print_graph_geometry(const GraphGeometry *geometry, int num_nodes);

void write_spectral_dimension_data(const SpectralDimensionResult *result, 
                                   const char *filename);
void write_laplacian_data(const LaplacianSpectrum *spectrum, int num_nodes,
                          const char *filename);

#endif
