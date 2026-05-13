#ifndef CAUSAL_GRAPH_V2_H
#define CAUSAL_GRAPH_V2_H

#include <complex.h>

#define MAX_NODES 700
#define MAX_EDGES 12000
#define MAX_NEIGHBORS 70
#define MAX_EIGENVECTORS 40

typedef enum {
    GAUGE_U1,
    GAUGE_SU2
} GaugeGroupType;

typedef struct {
    double re;
    double im;
} Complex;

typedef struct {
    int from;
    int to;
    Complex phase;
    double weight;
} DirectedEdgeWithPhase;

typedef struct {
    int num_nodes;
    int num_edges;
    DirectedEdgeWithPhase edges[MAX_EDGES];
    
    int in_neighbors[MAX_NODES][MAX_NEIGHBORS];
    int in_edge_indices[MAX_NODES][MAX_NEIGHBORS];
    int in_degree[MAX_NODES];
    
    int out_neighbors[MAX_NODES][MAX_NEIGHBORS];
    int out_edge_indices[MAX_NODES][MAX_NEIGHBORS];
    int out_degree[MAX_NODES];
    
    int max_in_degree;
    int max_out_degree;
    
    GaugeGroupType gauge_type;
} CausalGraphV2;

typedef struct {
    double eigenvalues[MAX_EIGENVECTORS];
    double eigenvectors[MAX_NODES][MAX_EIGENVECTORS];
    int num_eigenvalues;
    int cutoff_index;
} LaplacianSpectrum;

void init_causal_graph_v2(CausalGraphV2 *graph, int num_nodes, GaugeGroupType gauge_type);
void free_causal_graph_v2(CausalGraphV2 *graph);
int add_edge_with_phase(CausalGraphV2 *graph, int from, int to, double theta);
int remove_edge_v2(CausalGraphV2 *graph, int edge_idx);
int find_edge_v2(const CausalGraphV2 *graph, int from, int to);
void generate_er_graph_v2(CausalGraphV2 *graph, double p, unsigned int *seed);
void generate_small_world_v2(CausalGraphV2 *graph, int k, double beta, unsigned int *seed);

Complex complex_multiply(Complex a, Complex b);
Complex complex_conjugate(Complex a);
double complex_modulus(Complex a);
Complex u1_phase(double theta);
void su2_random_matrix(Complex *matrix, unsigned int *seed);

void compute_laplacian_spectrum(const CausalGraphV2 *graph, LaplacianSpectrum *spectrum, int num_eigenvalues);
void project_to_macro(const LaplacianSpectrum *spectrum, const double *rho, double *macro, int num_nodes, int cutoff);
void project_to_micro(const LaplacianSpectrum *spectrum, const double *rho, double *micro, int num_nodes, int cutoff);

#endif
