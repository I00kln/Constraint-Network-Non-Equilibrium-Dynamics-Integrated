#ifndef CONSTRAINT_NETWORK_V4_H
#define CONSTRAINT_NETWORK_V4_H

#include <math.h>

#define MAX_NODES 256
#define MAX_EDGES 1024
#define MAX_TRIANGLES 2048
#define MAX_FACES 2048

typedef struct {
    int i, j, k;
} Triangle;

typedef struct {
    int num_nodes;
    int num_edges;
    int num_triangles;
    int edges[MAX_EDGES][2];
    Triangle triangles[MAX_TRIANGLES];
    int node_degree[MAX_NODES];
    double edge_weight[MAX_EDGES];
} GraphV4;

typedef struct {
    double A[MAX_EDGES];
    double E[MAX_EDGES];
} PhaseSpaceV4;

typedef struct {
    double Omega[MAX_EDGES*2][MAX_EDGES*2];
    double Gamma[MAX_EDGES*2][MAX_EDGES*2];
    double G[MAX_EDGES*2][MAX_EDGES*2];
    double K[MAX_EDGES*2][MAX_EDGES*2];
    double M[MAX_EDGES*2][MAX_EDGES*2];
} GeometricStructures;

double geodesic_distance_U1(double F);
double compute_hamiltonian(const GraphV4 *graph, const PhaseSpaceV4 *ps, double beta);
double compute_gauss_constraint(const GraphV4 *graph, const PhaseSpaceV4 *ps, int node);
double compute_hamiltonian_constraint(const GraphV4 *graph, const PhaseSpaceV4 *ps, int node, double beta);
double compute_geometric_constraint_distance(const GraphV4 *graph, const PhaseSpaceV4 *ps, double beta);
void compute_constraint_manifold_projection(const GraphV4 *graph, PhaseSpaceV4 *ps, double beta, double dt);
void compute_symplectic_structure(const GraphV4 *graph, GeometricStructures *gs);
void compute_dissipation_tensor(const GraphV4 *graph, GeometricStructures *gs, double gamma);
void mixed_flow_step(PhaseSpaceV4 *ps, const GraphV4 *graph, const GeometricStructures *gs, 
                     double beta, double dt, double gamma);
void compute_hodge_laplacian(const GraphV4 *graph, double *laplacian, int size);
double compute_spectral_dimension(const GraphV4 *graph, double t);
void compute_effective_hessian(const GraphV4 *graph, const PhaseSpaceV4 *ps, 
                               GeometricStructures *gs, double beta);
int count_morse_index(const GeometricStructures *gs, int size);
int check_krein_collision(const GeometricStructures *gs, int size, double omega);
double compute_correlation_length(const GraphV4 *graph, const PhaseSpaceV4 *ps);
int check_reflection_positivity(const GraphV4 *graph, const PhaseSpaceV4 *ps, int num_samples);
void init_tetrahedron_graph_v4(GraphV4 *graph);
void init_hypercubic_graph_v4(GraphV4 *graph, int L);

#endif
