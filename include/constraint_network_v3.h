#ifndef CONSTRAINT_NETWORK_V3_H
#define CONSTRAINT_NETWORK_V3_H

#include <math.h>

#define MAX_NODES 256
#define MAX_EDGES 1024
#define MAX_TRIANGLES 2048

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
} GraphV3;

typedef struct {
    double A[MAX_EDGES];
    double E[MAX_EDGES];
} PhaseSpaceV3;

typedef struct {
    double real;
    double imag;
} Complex;

double geodesic_distance_U1(double F);
double holonomy_U1(double A1, double A2, double A3);
double compute_action_S(const GraphV3 *graph, const PhaseSpaceV3 *ps);
double compute_hamiltonian_geodesic(const GraphV3 *graph, const PhaseSpaceV3 *ps, double beta);
double compute_gauss_constraint(const GraphV3 *graph, const PhaseSpaceV3 *ps, int node);
double compute_hamiltonian_constraint_node(const GraphV3 *graph, const PhaseSpaceV3 *ps, int node, double beta);
void compute_poisson_bracket_HH(const GraphV3 *graph, const PhaseSpaceV3 *ps, 
                                 double *bracket, double beta);
int check_algebra_closure(const GraphV3 *graph, const PhaseSpaceV3 *ps, 
                          double *anomaly_norm, double beta);
void init_tetrahedron_graph_v3(GraphV3 *graph);
void init_hypercubic_graph_v3(GraphV3 *graph, int L);

#endif
