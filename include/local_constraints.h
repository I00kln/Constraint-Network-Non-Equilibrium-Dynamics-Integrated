#ifndef LOCAL_CONSTRAINTS_H
#define LOCAL_CONSTRAINTS_H

#include "constraint_network.h"

typedef struct {
    double H_node[MAX_NODES];
    double H_kin_node[MAX_NODES];
    double H_curv_node[MAX_NODES];
} LocalHamiltonian;

typedef struct {
    double poisson[MAX_NODES][MAX_NODES];
} PoissonMatrix;

void compute_local_hamiltonian_constraints(const Graph *graph, const PhaseSpace *ps,
                                           LocalHamiltonian *local_H, double beta);

void compute_decoherence_correction(const Graph *graph, const PhaseSpace *ps,
                                     double *G_sq_sum, double gamma);

void compute_local_hamiltonian_with_decoherence(const Graph *graph, const PhaseSpace *ps,
                                                 LocalHamiltonian *local_H, 
                                                 double beta, double gamma);

double compute_poisson_bracket_Hi_Hj(const Graph *graph, const PhaseSpace *ps,
                                     int i, int j, double beta, double epsilon);

double compute_poisson_bracket_Hi_Hj_with_decoherence(const Graph *graph, const PhaseSpace *ps,
                                                       int i, int j, double beta, 
                                                       double gamma, double epsilon);

void compute_poisson_matrix(const Graph *graph, const PhaseSpace *ps,
                            PoissonMatrix *pm, double beta, double epsilon);

void compute_poisson_matrix_with_decoherence(const Graph *graph, const PhaseSpace *ps,
                                             PoissonMatrix *pm, double beta, 
                                             double gamma, double epsilon);

void print_local_hamiltonian(const LocalHamiltonian *local_H, int num_nodes);
void print_poisson_matrix(const PoissonMatrix *pm, int num_nodes);

int verify_constraint_algebra_closure(const PoissonMatrix *pm, int num_nodes, 
                                      double tolerance, double *max_violation);

#endif
