#ifndef PHASE_SPACE_H
#define PHASE_SPACE_H

#include "constraint_network.h"

void init_phase_space(PhaseSpace *ps, const Graph *graph, int seed);
void compute_constraints(const Graph *graph, const PhaseSpace *ps, 
                         Constraints *constraints, double beta, double hbar_eff);
double compute_face_curvature(const Graph *graph, const PhaseSpace *ps, int face_idx);
double compute_ricci_curvature(double F);
void compute_causal_depth(const Graph *graph, const PhaseSpace *ps, 
                          Constraints *constraints, int reference_node);
void compute_diffusion_constraint(const Graph *graph, const PhaseSpace *ps,
                                   Constraints *constraints, double hbar_eff);
void compute_hamiltonian_derivatives(const Graph *graph, const PhaseSpace *ps,
                                     double *dH_dA, double *dH_dE, 
                                     double *dH_dq, double *dH_dp, 
                                     double beta, double alpha);
void print_phase_space(const PhaseSpace *ps, int num_edges);
void print_constraints(const Constraints *constraints, int num_nodes, int num_edges, int num_faces);

#endif
