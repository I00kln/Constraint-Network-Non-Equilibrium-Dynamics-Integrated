#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "constraint_network.h"

void set_alpha(double alpha);
void verlet_step(Graph *graph, PhaseSpace *ps, double dt, double beta);
void symplectic_rk4_step(Graph *graph, PhaseSpace *ps, double dt, double beta);
void symplectic_rk4_step_with_projection(Graph *graph, PhaseSpace *ps, double dt, 
                                          double beta, double hbar_eff);
void forest_ruth_step(Graph *graph, PhaseSpace *ps, double dt, double beta);
void normalize_angles(PhaseSpace *ps, int num_edges);
void enforce_q_positive(PhaseSpace *ps, int num_edges);
void project_diffusion_constraint(PhaseSpace *ps, int num_edges, double hbar_eff);

#endif
