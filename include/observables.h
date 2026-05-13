#ifndef OBSERVABLES_H
#define OBSERVABLES_H

#include "constraint_network.h"

double compute_gauss_violation(const Constraints *constraints, int num_nodes);
double compute_max_gauss(const Constraints *constraints, int num_nodes);
void update_simulation_state(SimulationState *state, const Constraints *constraints,
                             int num_nodes, double initial_energy, double time);
void write_simulation_data(FILE *fp, const SimulationState *state);

#endif
