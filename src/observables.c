#include "observables.h"

double compute_gauss_violation(const Constraints *constraints, int num_nodes) {
    double violation = 0.0;
    for (int n = 0; n < num_nodes; n++) {
        violation += constraints->G[n] * constraints->G[n];
    }
    return violation;
}

double compute_max_gauss(const Constraints *constraints, int num_nodes) {
    double max_g = 0.0;
    for (int n = 0; n < num_nodes; n++) {
        double abs_g = fabs(constraints->G[n]);
        if (abs_g > max_g) {
            max_g = abs_g;
        }
    }
    return max_g;
}

void update_simulation_state(SimulationState *state, const Constraints *constraints,
                             int num_nodes, double initial_energy, double time) {
    state->time = time;
    state->gauss_violation = compute_gauss_violation(constraints, num_nodes);
    state->max_gauss = compute_max_gauss(constraints, num_nodes);
    state->energy = constraints->H_total;
    state->energy_drift = (constraints->H_total - initial_energy) / 
                          (fabs(initial_energy) + 1e-10);
}

void write_simulation_data(FILE *fp, const SimulationState *state) {
    fprintf(fp, "%12.6f %16.12f %16.12f %16.12f %16.12f\n",
            state->time, state->gauss_violation, state->max_gauss,
            state->energy, state->energy_drift);
}
