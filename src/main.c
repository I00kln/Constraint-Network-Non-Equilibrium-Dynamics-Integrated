#include "constraint_network.h"
#include "graph_topology.h"
#include "phase_space.h"
#include "integrator.h"
#include "observables.h"

void print_simulation_header(const SimulationParams *params, int integrator_type) {
    printf("\n=== Phase I: Classical Constraint Dynamics Simulation ===\n");
    printf("Parameters:\n");
    printf("  Beta (coupling):     %8.4f\n", params->beta);
    printf("  Time step:           %8.6f\n", params->dt);
    printf("  Number of steps:     %8d\n", params->num_steps);
    printf("  Output interval:     %8d\n", params->output_interval);
    printf("  Integrator:          ");
    switch(integrator_type) {
        case 0: printf("Verlet (2nd order)\n"); break;
        case 1: printf("Symplectic RK4 (4th order)\n"); break;
        case 2: printf("Forest-Ruth (4th order)\n"); break;
        default: printf("Unknown\n");
    }
    printf("\n");
}

void print_progress(int step, int total_steps) {
    if (step % (total_steps / 10) == 0 || step == total_steps) {
        printf("Progress: %6.2f%% (step %d / %d)\n", 
               100.0 * step / total_steps, step, total_steps);
    }
}

int main(int argc, char *argv[]) {
    Graph graph;
    PhaseSpace ps;
    Constraints constraints;
    SimulationParams params;
    SimulationState state;
    int integrator_type = 0;
    
    params.beta = 1.0;
    params.dt = 0.001;
    params.num_steps = 10000;
    params.output_interval = 100;
    
    if (argc > 1) params.beta = atof(argv[1]);
    if (argc > 2) params.dt = atof(argv[2]);
    if (argc > 3) params.num_steps = atoi(argv[3]);
    if (argc > 4) integrator_type = atoi(argv[4]);
    
    init_tetrahedron_graph(&graph);
    print_graph_info(&graph);
    
    print_simulation_header(&params, integrator_type);
    
    init_phase_space(&ps, &graph, 12345);
    print_phase_space(&ps, graph.num_edges);
    
    compute_constraints(&graph, &ps, &constraints, params.beta);
    print_constraints(&constraints, graph.num_nodes);
    
    double initial_energy = constraints.H_total;
    double initial_gauss_violation = compute_gauss_violation(&constraints, graph.num_nodes);
    
    printf("\nInitial state:\n");
    printf("  Energy:              %16.12f\n", initial_energy);
    printf("  Gauss violation:     %16.12f\n", initial_gauss_violation);
    
    FILE *fp = fopen("data/simulation_output.dat", "w");
    if (!fp) {
        printf("Error: Cannot open output file!\n");
        return 1;
    }
    
    fprintf(fp, "# Time  Gauss_violation  Max_Gauss  Energy  Energy_drift\n");
    
    printf("\nStarting simulation...\n");
    
    for (int step = 0; step <= params.num_steps; step++) {
        double time = step * params.dt;
        
        compute_constraints(&graph, &ps, &constraints, params.beta);
        update_simulation_state(&state, &constraints, graph.num_nodes, 
                               initial_energy, time);
        
        if (step % params.output_interval == 0) {
            write_simulation_data(fp, &state);
            print_progress(step, params.num_steps);
        }
        
        if (step < params.num_steps) {
            switch(integrator_type) {
                case 0:
                    verlet_step(&graph, &ps, params.dt, params.beta);
                    break;
                case 1:
                    symplectic_rk4_step(&graph, &ps, params.dt, params.beta);
                    break;
                case 2:
                    forest_ruth_step(&graph, &ps, params.dt, params.beta);
                    break;
                default:
                    verlet_step(&graph, &ps, params.dt, params.beta);
            }
        }
    }
    
    fclose(fp);
    
    printf("\n=== Final Results ===\n");
    printf("Final Gauss violation:     %16.12f\n", state.gauss_violation);
    printf("Final energy drift:        %16.12f\n", state.energy_drift);
    printf("Max Gauss constraint:      %16.12f\n", state.max_gauss);
    
    printf("\n=== Success Criteria Check ===\n");
    int gauss_ok = (state.gauss_violation < 1e-4);
    int energy_ok = (fabs(state.energy_drift) < 1e-6);
    
    printf("Gauss violation < 1e-4:    %s\n", gauss_ok ? "PASS" : "FAIL");
    printf("Energy drift < 1e-6:       %s\n", energy_ok ? "PASS" : "FAIL");
    
    if (gauss_ok && energy_ok) {
        printf("\n*** Phase I simulation PASSED all criteria ***\n");
    } else {
        printf("\n*** Phase I simulation FAILED some criteria ***\n");
    }
    
    printf("\nData saved to: data/simulation_output.dat\n");
    
    return 0;
}
