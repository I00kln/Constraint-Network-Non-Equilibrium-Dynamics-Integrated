#include "local_constraints.h"
#include "phase_space.h"
#include <math.h>

void compute_local_hamiltonian_constraints(const Graph *graph, const PhaseSpace *ps,
                                           LocalHamiltonian *local_H, double beta) {
    for (int n = 0; n < graph->num_nodes; n++) {
        local_H->H_kin_node[n] = 0.0;
        local_H->H_curv_node[n] = 0.0;
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        int from = graph->edges[e].i;
        int to = graph->edges[e].j;
        double E_sq = ps->E[e] * ps->E[e] / ps->q[e];
        
        local_H->H_kin_node[from] += 0.5 * E_sq;
        local_H->H_kin_node[to] += 0.5 * E_sq;
    }
    
    for (int f = 0; f < graph->num_faces; f++) {
        double F = compute_face_curvature(graph, ps, f);
        double curv_contrib = -beta * cos(F);
        
        for (int i = 0; i < 3; i++) {
            int node = graph->faces[f].nodes[i];
            local_H->H_curv_node[node] += curv_contrib / 3.0;
        }
    }
    
    for (int n = 0; n < graph->num_nodes; n++) {
        int deg = graph->node_edge_count[n];
        local_H->H_kin_node[n] /= deg;
        local_H->H_curv_node[n] /= deg;
        local_H->H_node[n] = local_H->H_kin_node[n] + local_H->H_curv_node[n];
    }
}

void compute_decoherence_correction(const Graph *graph, const PhaseSpace *ps,
                                     double *G_sq_sum, double gamma) {
    *G_sq_sum = 0.0;
    
    for (int n = 0; n < graph->num_nodes; n++) {
        double G_n = 0.0;
        for (int i = 0; i < graph->node_edge_count[n]; i++) {
            int e = graph->node_edges[n][i];
            int from = graph->edges[e].i;
            if (from == n) {
                G_n += ps->E[e];
            } else {
                G_n -= ps->E[e];
            }
        }
        *G_sq_sum += G_n * G_n;
    }
}

void compute_local_hamiltonian_with_decoherence(const Graph *graph, const PhaseSpace *ps,
                                                 LocalHamiltonian *local_H, 
                                                 double beta, double gamma) {
    compute_local_hamiltonian_constraints(graph, ps, local_H, beta);
    
    double G_sq_sum;
    compute_decoherence_correction(graph, ps, &G_sq_sum, gamma);
    
    for (int n = 0; n < graph->num_nodes; n++) {
        local_H->H_node[n] += gamma * G_sq_sum;
    }
}

double compute_poisson_bracket_Hi_Hj(const Graph *graph, const PhaseSpace *ps,
                                     int i, int j, double beta, double epsilon) {
    static PhaseSpace ps_plus, ps_minus;
    static LocalHamiltonian H_plus, H_minus;
    
    double bracket = 0.0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        double A_orig = ps->A[e];
        
        ps_plus = *ps;
        ps_plus.A[e] = A_orig + epsilon;
        compute_local_hamiltonian_constraints(graph, &ps_plus, &H_plus, beta);
        
        ps_minus = *ps;
        ps_minus.A[e] = A_orig - epsilon;
        compute_local_hamiltonian_constraints(graph, &ps_minus, &H_minus, beta);
        
        double dHi_dA = (H_plus.H_node[i] - H_minus.H_node[i]) / (2.0 * epsilon);
        double dHj_dA = (H_plus.H_node[j] - H_minus.H_node[j]) / (2.0 * epsilon);
        
        double E_orig = ps->E[e];
        
        ps_plus = *ps;
        ps_plus.E[e] = E_orig + epsilon;
        compute_local_hamiltonian_constraints(graph, &ps_plus, &H_plus, beta);
        
        ps_minus = *ps;
        ps_minus.E[e] = E_orig - epsilon;
        compute_local_hamiltonian_constraints(graph, &ps_minus, &H_minus, beta);
        
        double dHi_dE = (H_plus.H_node[i] - H_minus.H_node[i]) / (2.0 * epsilon);
        double dHj_dE = (H_plus.H_node[j] - H_minus.H_node[j]) / (2.0 * epsilon);
        
        bracket += dHi_dA * dHj_dE - dHi_dE * dHj_dA;
    }
    
    return bracket;
}

double compute_poisson_bracket_Hi_Hj_with_decoherence(const Graph *graph, const PhaseSpace *ps,
                                                       int i, int j, double beta, 
                                                       double gamma, double epsilon) {
    static PhaseSpace ps_plus, ps_minus;
    static LocalHamiltonian H_plus, H_minus;
    
    double bracket = 0.0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        double A_orig = ps->A[e];
        
        ps_plus = *ps;
        ps_plus.A[e] = A_orig + epsilon;
        compute_local_hamiltonian_with_decoherence(graph, &ps_plus, &H_plus, beta, gamma);
        
        ps_minus = *ps;
        ps_minus.A[e] = A_orig - epsilon;
        compute_local_hamiltonian_with_decoherence(graph, &ps_minus, &H_minus, beta, gamma);
        
        double dHi_dA = (H_plus.H_node[i] - H_minus.H_node[i]) / (2.0 * epsilon);
        double dHj_dA = (H_plus.H_node[j] - H_minus.H_node[j]) / (2.0 * epsilon);
        
        double E_orig = ps->E[e];
        
        ps_plus = *ps;
        ps_plus.E[e] = E_orig + epsilon;
        compute_local_hamiltonian_with_decoherence(graph, &ps_plus, &H_plus, beta, gamma);
        
        ps_minus = *ps;
        ps_minus.E[e] = E_orig - epsilon;
        compute_local_hamiltonian_with_decoherence(graph, &ps_minus, &H_minus, beta, gamma);
        
        double dHi_dE = (H_plus.H_node[i] - H_minus.H_node[i]) / (2.0 * epsilon);
        double dHj_dE = (H_plus.H_node[j] - H_minus.H_node[j]) / (2.0 * epsilon);
        
        bracket += dHi_dA * dHj_dE - dHi_dE * dHj_dA;
    }
    
    return bracket;
}

void compute_poisson_matrix(const Graph *graph, const PhaseSpace *ps,
                            PoissonMatrix *pm, double beta, double epsilon) {
    for (int i = 0; i < graph->num_nodes; i++) {
        for (int j = 0; j < graph->num_nodes; j++) {
            pm->poisson[i][j] = compute_poisson_bracket_Hi_Hj(graph, ps, i, j, beta, epsilon);
        }
    }
}

void compute_poisson_matrix_with_decoherence(const Graph *graph, const PhaseSpace *ps,
                                             PoissonMatrix *pm, double beta, 
                                             double gamma, double epsilon) {
    for (int i = 0; i < graph->num_nodes; i++) {
        for (int j = 0; j < graph->num_nodes; j++) {
            pm->poisson[i][j] = compute_poisson_bracket_Hi_Hj_with_decoherence(
                graph, ps, i, j, beta, gamma, epsilon);
        }
    }
}

void print_local_hamiltonian(const LocalHamiltonian *local_H, int num_nodes) {
    printf("=== Local Hamiltonian Constraints ===\n");
    printf("Node    H_kin        H_curv       H_total\n");
    for (int n = 0; n < num_nodes; n++) {
        printf("%3d    %10.6f  %10.6f  %10.6f\n",
               n, local_H->H_kin_node[n], local_H->H_curv_node[n], local_H->H_node[n]);
    }
}

void print_poisson_matrix(const PoissonMatrix *pm, int num_nodes) {
    printf("\n=== Poisson Bracket Matrix {H_i, H_j} ===\n");
    printf("        ");
    for (int j = 0; j < num_nodes; j++) {
        printf("%10d ", j);
    }
    printf("\n");
    
    for (int i = 0; i < num_nodes; i++) {
        printf("%3d    ", i);
        for (int j = 0; j < num_nodes; j++) {
            printf("%10.6f ", pm->poisson[i][j]);
        }
        printf("\n");
    }
}

int verify_constraint_algebra_closure(const PoissonMatrix *pm, int num_nodes, 
                                      double tolerance, double *max_violation) {
    *max_violation = 0.0;
    int violations = 0;
    
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            double abs_val = fabs(pm->poisson[i][j]);
            if (abs_val > tolerance) {
                violations++;
                if (abs_val > *max_violation) {
                    *max_violation = abs_val;
                }
            }
        }
    }
    
    return violations;
}
