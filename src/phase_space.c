#include "phase_space.h"
#include "graph_topology.h"
#include <time.h>

void init_phase_space(PhaseSpace *ps, const Graph *graph, int seed) {
    srand(seed);
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->A[e] = ((double)rand() / RAND_MAX) * TWO_PI;
        ps->E[e] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        ps->q[e] = 1.0 + 0.1 * ((double)rand() / RAND_MAX - 0.5);
        ps->p[e] = 0.1 * ((double)rand() / RAND_MAX - 0.5);
    }
    
    for (int n = 0; n < graph->num_nodes; n++) {
        double sum = 0.0;
        for (int i = 0; i < graph->node_edge_count[n]; i++) {
            int e = graph->node_edges[n][i];
            int from = graph->edges[e].i;
            if (from == n) {
                sum += ps->E[e];
            } else {
                sum -= ps->E[e];
            }
        }
        
        if (graph->node_edge_count[n] > 0) {
            int e = graph->node_edges[n][0];
            int from = graph->edges[e].i;
            if (from == n) {
                ps->E[e] -= sum;
            } else {
                ps->E[e] += sum;
            }
        }
    }
}

double compute_face_curvature(const Graph *graph, const PhaseSpace *ps, int face_idx) {
    const Face *face = &graph->faces[face_idx];
    double F = 0.0;
    
    for (int i = 0; i < 3; i++) {
        int n1 = face->nodes[i];
        int n2 = face->nodes[(i + 1) % 3];
        
        int e = -1;
        for (int j = 0; j < graph->num_edges; j++) {
            if ((graph->edges[j].i == n1 && graph->edges[j].j == n2) ||
                (graph->edges[j].i == n2 && graph->edges[j].j == n1)) {
                e = j;
                break;
            }
        }
        
        if (e == -1) {
            continue;
        }
        
        int from = graph->edges[e].i;
        int to = graph->edges[e].j;
        
        if (from == n1 && to == n2) {
            F += ps->A[e];
        } else if (from == n2 && to == n1) {
            F -= ps->A[e];
        }
    }
    
    while (F > PI) F -= TWO_PI;
    while (F < -PI) F += TWO_PI;
    
    return F;
}

double compute_ricci_curvature(double F) {
    return 1.0 - cos(F);
}

void compute_causal_depth(const Graph *graph, const PhaseSpace *ps, 
                          Constraints *constraints, int reference_node) {
    for (int n = 0; n < graph->num_nodes; n++) {
        constraints->tau[n] = 0.0;
    }
    
    constraints->tau[reference_node] = 0.0;
    
    for (int iter = 0; iter < graph->num_nodes; iter++) {
        for (int f = 0; f < graph->num_faces; f++) {
            double F = compute_face_curvature(graph, ps, f);
            double Ric = compute_ricci_curvature(F);
            
            for (int i = 0; i < 3; i++) {
                int n1 = graph->faces[f].nodes[i];
                int n2 = graph->faces[f].nodes[(i + 1) % 3];
                
                if (constraints->tau[n1] > 0 || n1 == reference_node) {
                    double new_tau = constraints->tau[n1] + Ric;
                    if (constraints->tau[n2] < new_tau) {
                        constraints->tau[n2] = new_tau;
                    }
                }
                if (constraints->tau[n2] > 0 || n2 == reference_node) {
                    double new_tau = constraints->tau[n2] + Ric;
                    if (constraints->tau[n1] < new_tau) {
                        constraints->tau[n1] = new_tau;
                    }
                }
            }
        }
    }
}

void compute_diffusion_constraint(const Graph *graph, const PhaseSpace *ps,
                                   Constraints *constraints, double hbar_eff) {
    for (int e = 0; e < graph->num_edges; e++) {
        constraints->S[e] = ps->p[e] * ps->q[e] - hbar_eff;
    }
}

void compute_constraints(const Graph *graph, const PhaseSpace *ps, 
                         Constraints *constraints, double beta, double hbar_eff) {
    for (int n = 0; n < graph->num_nodes; n++) {
        constraints->G[n] = 0.0;
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        int from = graph->edges[e].i;
        int to = graph->edges[e].j;
        
        constraints->G[from] += ps->E[e];
        constraints->G[to] -= ps->E[e];
    }
    
    compute_diffusion_constraint(graph, ps, constraints, hbar_eff);
    
    constraints->H_kin = 0.0;
    constraints->H_qp = 0.0;
    double alpha = 1.0;
    for (int e = 0; e < graph->num_edges; e++) {
        constraints->H_kin += 0.5 * ps->E[e] * ps->E[e] / ps->q[e];
        constraints->H_qp += 0.5 * alpha * ps->p[e] * ps->p[e] / ps->q[e];
    }
    
    constraints->H_curv = 0.0;
    for (int f = 0; f < graph->num_faces; f++) {
        double F = compute_face_curvature(graph, ps, f);
        constraints->Ric[f] = compute_ricci_curvature(F);
        constraints->H_curv -= beta * cos(F);
    }
    
    constraints->H_total = constraints->H_kin + constraints->H_qp + constraints->H_curv;
    
    compute_causal_depth(graph, ps, constraints, 0);
}

void compute_hamiltonian_derivatives(const Graph *graph, const PhaseSpace *ps,
                                     double *dH_dA, double *dH_dE, 
                                     double *dH_dq, double *dH_dp, 
                                     double beta, double alpha) {
    for (int e = 0; e < graph->num_edges; e++) {
        dH_dE[e] = ps->E[e] / ps->q[e];
        dH_dA[e] = 0.0;
        dH_dp[e] = alpha * ps->p[e] / ps->q[e];
        dH_dq[e] = -0.5 * ps->E[e] * ps->E[e] / (ps->q[e] * ps->q[e])
                   - 0.5 * alpha * ps->p[e] * ps->p[e] / (ps->q[e] * ps->q[e]);
    }
    
    for (int f = 0; f < graph->num_faces; f++) {
        const Face *face = &graph->faces[f];
        double F = compute_face_curvature(graph, ps, f);
        double dH_dF = beta * sin(F);
        
        for (int i = 0; i < 3; i++) {
            int n1 = face->nodes[i];
            int n2 = face->nodes[(i + 1) % 3];
            
            int e = -1;
            for (int j = 0; j < graph->num_edges; j++) {
                if ((graph->edges[j].i == n1 && graph->edges[j].j == n2) ||
                    (graph->edges[j].i == n2 && graph->edges[j].j == n1)) {
                    e = j;
                    break;
                }
            }
            
            if (e == -1) continue;
            
            int from = graph->edges[e].i;
            int to = graph->edges[e].j;
            
            if (from == n1 && to == n2) {
                dH_dA[e] += dH_dF;
            } else if (from == n2 && to == n1) {
                dH_dA[e] -= dH_dF;
            }
        }
    }
}

void print_phase_space(const PhaseSpace *ps, int num_edges) {
    printf("=== Phase Space Variables ===\n");
    printf("Edge    A (angle)    E (field)    q (strength)    p (momentum)\n");
    for (int e = 0; e < num_edges; e++) {
        printf("%3d    %8.5f    %10.6f    %10.6f    %10.6f\n", 
               e, ps->A[e], ps->E[e], ps->q[e], ps->p[e]);
    }
}

void print_constraints(const Constraints *constraints, int num_nodes, int num_edges, int num_faces) {
    printf("=== Constraints ===\n");
    printf("Gauss constraints:\n");
    for (int n = 0; n < num_nodes; n++) {
        printf("  G[%d] = %12.8f\n", n, constraints->G[n]);
    }
    
    printf("\nDiffusion constraints:\n");
    for (int e = 0; e < num_edges && e < 6; e++) {
        printf("  S[%d] = %12.8f\n", e, constraints->S[e]);
    }
    
    printf("\nCausal depth (tau):\n");
    for (int n = 0; n < num_nodes; n++) {
        printf("  tau[%d] = %12.8f\n", n, constraints->tau[n]);
    }
    
    printf("\nRicci curvature:\n");
    for (int f = 0; f < num_faces; f++) {
        printf("  Ric[%d] = %12.8f\n", f, constraints->Ric[f]);
    }
    
    printf("\nHamiltonian:\n");
    printf("  H_kin  = %12.8f\n", constraints->H_kin);
    printf("  H_curv = %12.8f\n", constraints->H_curv);
    printf("  H_tot  = %12.8f\n", constraints->H_total);
}
