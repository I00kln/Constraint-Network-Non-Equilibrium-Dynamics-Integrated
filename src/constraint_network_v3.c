#include "constraint_network_v3.h"
#include <stdlib.h>
#include <stdio.h>

double geodesic_distance_U1(double F) {
    double F_mod = fmod(F, 2.0 * M_PI);
    if (F_mod > M_PI) F_mod = 2.0 * M_PI - F_mod;
    if (F_mod < -M_PI) F_mod = -2.0 * M_PI - F_mod;
    return fabs(F_mod);
}

double holonomy_U1(double A1, double A2, double A3) {
    return A1 + A2 + A3;
}

double compute_action_S(const GraphV3 *graph, const PhaseSpaceV3 *ps) {
    double S = 0.0;
    
    for (int t = 0; t < graph->num_triangles; t++) {
        int i = graph->triangles[t].i;
        int j = graph->triangles[t].j;
        int k = graph->triangles[t].k;
        
        double F_triangle = 0.0;
        
        for (int e = 0; e < graph->num_edges; e++) {
            int ei = graph->edges[e][0];
            int ej = graph->edges[e][1];
            
            if ((ei == i && ej == j) || (ei == k && ej == i)) {
                F_triangle += ps->A[e];
            } else if ((ei == j && ej == k)) {
                F_triangle += ps->A[e];
            } else if ((ej == i && ei == j) || (ej == k && ei == i)) {
                F_triangle -= ps->A[e];
            } else if ((ej == j && ei == k)) {
                F_triangle -= ps->A[e];
            }
        }
        
        double d = geodesic_distance_U1(F_triangle);
        S += 0.5 * d * d;
    }
    
    return S;
}

double compute_hamiltonian_geodesic(const GraphV3 *graph, const PhaseSpaceV3 *ps, double beta) {
    double H = 0.0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        H += 0.5 * ps->E[e] * ps->E[e];
    }
    
    for (int t = 0; t < graph->num_triangles; t++) {
        int i = graph->triangles[t].i;
        int j = graph->triangles[t].j;
        int k = graph->triangles[t].k;
        
        double F_triangle = 0.0;
        int edge_sign[MAX_EDGES] = {0};
        
        for (int e = 0; e < graph->num_edges; e++) {
            int ei = graph->edges[e][0];
            int ej = graph->edges[e][1];
            
            if ((ei == i && ej == j)) {
                F_triangle += ps->A[e];
                edge_sign[e] = 1;
            } else if ((ei == j && ej == k)) {
                F_triangle += ps->A[e];
                edge_sign[e] = 1;
            } else if ((ei == k && ej == i)) {
                F_triangle += ps->A[e];
                edge_sign[e] = 1;
            } else if ((ej == i && ei == j)) {
                F_triangle -= ps->A[e];
                edge_sign[e] = -1;
            } else if ((ej == j && ei == k)) {
                F_triangle -= ps->A[e];
                edge_sign[e] = -1;
            } else if ((ej == k && ei == i)) {
                F_triangle -= ps->A[e];
                edge_sign[e] = -1;
            }
        }
        
        double d = geodesic_distance_U1(F_triangle);
        H += beta * d * d;
    }
    
    return H;
}

double compute_gauss_constraint(const GraphV3 *graph, const PhaseSpaceV3 *ps, int node) {
    double G = 0.0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        int i = graph->edges[e][0];
        int j = graph->edges[e][1];
        
        if (i == node) {
            G += ps->E[e];
        } else if (j == node) {
            G -= ps->E[e];
        }
    }
    
    return G;
}

double compute_hamiltonian_constraint_node(const GraphV3 *graph, const PhaseSpaceV3 *ps, 
                                           int node, double beta) {
    double H_node = 0.0;
    int deg = graph->node_degree[node];
    if (deg == 0) deg = 1;
    
    for (int e = 0; e < graph->num_edges; e++) {
        int i = graph->edges[e][0];
        int j = graph->edges[e][1];
        
        if (i == node || j == node) {
            H_node += 0.5 * ps->E[e] * ps->E[e] / deg;
        }
    }
    
    for (int t = 0; t < graph->num_triangles; t++) {
        int ti = graph->triangles[t].i;
        int tj = graph->triangles[t].j;
        int tk = graph->triangles[t].k;
        
        if (ti == node || tj == node || tk == node) {
            double F_triangle = 0.0;
            
            for (int e = 0; e < graph->num_edges; e++) {
                int ei = graph->edges[e][0];
                int ej = graph->edges[e][1];
                
                if ((ei == ti && ej == tj)) F_triangle += ps->A[e];
                else if ((ei == tj && ej == tk)) F_triangle += ps->A[e];
                else if ((ei == tk && ej == ti)) F_triangle += ps->A[e];
                else if ((ej == ti && ei == tj)) F_triangle -= ps->A[e];
                else if ((ej == tj && ei == tk)) F_triangle -= ps->A[e];
                else if ((ej == tk && ei == ti)) F_triangle -= ps->A[e];
            }
            
            double d = geodesic_distance_U1(F_triangle);
            H_node += beta * d * d / 3.0;
        }
    }
    
    return H_node;
}

void compute_poisson_bracket_HH(const GraphV3 *graph, const PhaseSpaceV3 *ps, 
                                 double *bracket, double beta) {
    double eps = 1e-6;
    PhaseSpaceV3 ps_plus, ps_minus;
    
    for (int i = 0; i < graph->num_nodes; i++) {
        for (int j = 0; j < graph->num_nodes; j++) {
            double bracket_ij = 0.0;
            
            for (int e = 0; e < graph->num_edges; e++) {
                for (int k = 0; k < graph->num_edges; k++) {
                    ps_plus = *ps;
                    ps_minus = *ps;
                    ps_plus.A[k] += eps;
                    ps_minus.A[k] -= eps;
                    
                    double H_i_plus = compute_hamiltonian_constraint_node(graph, &ps_plus, i, beta);
                    double H_i_minus = compute_hamiltonian_constraint_node(graph, &ps_minus, i, beta);
                    double dH_i_dA = (H_i_plus - H_i_minus) / (2.0 * eps);
                    
                    ps_plus = *ps;
                    ps_minus = *ps;
                    ps_plus.E[k] += eps;
                    ps_minus.E[k] -= eps;
                    
                    double H_j_plus = compute_hamiltonian_constraint_node(graph, &ps_plus, j, beta);
                    double H_j_minus = compute_hamiltonian_constraint_node(graph, &ps_minus, j, beta);
                    double dH_j_dE = (H_j_plus - H_j_minus) / (2.0 * eps);
                    
                    ps_plus = *ps;
                    ps_minus = *ps;
                    ps_plus.E[k] += eps;
                    ps_minus.E[k] -= eps;
                    
                    H_i_plus = compute_hamiltonian_constraint_node(graph, &ps_plus, i, beta);
                    H_i_minus = compute_hamiltonian_constraint_node(graph, &ps_minus, i, beta);
                    double dH_i_dE = (H_i_plus - H_i_minus) / (2.0 * eps);
                    
                    ps_plus = *ps;
                    ps_minus = *ps;
                    ps_plus.A[k] += eps;
                    ps_minus.A[k] -= eps;
                    
                    H_j_plus = compute_hamiltonian_constraint_node(graph, &ps_plus, j, beta);
                    H_j_minus = compute_hamiltonian_constraint_node(graph, &ps_minus, j, beta);
                    double dH_j_dA = (H_j_plus - H_j_minus) / (2.0 * eps);
                    
                    double delta = (e == k) ? 1.0 : 0.0;
                    bracket_ij += delta * (dH_i_dA * dH_j_dE - dH_i_dE * dH_j_dA);
                }
            }
            
            bracket[i * graph->num_nodes + j] = bracket_ij;
        }
    }
}

int check_algebra_closure(const GraphV3 *graph, const PhaseSpaceV3 *ps, 
                          double *anomaly_norm, double beta) {
    int N = graph->num_nodes;
    double *bracket = (double *)malloc(N * N * sizeof(double));
    double *G_constraints = (double *)malloc(N * sizeof(double));
    
    compute_poisson_bracket_HH(graph, ps, bracket, beta);
    
    for (int i = 0; i < N; i++) {
        G_constraints[i] = compute_gauss_constraint(graph, ps, i);
    }
    
    double total_anomaly = 0.0;
    int closed = 1;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double bracket_ij = bracket[i * N + j];
            
            double projected = 0.0;
            for (int k = 0; k < N; k++) {
                double f_ijk = 0.0;
                if (k == i) f_ijk = bracket_ij / (G_constraints[j] + 1e-10);
                else if (k == j) f_ijk = -bracket_ij / (G_constraints[i] + 1e-10);
                projected += f_ijk * G_constraints[k];
            }
            
            double anomaly = bracket_ij - projected;
            total_anomaly += anomaly * anomaly;
            
            if (fabs(anomaly) > 1e-4) {
                closed = 0;
            }
        }
    }
    
    *anomaly_norm = sqrt(total_anomaly);
    
    free(bracket);
    free(G_constraints);
    
    return closed;
}

void init_tetrahedron_graph_v3(GraphV3 *graph) {
    graph->num_nodes = 4;
    graph->num_edges = 6;
    graph->num_triangles = 4;
    
    graph->edges[0][0] = 0; graph->edges[0][1] = 1;
    graph->edges[1][0] = 1; graph->edges[1][1] = 2;
    graph->edges[2][0] = 2; graph->edges[2][1] = 0;
    graph->edges[3][0] = 0; graph->edges[3][1] = 3;
    graph->edges[4][0] = 1; graph->edges[4][1] = 3;
    graph->edges[5][0] = 2; graph->edges[5][1] = 3;
    
    graph->triangles[0].i = 0; graph->triangles[0].j = 1; graph->triangles[0].k = 2;
    graph->triangles[1].i = 0; graph->triangles[1].j = 1; graph->triangles[1].k = 3;
    graph->triangles[2].i = 1; graph->triangles[2].j = 2; graph->triangles[2].k = 3;
    graph->triangles[3].i = 2; graph->triangles[3].j = 0; graph->triangles[3].k = 3;
    
    for (int i = 0; i < 4; i++) {
        graph->node_degree[i] = 3;
    }
}

void init_hypercubic_graph_v3(GraphV3 *graph, int L) {
    graph->num_nodes = L * L;
    graph->num_edges = 2 * L * L;
    graph->num_triangles = 0;
    
    int edge_idx = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int node = i * L + j;
            int right = i * L + ((j + 1) % L);
            int up = ((i + 1) % L) * L + j;
            
            graph->edges[edge_idx][0] = node;
            graph->edges[edge_idx][1] = right;
            edge_idx++;
            
            graph->edges[edge_idx][0] = node;
            graph->edges[edge_idx][1] = up;
            edge_idx++;
        }
    }
    
    for (int i = 0; i < L * L; i++) {
        graph->node_degree[i] = 4;
    }
}
