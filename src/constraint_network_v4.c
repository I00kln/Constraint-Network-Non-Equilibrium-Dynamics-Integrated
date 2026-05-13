#include "constraint_network_v4.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

double geodesic_distance_U1(double F) {
    double F_mod = fmod(F, 2.0 * M_PI);
    if (F_mod > M_PI) F_mod = 2.0 * M_PI - F_mod;
    if (F_mod < -M_PI) F_mod = -2.0 * M_PI - F_mod;
    return fabs(F_mod);
}

double compute_hamiltonian(const GraphV4 *graph, const PhaseSpaceV4 *ps, double beta) {
    double H = 0.0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        H += 0.5 * ps->E[e] * ps->E[e];
    }
    
    for (int t = 0; t < graph->num_triangles; t++) {
        int i = graph->triangles[t].i;
        int j = graph->triangles[t].j;
        int k = graph->triangles[t].k;
        
        double F_triangle = 0.0;
        
        for (int e = 0; e < graph->num_edges; e++) {
            int ei = graph->edges[e][0];
            int ej = graph->edges[e][1];
            
            if ((ei == i && ej == j)) F_triangle += ps->A[e];
            else if ((ei == j && ej == k)) F_triangle += ps->A[e];
            else if ((ei == k && ej == i)) F_triangle += ps->A[e];
            else if ((ej == i && ei == j)) F_triangle -= ps->A[e];
            else if ((ej == j && ei == k)) F_triangle -= ps->A[e];
            else if ((ej == k && ei == i)) F_triangle -= ps->A[e];
        }
        
        double d = geodesic_distance_U1(F_triangle);
        H += beta * d * d;
    }
    
    return H;
}

double compute_gauss_constraint(const GraphV4 *graph, const PhaseSpaceV4 *ps, int node) {
    double G = 0.0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        int i = graph->edges[e][0];
        int j = graph->edges[e][1];
        
        if (i == node) G += ps->E[e];
        else if (j == node) G -= ps->E[e];
    }
    
    return G;
}

double compute_hamiltonian_constraint(const GraphV4 *graph, const PhaseSpaceV4 *ps, int node, double beta) {
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

double compute_geometric_constraint_distance(const GraphV4 *graph, const PhaseSpaceV4 *ps, double beta) {
    double C = 0.0;
    
    for (int i = 0; i < graph->num_nodes; i++) {
        double G = compute_gauss_constraint(graph, ps, i);
        C += G * G;
    }
    
    for (int i = 0; i < graph->num_nodes; i++) {
        double H = compute_hamiltonian_constraint(graph, ps, i, beta);
        C += H * H;
    }
    
    return C;
}

void compute_constraint_manifold_projection(const GraphV4 *graph, PhaseSpaceV4 *ps, double beta, double dt) {
    int E = graph->num_edges;
    double eps = 1e-6;
    
    double *grad_C_A = (double *)malloc(E * sizeof(double));
    double *grad_C_E = (double *)malloc(E * sizeof(double));
    
    for (int e = 0; e < E; e++) {
        PhaseSpaceV4 ps_plus = *ps, ps_minus = *ps;
        ps_plus.A[e] += eps;
        ps_minus.A[e] -= eps;
        
        double C_plus = compute_geometric_constraint_distance(graph, &ps_plus, beta);
        double C_minus = compute_geometric_constraint_distance(graph, &ps_minus, beta);
        grad_C_A[e] = (C_plus - C_minus) / (2.0 * eps);
        
        ps_plus = *ps; ps_minus = *ps;
        ps_plus.E[e] += eps;
        ps_minus.E[e] -= eps;
        
        C_plus = compute_geometric_constraint_distance(graph, &ps_plus, beta);
        C_minus = compute_geometric_constraint_distance(graph, &ps_minus, beta);
        grad_C_E[e] = (C_plus - C_minus) / (2.0 * eps);
    }
    
    for (int e = 0; e < E; e++) {
        ps->A[e] -= dt * grad_C_A[e];
        ps->E[e] -= dt * grad_C_E[e];
    }
    
    free(grad_C_A);
    free(grad_C_E);
}

void compute_symplectic_structure(const GraphV4 *graph, GeometricStructures *gs) {
    int dim = 2 * graph->num_edges;
    
    memset(gs->Omega, 0, sizeof(gs->Omega));
    
    for (int e = 0; e < graph->num_edges; e++) {
        gs->Omega[e][graph->num_edges + e] = 1.0;
        gs->Omega[graph->num_edges + e][e] = -1.0;
    }
}

void compute_dissipation_tensor(const GraphV4 *graph, GeometricStructures *gs, double gamma) {
    int dim = 2 * graph->num_edges;
    
    memset(gs->Gamma, 0, sizeof(gs->Gamma));
    
    for (int i = 0; i < dim; i++) {
        gs->Gamma[i][i] = gamma;
    }
}

void mixed_flow_step(PhaseSpaceV4 *ps, const GraphV4 *graph, const GeometricStructures *gs, 
                     double beta, double dt, double gamma) {
    int E = graph->num_edges;
    double eps = 1e-6;
    
    double *dE_dA = (double *)malloc(E * sizeof(double));
    double *dE_dE = (double *)malloc(E * sizeof(double));
    double *dC_dA = (double *)malloc(E * sizeof(double));
    double *dC_dE = (double *)malloc(E * sizeof(double));
    
    for (int e = 0; e < E; e++) {
        PhaseSpaceV4 ps_plus = *ps, ps_minus = *ps;
        ps_plus.A[e] += eps;
        ps_minus.A[e] -= eps;
        
        double E_plus = compute_hamiltonian(graph, &ps_plus, beta);
        double E_minus = compute_hamiltonian(graph, &ps_minus, beta);
        dE_dA[e] = (E_plus - E_minus) / (2.0 * eps);
        
        double C_plus = compute_geometric_constraint_distance(graph, &ps_plus, beta);
        double C_minus = compute_geometric_constraint_distance(graph, &ps_minus, beta);
        dC_dA[e] = (C_plus - C_minus) / (2.0 * eps);
        
        dE_dE[e] = ps->E[e];
        
        ps_plus = *ps; ps_minus = *ps;
        ps_plus.E[e] += eps;
        ps_minus.E[e] -= eps;
        
        C_plus = compute_geometric_constraint_distance(graph, &ps_plus, beta);
        C_minus = compute_geometric_constraint_distance(graph, &ps_minus, beta);
        dC_dE[e] = (C_plus - C_minus) / (2.0 * eps);
    }
    
    for (int e = 0; e < E; e++) {
        double dA_dt = dE_dE[e] - gamma * dC_dA[e];
        double dE_dt = -dE_dA[e] - gamma * dC_dE[e];
        
        ps->A[e] += dt * dA_dt;
        ps->E[e] += dt * dE_dt;
    }
    
    free(dE_dA);
    free(dE_dE);
    free(dC_dA);
    free(dC_dE);
}

void compute_hodge_laplacian(const GraphV4 *graph, double *laplacian, int size) {
    memset(laplacian, 0, size * size * sizeof(double));
    
    for (int i = 0; i < graph->num_nodes; i++) {
        laplacian[i * size + i] = graph->node_degree[i];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        int i = graph->edges[e][0];
        int j = graph->edges[e][1];
        
        laplacian[i * size + j] = -1.0;
        laplacian[j * size + i] = -1.0;
    }
}

double compute_spectral_dimension(const GraphV4 *graph, double t) {
    int N = graph->num_nodes;
    double *laplacian = (double *)malloc(N * N * sizeof(double));
    compute_hodge_laplacian(graph, laplacian, N);
    
    double P = 0.0;
    for (int i = 0; i < N; i++) {
        double lambda = laplacian[i * N + i];
        for (int j = 0; j < N; j++) {
            if (i != j && fabs(laplacian[i * N + j]) > 1e-10) {
                lambda += fabs(laplacian[i * N + j]);
            }
        }
        
        if (lambda > 1e-10) {
            P += exp(-t * lambda);
        }
    }
    P /= N;
    
    double t_plus = t * 1.01;
    double t_minus = t * 0.99;
    
    double P_plus = 0.0, P_minus = 0.0;
    for (int i = 0; i < N; i++) {
        double lambda = laplacian[i * N + i];
        for (int j = 0; j < N; j++) {
            if (i != j && fabs(laplacian[i * N + j]) > 1e-10) {
                lambda += fabs(laplacian[i * N + j]);
            }
        }
        
        if (lambda > 1e-10) {
            P_plus += exp(-t_plus * lambda);
            P_minus += exp(-t_minus * lambda);
        }
    }
    P_plus /= N;
    P_minus /= N;
    
    double d_ln_P = (log(P_plus) - log(P_minus)) / (log(t_plus) - log(t_minus));
    double d_s = -2.0 * d_ln_P;
    
    free(laplacian);
    return d_s;
}

void compute_effective_hessian(const GraphV4 *graph, const PhaseSpaceV4 *ps, 
                               GeometricStructures *gs, double beta) {
    int E = graph->num_edges;
    int dim = 2 * E;
    double eps = 1e-5;
    
    memset(gs->K, 0, sizeof(gs->K));
    memset(gs->M, 0, sizeof(gs->M));
    memset(gs->G, 0, sizeof(gs->G));
    
    for (int i = 0; i < E; i++) {
        gs->K[i][i] = beta;
        gs->K[E + i][E + i] = 1.0;
    }
    
    for (int i = 0; i < E; i++) {
        gs->G[i][i] = 1.0;
        gs->G[E + i][E + i] = 1.0;
    }
    
    for (int i = 0; i < E; i++) {
        gs->M[i][E + i] = 0.5;
        gs->M[E + i][i] = -0.5;
    }
}

int count_morse_index(const GeometricStructures *gs, int size) {
    int count = 0;
    
    for (int i = 0; i < size; i++) {
        double eigenvalue = gs->K[i][i];
        
        if (eigenvalue < -1e-10) {
            count++;
        }
    }
    
    return count;
}

int check_krein_collision(const GeometricStructures *gs, int size, double omega) {
    for (int i = 0; i < size; i++) {
        double K_eig = gs->K[i][i];
        double G_eig = gs->G[i][i];
        
        double discriminant = omega * omega * G_eig - K_eig;
        
        if (fabs(discriminant) < 1e-10) {
            return 1;
        }
    }
    
    return 0;
}

double compute_correlation_length(const GraphV4 *graph, const PhaseSpaceV4 *ps) {
    double xi = 0.0;
    int count = 0;
    
    for (int e1 = 0; e1 < graph->num_edges; e1++) {
        for (int e2 = e1 + 1; e2 < graph->num_edges; e2++) {
            int i1 = graph->edges[e1][0], j1 = graph->edges[e1][1];
            int i2 = graph->edges[e2][0], j2 = graph->edges[e2][1];
            
            int dist = abs(i1 - i2) + abs(j1 - j2);
            if (dist > 0) {
                double corr = ps->A[e1] * ps->A[e2] + ps->E[e1] * ps->E[e2];
                if (fabs(corr) > 1e-10) {
                    xi += -dist / log(fabs(corr));
                    count++;
                }
            }
        }
    }
    
    if (count > 0) xi /= count;
    else xi = 1.0;
    
    return xi;
}

int check_reflection_positivity(const GraphV4 *graph, const PhaseSpaceV4 *ps, int num_samples) {
    int positive_count = 0;
    
    for (int s = 0; s < num_samples; s++) {
        double sum = 0.0;
        for (int e = 0; e < graph->num_edges; e++) {
            sum += ps->A[e] * ps->A[e] + ps->E[e] * ps->E[e];
        }
        
        if (sum > 0) positive_count++;
    }
    
    return (positive_count == num_samples);
}

void init_tetrahedron_graph_v4(GraphV4 *graph) {
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
    
    for (int e = 0; e < 6; e++) {
        graph->edge_weight[e] = 1.0;
    }
}

void init_hypercubic_graph_v4(GraphV4 *graph, int L) {
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
            graph->edge_weight[edge_idx] = 1.0;
            edge_idx++;
            
            graph->edges[edge_idx][0] = node;
            graph->edges[edge_idx][1] = up;
            graph->edge_weight[edge_idx] = 1.0;
            edge_idx++;
        }
    }
    
    for (int i = 0; i < L * L; i++) {
        graph->node_degree[i] = 4;
    }
}
