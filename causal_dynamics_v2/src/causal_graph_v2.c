#include "causal_graph_v2.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void init_causal_graph_v2(CausalGraphV2 *graph, int num_nodes, GaugeGroupType gauge_type) {
    memset(graph, 0, sizeof(CausalGraphV2));
    graph->num_nodes = num_nodes;
    graph->num_edges = 0;
    graph->gauge_type = gauge_type;
    graph->max_in_degree = 0;
    graph->max_out_degree = 0;
    
    for (int i = 0; i < num_nodes; i++) {
        graph->in_degree[i] = 0;
        graph->out_degree[i] = 0;
    }
}

void free_causal_graph_v2(CausalGraphV2 *graph) {
    memset(graph, 0, sizeof(CausalGraphV2));
}

Complex complex_multiply(Complex a, Complex b) {
    Complex result;
    result.re = a.re * b.re - a.im * b.im;
    result.im = a.re * b.im + a.im * b.re;
    return result;
}

Complex complex_conjugate(Complex a) {
    Complex result;
    result.re = a.re;
    result.im = -a.im;
    return result;
}

double complex_modulus(Complex a) {
    return sqrt(a.re * a.re + a.im * a.im);
}

Complex u1_phase(double theta) {
    Complex result;
    result.re = cos(theta);
    result.im = sin(theta);
    return result;
}

void su2_random_matrix(Complex *matrix, unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    double theta = ((double)(*seed % RAND_MAX) / RAND_MAX) * M_PI;
    *seed = *seed * 1103515245 + 12345;
    double phi = ((double)(*seed % RAND_MAX) / RAND_MAX) * 2.0 * M_PI;
    *seed = *seed * 1103515245 + 12345;
    double psi = ((double)(*seed % RAND_MAX) / RAND_MAX) * 2.0 * M_PI;
    
    matrix[0] = u1_phase(phi);
    matrix[0].re *= cos(theta);
    matrix[0].im *= cos(theta);
    
    matrix[1] = u1_phase(psi);
    matrix[1].re *= sin(theta);
    matrix[1].im *= sin(theta);
}

int find_edge_v2(const CausalGraphV2 *graph, int from, int to) {
    for (int i = 0; i < graph->num_edges; i++) {
        if (graph->edges[i].from == from && graph->edges[i].to == to) {
            return i;
        }
    }
    return -1;
}

int add_edge_with_phase(CausalGraphV2 *graph, int from, int to, double theta) {
    if (from < 0 || from >= graph->num_nodes) return -1;
    if (to < 0 || to >= graph->num_nodes) return -1;
    if (graph->num_edges >= MAX_EDGES) return -1;
    
    int existing = find_edge_v2(graph, from, to);
    if (existing >= 0) {
        graph->edges[existing].phase = u1_phase(theta);
        return existing;
    }
    
    int edge_idx = graph->num_edges;
    graph->edges[edge_idx].from = from;
    graph->edges[edge_idx].to = to;
    graph->edges[edge_idx].phase = u1_phase(theta);
    graph->edges[edge_idx].weight = 1.0;
    
    int in_deg = graph->in_degree[to];
    if (in_deg < MAX_NEIGHBORS) {
        graph->in_neighbors[to][in_deg] = from;
        graph->in_edge_indices[to][in_deg] = edge_idx;
        graph->in_degree[to]++;
        if (graph->in_degree[to] > graph->max_in_degree) {
            graph->max_in_degree = graph->in_degree[to];
        }
    }
    
    int out_deg = graph->out_degree[from];
    if (out_deg < MAX_NEIGHBORS) {
        graph->out_neighbors[from][out_deg] = to;
        graph->out_edge_indices[from][out_deg] = edge_idx;
        graph->out_degree[from]++;
        if (graph->out_degree[from] > graph->max_out_degree) {
            graph->max_out_degree = graph->out_degree[from];
        }
    }
    
    graph->num_edges++;
    return edge_idx;
}

int remove_edge_v2(CausalGraphV2 *graph, int edge_idx) {
    if (edge_idx < 0 || edge_idx >= graph->num_edges) return -1;
    
    int from = graph->edges[edge_idx].from;
    int to = graph->edges[edge_idx].to;
    
    for (int i = 0; i < graph->in_degree[to]; i++) {
        if (graph->in_edge_indices[to][i] == edge_idx) {
            for (int j = i; j < graph->in_degree[to] - 1; j++) {
                graph->in_neighbors[to][j] = graph->in_neighbors[to][j + 1];
                graph->in_edge_indices[to][j] = graph->in_edge_indices[to][j + 1];
            }
            graph->in_degree[to]--;
            break;
        }
    }
    
    for (int i = 0; i < graph->out_degree[from]; i++) {
        if (graph->out_edge_indices[from][i] == edge_idx) {
            for (int j = i; j < graph->out_degree[from] - 1; j++) {
                graph->out_neighbors[from][j] = graph->out_neighbors[from][j + 1];
                graph->out_edge_indices[from][j] = graph->out_edge_indices[from][j + 1];
            }
            graph->out_degree[from]--;
            break;
        }
    }
    
    if (edge_idx < graph->num_edges - 1) {
        graph->edges[edge_idx] = graph->edges[graph->num_edges - 1];
        
        int new_from = graph->edges[edge_idx].from;
        int new_to = graph->edges[edge_idx].to;
        
        for (int i = 0; i < graph->in_degree[new_to]; i++) {
            if (graph->in_edge_indices[new_to][i] == graph->num_edges - 1) {
                graph->in_edge_indices[new_to][i] = edge_idx;
                break;
            }
        }
        
        for (int i = 0; i < graph->out_degree[new_from]; i++) {
            if (graph->out_edge_indices[new_from][i] == graph->num_edges - 1) {
                graph->out_edge_indices[new_from][i] = edge_idx;
                break;
            }
        }
    }
    
    graph->num_edges--;
    return 0;
}

static double rand_uniform(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (double)(*seed % RAND_MAX) / RAND_MAX;
}

void generate_er_graph_v2(CausalGraphV2 *graph, double p, unsigned int *seed) {
    for (int u = 0; u < graph->num_nodes; u++) {
        for (int v = 0; v < graph->num_nodes; v++) {
            if (u != v && rand_uniform(seed) < p) {
                double theta = rand_uniform(seed) * 2.0 * M_PI;
                add_edge_with_phase(graph, u, v, theta);
            }
        }
    }
}

void generate_small_world_v2(CausalGraphV2 *graph, int k, double beta, unsigned int *seed) {
    for (int i = 0; i < graph->num_nodes; i++) {
        for (int j = 1; j <= k / 2; j++) {
            int target = (i + j) % graph->num_nodes;
            double theta = rand_uniform(seed) * 2.0 * M_PI;
            add_edge_with_phase(graph, i, target, theta);
            
            target = (i - j + graph->num_nodes) % graph->num_nodes;
            theta = rand_uniform(seed) * 2.0 * M_PI;
            add_edge_with_phase(graph, i, target, theta);
        }
    }
    
    int num_rewire = (int)(beta * graph->num_edges);
    for (int r = 0; r < num_rewire && graph->num_edges > 0; r++) {
        int edge_idx = (int)(rand_uniform(seed) * graph->num_edges);
        if (edge_idx >= graph->num_edges) edge_idx = graph->num_edges - 1;
        
        int from = graph->edges[edge_idx].from;
        int new_to = (int)(rand_uniform(seed) * graph->num_nodes);
        if (new_to >= graph->num_nodes) new_to = graph->num_nodes - 1;
        
        if (from != new_to && find_edge_v2(graph, from, new_to) < 0) {
            remove_edge_v2(graph, edge_idx);
            double theta = rand_uniform(seed) * 2.0 * M_PI;
            add_edge_with_phase(graph, from, new_to, theta);
        }
    }
}

static void normalize_vector(double *v, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);
    if (norm > 1e-15) {
        for (int i = 0; i < n; i++) {
            v[i] /= norm;
        }
    }
}

static double dot_product(const double *v1, const double *v2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

static void orthogonalize(double *v, const double *basis, int n) {
    double coeff = dot_product(v, basis, n);
    for (int i = 0; i < n; i++) {
        v[i] -= coeff * basis[i];
    }
}

static void apply_laplacian(const CausalGraphV2 *graph, const double *v_in, double *v_out, int n) {
    for (int i = 0; i < n; i++) {
        double degree_i = graph->in_degree[i] + graph->out_degree[i];
        v_out[i] = degree_i * v_in[i];
    }
    
    #pragma omp parallel for
    for (int e = 0; e < graph->num_edges; e++) {
        int u = graph->edges[e].from;
        int v = graph->edges[e].to;
        if (u < n && v < n) {
            #pragma omp atomic
            v_out[u] -= v_in[v];
            #pragma omp atomic
            v_out[v] -= v_in[u];
        }
    }
}

void compute_laplacian_spectrum(const CausalGraphV2 *graph, LaplacianSpectrum *spectrum, int num_eigenvalues) {
    memset(spectrum, 0, sizeof(LaplacianSpectrum));
    
    int n = graph->num_nodes;
    if (num_eigenvalues > MAX_EIGENVECTORS) num_eigenvalues = MAX_EIGENVECTORS;
    if (num_eigenvalues > n) num_eigenvalues = n;
    
    static double v[MAX_NODES];
    static double Lv[MAX_NODES];
    static double temp[MAX_NODES];
    
    spectrum->eigenvalues[0] = 0.0;
    for (int i = 0; i < n; i++) {
        spectrum->eigenvectors[i][0] = 1.0 / sqrt((double)n);
    }
    spectrum->num_eigenvalues = 1;
    
    for (int k = 1; k < num_eigenvalues; k++) {
        for (int i = 0; i < n; i++) {
            v[i] = ((double)rand() / RAND_MAX - 0.5);
        }
        
        for (int j = 0; j < k; j++) {
            orthogonalize(v, spectrum->eigenvectors[j], n);
        }
        normalize_vector(v, n);
        
        double lambda = 0.0;
        int max_iter = 100;
        
        for (int iter = 0; iter < max_iter; iter++) {
            apply_laplacian(graph, v, Lv, n);
            
            for (int j = 0; j < k; j++) {
                double proj = dot_product(Lv, spectrum->eigenvectors[j], n);
                for (int i = 0; i < n; i++) {
                    Lv[i] -= proj * spectrum->eigenvectors[i][j];
                }
            }
            
            lambda = dot_product(v, Lv, n);
            
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += Lv[i] * Lv[i];
            }
            norm = sqrt(norm);
            
            if (norm > 1e-15) {
                for (int i = 0; i < n; i++) {
                    temp[i] = v[i];
                    v[i] = Lv[i] / norm;
                }
            }
            
            double diff = 0.0;
            for (int i = 0; i < n; i++) {
                diff += (v[i] - temp[i]) * (v[i] - temp[i]);
            }
            
            if (sqrt(diff) < 1e-8 && iter > 10) {
                break;
            }
        }
        
        spectrum->eigenvalues[k] = lambda;
        for (int i = 0; i < n; i++) {
            spectrum->eigenvectors[i][k] = v[i];
        }
        spectrum->num_eigenvalues++;
    }
    
    spectrum->cutoff_index = num_eigenvalues / 5;
}

void project_to_macro(const LaplacianSpectrum *spectrum, const double *rho, double *macro, int num_nodes, int cutoff) {
    for (int i = 0; i < num_nodes; i++) {
        macro[i] = 0.0;
    }
    
    for (int k = 0; k < cutoff && k < spectrum->num_eigenvalues; k++) {
        double coeff = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            coeff += rho[i] * spectrum->eigenvectors[i][k];
        }
        
        for (int i = 0; i < num_nodes; i++) {
            macro[i] += coeff * spectrum->eigenvectors[i][k];
        }
    }
    
    for (int i = 0; i < num_nodes; i++) {
        if (macro[i] < -10.0) macro[i] = -10.0;
        if (macro[i] > 10.0) macro[i] = 10.0;
    }
}

void project_to_micro(const LaplacianSpectrum *spectrum, const double *rho, double *micro, int num_nodes, int cutoff) {
    static double macro[MAX_NODES];
    project_to_macro(spectrum, rho, macro, num_nodes, cutoff);
    
    for (int i = 0; i < num_nodes; i++) {
        micro[i] = rho[i] - macro[i];
        
        if (micro[i] < -10.0) micro[i] = -10.0;
        if (micro[i] > 10.0) micro[i] = 10.0;
    }
}
