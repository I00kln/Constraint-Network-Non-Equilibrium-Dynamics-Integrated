#include "spectral_dimension.h"
#include "graph_topology.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void init_random_walker(RandomWalker *walker, int start_node) {
    walker->current_node = start_node;
    walker->steps_taken = 0;
    for (int i = 0; i < MAX_NODES; i++) {
        walker->visited[i] = 0;
        walker->visit_count[i] = 0;
    }
    walker->visited[start_node] = 1;
    walker->visit_count[start_node] = 1;
}

int random_walk_step(RandomWalker *walker, const Graph *graph) {
    int current = walker->current_node;
    
    int neighbors[MAX_NODES];
    int num_neighbors = 0;
    
    for (int e = 0; e < graph->num_edges; e++) {
        if (graph->edges[e].i == current) {
            neighbors[num_neighbors++] = graph->edges[e].j;
        } else if (graph->edges[e].j == current) {
            neighbors[num_neighbors++] = graph->edges[e].i;
        }
    }
    
    if (num_neighbors == 0) {
        return current;
    }
    
    int next = neighbors[rand() % num_neighbors];
    
    walker->current_node = next;
    walker->steps_taken++;
    walker->visited[next] = 1;
    walker->visit_count[next]++;
    
    return next;
}

void run_random_walk(const Graph *graph, int num_steps, RandomWalker *walker) {
    for (int s = 0; s < num_steps; s++) {
        random_walk_step(walker, graph);
    }
}

void compute_return_probability(const Graph *graph, int num_walkers, int max_steps,
                                double *return_prob, unsigned int seed) {
    srand(seed);
    
    for (int t = 0; t < max_steps; t++) {
        return_prob[t] = 0.0;
    }
    
    for (int w = 0; w < num_walkers; w++) {
        int start_node = rand() % graph->num_nodes;
        
        RandomWalker walker;
        init_random_walker(&walker, start_node);
        
        for (int t = 0; t < max_steps; t++) {
            random_walk_step(&walker, graph);
            
            if (walker.current_node == start_node) {
                return_prob[t] += 1.0;
            }
        }
    }
    
    for (int t = 0; t < max_steps; t++) {
        return_prob[t] /= num_walkers;
    }
}

double fit_spectral_dimension(const double *return_prob, int num_steps, 
                              int fit_start, int fit_end) {
    if (fit_end >= num_steps) fit_end = num_steps - 1;
    if (fit_start < 1) fit_start = 1;
    
    double sum_log_t = 0.0;
    double sum_log_P = 0.0;
    double sum_log_t_sq = 0.0;
    double sum_log_t_log_P = 0.0;
    int n = 0;
    
    for (int t = fit_start; t <= fit_end; t++) {
        if (return_prob[t] > 1e-10) {
            double log_t = log((double)t);
            double log_P = log(return_prob[t]);
            
            sum_log_t += log_t;
            sum_log_P += log_P;
            sum_log_t_sq += log_t * log_t;
            sum_log_t_log_P += log_t * log_P;
            n++;
        }
    }
    
    if (n < 2) return 0.0;
    
    double slope = (n * sum_log_t_log_P - sum_log_t * sum_log_P) / 
                   (n * sum_log_t_sq - sum_log_t * sum_log_t);
    
    return -2.0 * slope;
}

void compute_spectral_dimension(const Graph *graph, int num_walkers, int max_steps,
                                SpectralDimensionResult *result, unsigned int seed) {
    compute_return_probability(graph, num_walkers, max_steps, 
                              result->return_probability, seed);
    result->num_steps = max_steps;
    
    int fit_start = max_steps / 10;
    int fit_end = max_steps / 2;
    
    result->spectral_dimension = fit_spectral_dimension(result->return_probability, 
                                                        max_steps, fit_start, fit_end);
    
    double sum_sq_err = 0.0;
    int count = 0;
    for (int t = fit_start; t <= fit_end && t < max_steps; t++) {
        if (result->return_probability[t] > 1e-10) {
            double predicted = pow((double)t, -result->spectral_dimension / 2.0);
            double actual = result->return_probability[t];
            sum_sq_err += (predicted - actual) * (predicted - actual);
            count++;
        }
    }
    result->fit_error = count > 0 ? sqrt(sum_sq_err / count) : 0.0;
}

void compute_graph_laplacian(const Graph *graph, double laplacian[MAX_NODES][MAX_NODES]) {
    for (int i = 0; i < graph->num_nodes; i++) {
        for (int j = 0; j < graph->num_nodes; j++) {
            laplacian[i][j] = 0.0;
        }
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        int i = graph->edges[e].i;
        int j = graph->edges[e].j;
        
        laplacian[i][i] += 1.0;
        laplacian[j][j] += 1.0;
        laplacian[i][j] -= 1.0;
        laplacian[j][i] -= 1.0;
    }
}

void compute_laplacian_spectrum(const Graph *graph, LaplacianSpectrum *spectrum) {
    compute_graph_laplacian(graph, spectrum->laplacian);
    
    int n = graph->num_nodes;
    spectrum->num_eigenvalues = n;
    
    double temp[MAX_NODES][MAX_NODES];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp[i][j] = spectrum->laplacian[i][j];
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            spectrum->eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double a_ii = temp[i][i];
                double a_jj = temp[j][j];
                double a_ij = temp[i][j];
                
                if (fabs(a_ij) < 1e-10) continue;
                
                double theta = 0.5 * atan2(2.0 * a_ij, a_jj - a_ii);
                double c = cos(theta);
                double s = sin(theta);
                
                for (int k = 0; k < n; k++) {
                    double temp_ik = temp[i][k];
                    double temp_jk = temp[j][k];
                    temp[i][k] = c * temp_ik - s * temp_jk;
                    temp[j][k] = s * temp_ik + c * temp_jk;
                    
                    double temp_ki = temp[k][i];
                    double temp_kj = temp[k][j];
                    temp[k][i] = c * temp_ki - s * temp_kj;
                    temp[k][j] = s * temp_ki + c * temp_kj;
                    
                    double eig_ik = spectrum->eigenvectors[i][k];
                    double eig_jk = spectrum->eigenvectors[j][k];
                    spectrum->eigenvectors[i][k] = c * eig_ik - s * eig_jk;
                    spectrum->eigenvectors[j][k] = s * eig_ik + c * eig_jk;
                }
            }
        }
    }
    
    for (int i = 0; i < n; i++) {
        spectrum->eigenvalues[i] = temp[i][i];
    }
    
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (spectrum->eigenvalues[i] > spectrum->eigenvalues[j]) {
                double temp_val = spectrum->eigenvalues[i];
                spectrum->eigenvalues[i] = spectrum->eigenvalues[j];
                spectrum->eigenvalues[j] = temp_val;
                
                for (int k = 0; k < n; k++) {
                    double temp_vec = spectrum->eigenvectors[i][k];
                    spectrum->eigenvectors[i][k] = spectrum->eigenvectors[j][k];
                    spectrum->eigenvectors[j][k] = temp_vec;
                }
            }
        }
    }
}

double compute_spectral_dimension_from_spectrum(const LaplacianSpectrum *spectrum, 
                                                double t_min, double t_max) {
    double sum = 0.0;
    int count = 0;
    
    for (int i = 0; i < spectrum->num_eigenvalues; i++) {
        double lambda = spectrum->eigenvalues[i];
        if (lambda > 1e-10) {
            double t = 1.0 / lambda;
            if (t >= t_min && t <= t_max) {
                sum += 2.0;
                count++;
            }
        }
    }
    
    return count > 0 ? sum / count : 0.0;
}

void compute_graph_geometry(const Graph *graph, GraphGeometry *geometry) {
    int n = graph->num_nodes;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            geometry->distance_matrix[i][j] = -1.0;
        }
    }
    
    for (int i = 0; i < n; i++) {
        geometry->distance_matrix[i][i] = 0.0;
        
        for (int e = 0; e < graph->num_edges; e++) {
            if (graph->edges[e].i == i || graph->edges[e].j == i) {
                int j = (graph->edges[e].i == i) ? graph->edges[e].j : graph->edges[e].i;
                geometry->distance_matrix[i][j] = 1.0;
            }
        }
    }
    
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (geometry->distance_matrix[i][k] >= 0 && 
                    geometry->distance_matrix[k][j] >= 0) {
                    double new_dist = geometry->distance_matrix[i][k] + 
                                     geometry->distance_matrix[k][j];
                    if (geometry->distance_matrix[i][j] < 0 || 
                        new_dist < geometry->distance_matrix[i][j]) {
                        geometry->distance_matrix[i][j] = new_dist;
                    }
                }
            }
        }
    }
    
    geometry->graph_diameter = 0;
    geometry->average_distance = 0.0;
    int count = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (geometry->distance_matrix[i][j] > geometry->graph_diameter) {
                geometry->graph_diameter = (int)geometry->distance_matrix[i][j];
            }
            if (geometry->distance_matrix[i][j] >= 0) {
                geometry->average_distance += geometry->distance_matrix[i][j];
                count++;
            }
        }
    }
    
    if (count > 0) {
        geometry->average_distance /= count;
    }
}

double compute_hausdorff_dimension(const Graph *graph) {
    GraphGeometry geometry;
    compute_graph_geometry(graph, &geometry);
    
    int n = graph->num_nodes;
    if (n <= 1) return 0.0;
    
    double r = geometry.graph_diameter;
    if (r < 1.0) r = 1.0;
    
    double d_H = log((double)n) / log(r + 1.0);
    
    return d_H;
}

void print_spectral_result(const SpectralDimensionResult *result) {
    printf("\n=== Spectral Dimension Result ===\n");
    printf("Spectral dimension: %.6f\n", result->spectral_dimension);
    printf("Fit error: %.6e\n", result->fit_error);
    printf("Number of steps: %d\n", result->num_steps);
    
    printf("\nReturn probability (first 20 steps):\n");
    for (int t = 0; t < 20 && t < result->num_steps; t++) {
        printf("  t=%3d: P(t)=%.6f\n", t, result->return_probability[t]);
    }
}

void print_laplacian_spectrum(const LaplacianSpectrum *spectrum, int num_nodes) {
    printf("\n=== Laplacian Spectrum ===\n");
    printf("Eigenvalues:\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("  λ_%d = %.6f\n", i, spectrum->eigenvalues[i]);
    }
    
    printf("\nEigenvectors:\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("  v_%d: ", i);
        for (int j = 0; j < num_nodes; j++) {
            printf("%.4f ", spectrum->eigenvectors[i][j]);
        }
        printf("\n");
    }
}

void print_graph_geometry(const GraphGeometry *geometry, int num_nodes) {
    printf("\n=== Graph Geometry ===\n");
    printf("Graph diameter: %d\n", geometry->graph_diameter);
    printf("Average distance: %.6f\n", geometry->average_distance);
    
    printf("\nDistance matrix:\n");
    printf("     ");
    for (int j = 0; j < num_nodes; j++) {
        printf("%6d ", j);
    }
    printf("\n");
    
    for (int i = 0; i < num_nodes; i++) {
        printf("%3d ", i);
        for (int j = 0; j < num_nodes; j++) {
            printf("%6.1f ", geometry->distance_matrix[i][j]);
        }
        printf("\n");
    }
}

void write_spectral_dimension_data(const SpectralDimensionResult *result, 
                                   const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "# Spectral Dimension Data\n");
    fprintf(fp, "# Spectral dimension: %.6f\n", result->spectral_dimension);
    fprintf(fp, "# t  P(t)\n");
    
    for (int t = 0; t < result->num_steps; t++) {
        fprintf(fp, "%d  %.10f\n", t, result->return_probability[t]);
    }
    
    fclose(fp);
}

void write_laplacian_data(const LaplacianSpectrum *spectrum, int num_nodes,
                          const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "# Laplacian Spectrum\n");
    fprintf(fp, "# i  eigenvalue\n");
    
    for (int i = 0; i < num_nodes; i++) {
        fprintf(fp, "%d  %.10f\n", i, spectrum->eigenvalues[i]);
    }
    
    fclose(fp);
}
