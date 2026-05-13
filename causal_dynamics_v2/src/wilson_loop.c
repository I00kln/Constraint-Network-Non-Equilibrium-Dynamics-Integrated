#include "wilson_loop.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int visited_wl[MAX_NODES];
static int current_loop_edges[MAX_LOOP_LENGTH];
static int current_loop_dirs[MAX_LOOP_LENGTH];
static int current_loop_len;

void compute_wilson_loop(const CausalGraphV2 *graph, WilsonLoop *loop) {
    Complex holonomy;
    holonomy.re = 1.0;
    holonomy.im = 0.0;
    
    for (int i = 0; i < loop->length; i++) {
        int edge_idx = loop->edges[i];
        if (edge_idx < 0 || edge_idx >= graph->num_edges) continue;
        
        Complex phase = graph->edges[edge_idx].phase;
        if (loop->edge_dirs[i] < 0) {
            phase = complex_conjugate(phase);
        }
        
        holonomy = complex_multiply(holonomy, phase);
    }
    
    loop->holonomy = holonomy;
    loop->modulus = complex_modulus(holonomy);
    loop->phase = atan2(holonomy.im, holonomy.re);
    if (loop->phase < 0) loop->phase += 2.0 * M_PI;
}

static void dfs_find_loops(const CausalGraphV2 *graph, int start, int current,
                           int depth, int max_length, WilsonLoopCollection *wlc) {
    if (depth > max_length) return;
    if (wlc->num_loops >= MAX_LOOPS) return;
    
    visited_wl[current] = 1;
    
    int out_deg = graph->out_degree[current];
    if (out_deg > MAX_NEIGHBORS) out_deg = MAX_NEIGHBORS;
    
    for (int i = 0; i < out_deg; i++) {
        int next = graph->out_neighbors[current][i];
        if (next < 0 || next >= graph->num_nodes) continue;
        
        int edge_idx = graph->out_edge_indices[current][i];
        
        if (next == start && depth >= 2) {
            if (wlc->num_loops < MAX_LOOPS) {
                WilsonLoop *loop = &wlc->loops[wlc->num_loops];
                loop->length = current_loop_len;
                for (int j = 0; j < current_loop_len; j++) {
                    loop->edges[j] = current_loop_edges[j];
                    loop->edge_dirs[j] = current_loop_dirs[j];
                }
                compute_wilson_loop(graph, loop);
                wlc->num_loops++;
            }
        } else if (!visited_wl[next] && depth < max_length) {
            if (current_loop_len < MAX_LOOP_LENGTH) {
                current_loop_edges[current_loop_len] = edge_idx;
                current_loop_dirs[current_loop_len] = 1;
                current_loop_len++;
                
                dfs_find_loops(graph, start, next, depth + 1, max_length, wlc);
                
                current_loop_len--;
            }
        }
    }
    
    visited_wl[current] = 0;
}

void find_all_wilson_loops(const CausalGraphV2 *graph, WilsonLoopCollection *wlc,
                           int min_length, int max_length) {
    memset(wlc, 0, sizeof(WilsonLoopCollection));
    
    for (int v = 0; v < graph->num_nodes; v++) {
        visited_wl[v] = 0;
    }
    current_loop_len = 0;
    
    for (int v = 0; v < graph->num_nodes; v++) {
        dfs_find_loops(graph, v, v, 1, max_length, wlc);
    }
}

void analyze_wilson_distribution(WilsonLoopCollection *wlc) {
    if (wlc->num_loops == 0) {
        wlc->avg_modulus = 0.0;
        wlc->std_modulus = 0.0;
        wlc->avg_phase = 0.0;
        return;
    }
    
    double sum_mod = 0.0;
    double sum_phase = 0.0;
    
    for (int i = 0; i < wlc->num_loops; i++) {
        sum_mod += wlc->loops[i].modulus;
        sum_phase += wlc->loops[i].phase;
    }
    
    wlc->avg_modulus = sum_mod / wlc->num_loops;
    wlc->avg_phase = sum_phase / wlc->num_loops;
    
    double var_mod = 0.0;
    for (int i = 0; i < wlc->num_loops; i++) {
        double diff = wlc->loops[i].modulus - wlc->avg_modulus;
        var_mod += diff * diff;
    }
    wlc->std_modulus = sqrt(var_mod / wlc->num_loops);
}

int detect_gauge_peaks(WilsonLoopCollection *wlc, double threshold) {
    if (wlc->num_loops < 10) return 0;
    
    static int histogram[100];
    memset(histogram, 0, sizeof(histogram));
    
    for (int i = 0; i < wlc->num_loops; i++) {
        int bin = (int)(wlc->loops[i].phase / (2.0 * M_PI) * 100);
        if (bin >= 100) bin = 99;
        if (bin < 0) bin = 0;
        histogram[bin]++;
    }
    
    int num_peaks = 0;
    int max_count = 0;
    for (int i = 0; i < 100; i++) {
        if (histogram[i] > max_count) max_count = histogram[i];
    }
    
    for (int i = 1; i < 99 && num_peaks < 10; i++) {
        if (histogram[i] > histogram[i-1] && histogram[i] > histogram[i+1]) {
            if (histogram[i] > threshold * max_count) {
                wlc->peak_positions[num_peaks] = (double)i / 100.0 * 2.0 * M_PI;
                wlc->peak_heights[num_peaks] = (double)histogram[i] / wlc->num_loops;
                num_peaks++;
            }
        }
    }
    
    wlc->num_peaks = num_peaks;
    return num_peaks;
}

double wasserstein_distance_1d(const double *dist1, const double *dist2, int n) {
    double distance = 0.0;
    double cdf1 = 0.0;
    double cdf2 = 0.0;
    
    for (int i = 0; i < n; i++) {
        cdf1 += dist1[i];
        cdf2 += dist2[i];
        distance += fabs(cdf1 - cdf2);
    }
    
    return distance / n;
}

void compute_ollivier_ricci(const CausalGraphV2 *graph, RicciCurvature *ricci) {
    memset(ricci, 0, sizeof(RicciCurvature));
    
    for (int v = 0; v < graph->num_nodes; v++) {
        int deg_v = graph->in_degree[v] + graph->out_degree[v];
        if (deg_v == 0) {
            ricci->curvature[v] = 0.0;
            continue;
        }
        
        double ricci_sum = 0.0;
        int neighbor_count = 0;
        
        for (int i = 0; i < graph->in_degree[v]; i++) {
            int u = graph->in_neighbors[v][i];
            int deg_u = graph->in_degree[u] + graph->out_degree[u];
            
            int common = 0;
            for (int j = 0; j < graph->in_degree[v]; j++) {
                int w = graph->in_neighbors[v][j];
                for (int k = 0; k < graph->in_degree[u]; k++) {
                    if (graph->in_neighbors[u][k] == w) common++;
                }
                for (int k = 0; k < graph->out_degree[u]; k++) {
                    if (graph->out_neighbors[u][k] == w) common++;
                }
            }
            for (int j = 0; j < graph->out_degree[v]; j++) {
                int w = graph->out_neighbors[v][j];
                for (int k = 0; k < graph->in_degree[u]; k++) {
                    if (graph->in_neighbors[u][k] == w) common++;
                }
                for (int k = 0; k < graph->out_degree[u]; k++) {
                    if (graph->out_neighbors[u][k] == w) common++;
                }
            }
            
            double kappa = 1.0 - 0.5 * (1.0/deg_v + 1.0/deg_u) + (double)common / (deg_v * deg_u);
            ricci_sum += kappa;
            neighbor_count++;
        }
        
        for (int i = 0; i < graph->out_degree[v]; i++) {
            int u = graph->out_neighbors[v][i];
            int deg_u = graph->in_degree[u] + graph->out_degree[u];
            
            int common = 0;
            for (int j = 0; j < graph->in_degree[v]; j++) {
                int w = graph->in_neighbors[v][j];
                for (int k = 0; k < graph->in_degree[u]; k++) {
                    if (graph->in_neighbors[u][k] == w) common++;
                }
                for (int k = 0; k < graph->out_degree[u]; k++) {
                    if (graph->out_neighbors[u][k] == w) common++;
                }
            }
            for (int j = 0; j < graph->out_degree[v]; j++) {
                int w = graph->out_neighbors[v][j];
                for (int k = 0; k < graph->in_degree[u]; k++) {
                    if (graph->in_neighbors[u][k] == w) common++;
                }
                for (int k = 0; k < graph->out_degree[u]; k++) {
                    if (graph->out_neighbors[u][k] == w) common++;
                }
            }
            
            double kappa = 1.0 - 0.5 * (1.0/deg_v + 1.0/deg_u) + (double)common / (deg_v * deg_u);
            ricci_sum += kappa;
            neighbor_count++;
        }
        
        ricci->curvature[v] = neighbor_count > 0 ? ricci_sum / neighbor_count : 0.0;
    }
    
    double sum = 0.0;
    double min_val = ricci->curvature[0];
    double max_val = ricci->curvature[0];
    
    for (int v = 0; v < graph->num_nodes; v++) {
        sum += ricci->curvature[v];
        if (ricci->curvature[v] < min_val) min_val = ricci->curvature[v];
        if (ricci->curvature[v] > max_val) max_val = ricci->curvature[v];
    }
    
    ricci->avg_curvature = sum / graph->num_nodes;
    ricci->min_curvature = min_val;
    ricci->max_curvature = max_val;
    
    double var = 0.0;
    for (int v = 0; v < graph->num_nodes; v++) {
        double diff = ricci->curvature[v] - ricci->avg_curvature;
        var += diff * diff;
    }
    ricci->std_curvature = sqrt(var / graph->num_nodes);
}

void compute_forman_ricci(const CausalGraphV2 *graph, RicciCurvature *ricci) {
    memset(ricci, 0, sizeof(RicciCurvature));
    
    for (int v = 0; v < graph->num_nodes; v++) {
        int deg_in = graph->in_degree[v];
        int deg_out = graph->out_degree[v];
        
        ricci->curvature[v] = (deg_in + deg_out) - 2.0 * (deg_in + deg_out);
    }
    
    double sum = 0.0;
    ricci->min_curvature = ricci->curvature[0];
    ricci->max_curvature = ricci->curvature[0];
    
    for (int v = 0; v < graph->num_nodes; v++) {
        sum += ricci->curvature[v];
        if (ricci->curvature[v] < ricci->min_curvature) ricci->min_curvature = ricci->curvature[v];
        if (ricci->curvature[v] > ricci->max_curvature) ricci->max_curvature = ricci->curvature[v];
    }
    
    ricci->avg_curvature = sum / graph->num_nodes;
    
    double var = 0.0;
    for (int v = 0; v < graph->num_nodes; v++) {
        double diff = ricci->curvature[v] - ricci->avg_curvature;
        var += diff * diff;
    }
    ricci->std_curvature = sqrt(var / graph->num_nodes);
}
