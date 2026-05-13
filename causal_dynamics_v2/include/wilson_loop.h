#ifndef WILSON_LOOP_H
#define WILSON_LOOP_H

#include "causal_graph_v2.h"

#define MAX_LOOPS 1000
#define MAX_LOOP_LENGTH 20

typedef struct {
    int edges[MAX_LOOP_LENGTH];
    int edge_dirs[MAX_LOOP_LENGTH];
    int length;
    Complex holonomy;
    double modulus;
    double phase;
} WilsonLoop;

typedef struct {
    WilsonLoop loops[MAX_LOOPS];
    int num_loops;
    double avg_modulus;
    double std_modulus;
    double avg_phase;
    int num_peaks;
    double peak_positions[10];
    double peak_heights[10];
} WilsonLoopCollection;

void compute_wilson_loop(const CausalGraphV2 *graph, WilsonLoop *loop);
void find_all_wilson_loops(const CausalGraphV2 *graph, WilsonLoopCollection *wlc, 
                           int min_length, int max_length);
void analyze_wilson_distribution(WilsonLoopCollection *wlc);
int detect_gauge_peaks(WilsonLoopCollection *wlc, double threshold);

typedef struct {
    double curvature[MAX_NODES];
    double avg_curvature;
    double std_curvature;
    double min_curvature;
    double max_curvature;
} RicciCurvature;

void compute_ollivier_ricci(const CausalGraphV2 *graph, RicciCurvature *ricci);
void compute_forman_ricci(const CausalGraphV2 *graph, RicciCurvature *ricci);
double wasserstein_distance_1d(const double *dist1, const double *dist2, int n);

#endif
