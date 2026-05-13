#ifndef GEOMETRY_EMERGENCE_H
#define GEOMETRY_EMERGENCE_H

#include "causal_graph_v2.h"
#include "null_models.h"
#include "wilson_loop.h"

#define MAX_TIME_STEPS 500
#define MAX_DISTANCE 30

typedef struct {
    double correlation[MAX_DISTANCE][MAX_TIME_STEPS];
    int max_distance;
    int max_time;
    double light_cone_velocity;
    int light_cone_detected;
    double anisotropy;
} ResponseFunction;

typedef struct {
    double slope;
    double intercept;
    double r_squared;
    int significant;
} EinsteinFit;

void compute_response_function(ResponseFunction *resp, const double *rho_history[MAX_TIME_STEPS],
                                int num_nodes, int num_steps, int origin_node);
double fit_light_cone_velocity(ResponseFunction *resp);
double compute_anisotropy(ResponseFunction *resp);

typedef struct {
    double energy_density[MAX_NODES];
    double curvature[MAX_NODES];
    EinsteinFit fit_result;
    double effective_kappa;
} EinsteinRelation;

void compute_einstein_relation(const RicciCurvature *ricci, 
                               const InformationStateV2 *state,
                               EinsteinRelation *relation);
void fit_curvature_density_relation(EinsteinRelation *relation, int num_nodes);

typedef struct {
    int num_clusters;
    double cluster_centers[MAX_EIGENVECTORS];
    double cluster_widths[MAX_EIGENVECTORS];
    int cluster_counts[MAX_EIGENVECTORS];
    int three_generation_detected;
    double mass_ratios[3];
} SpectralClusterResult;

void analyze_eigenvalue_clusters(const LaplacianSpectrum *spectrum, 
                                  SpectralClusterResult *result,
                                  double gap_threshold);
int detect_three_generations(SpectralClusterResult *result);
void compute_mass_ratios(SpectralClusterResult *result);

#endif
