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

typedef struct {
    int depth_counts[11];
    int gen1_nodes;
    int gen2_nodes;
    int gen3_nodes;
    int total_nodes;
    double gen1_fraction;
    double gen2_fraction;
    double gen3_fraction;
    int three_gen_detected;
    double mass_ratio_12;
    double mass_ratio_23;
} ThreeGenerationResult;

void analyze_eigenvalue_clusters(const LaplacianSpectrum *spectrum, 
                                  SpectralClusterResult *result,
                                  double gap_threshold);
void analyze_eigenvalue_clusters_kmeans(const LaplacianSpectrum *spectrum,
                                        SpectralClusterResult *result,
                                        int max_k);
int detect_three_generations(SpectralClusterResult *result);
void compute_mass_ratios(SpectralClusterResult *result);

double compute_ipr(const double *eigenvector, int n);
void analyze_eigenmode_localization(const LaplacianSpectrum *spectrum,
                                     double *ipr_values,
                                     int *localized_modes,
                                     double ipr_threshold);
void analyze_recursive_depth_distribution(const CausalGraphV2 *graph,
                                           int *depth_counts,
                                           int max_depth);
void analyze_three_generation_structure(const CausalGraphV2 *graph,
                                         const LaplacianSpectrum *spectrum,
                                         ThreeGenerationResult *result);

#endif
