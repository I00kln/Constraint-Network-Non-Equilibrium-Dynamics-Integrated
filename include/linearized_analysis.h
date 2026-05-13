#ifndef LINEARIZED_ANALYSIS_H
#define LINEARIZED_ANALYSIS_H

#include "constraint_network.h"
#include "spectral_dimension.h"

#define MAX_MODES 50
#define MAX_DEGREES_FREEDOM 100

typedef struct {
    double frequency;
    double damping;
    double amplitude[MAX_MODES];
    int mode_type;
    int spin;
    int is_massless;
} VibrationMode;

typedef struct {
    int num_modes;
    VibrationMode modes[MAX_MODES];
    int num_massless_modes;
    int num_spin2_modes;
} ModeSpectrum;

typedef struct {
    double hessian[MAX_DEGREES_FREEDOM][MAX_DEGREES_FREEDOM];
    double mass_matrix[MAX_DEGREES_FREEDOM][MAX_DEGREES_FREEDOM];
    int num_dof;
} LinearizedSystem;

typedef struct {
    double polarization[4][4];
    int is_transverse;
    int is_traceless;
    int spin_value;
} PolarizationTensor;

typedef struct {
    double effective_metric[4][4];
    double inverse_metric[4][4];
    double determinant;
} EffectiveGeometry;

typedef struct {
    double dispersion_relation[MAX_MODES];
    double group_velocity[MAX_MODES];
    double phase_velocity[MAX_MODES];
    int num_data_points;
} DispersionData;

void compute_hessian_matrix(const Graph *graph, const PhaseSpace *ps, 
                            double beta, double hessian[MAX_DEGREES_FREEDOM][MAX_DEGREES_FREEDOM]);
void compute_mass_matrix(const Graph *graph, double mass[MAX_DEGREES_FREEDOM][MAX_DEGREES_FREEDOM]);

void linearize_around_background(const Graph *graph, const PhaseSpace *background,
                                 double beta, LinearizedSystem *system);

void compute_normal_modes(const LinearizedSystem *system, ModeSpectrum *spectrum);
void classify_modes(ModeSpectrum *spectrum, const Graph *graph);

int check_massless_mode(const VibrationMode *mode, double threshold);
int check_spin2_polarization(const PolarizationTensor *polar);

void compute_polarization_tensor(const VibrationMode *mode, const Graph *graph,
                                 PolarizationTensor *polar);

void compute_effective_geometry(const Graph *graph, const PhaseSpace *ps,
                                EffectiveGeometry *geometry);

void compute_dispersion_relation(const ModeSpectrum *spectrum, 
                                 DispersionData *dispersion);

double compute_effective_speed_of_light(const ModeSpectrum *spectrum);

void print_mode_spectrum(const ModeSpectrum *spectrum);
void print_linearized_system(const LinearizedSystem *system, int num_dof);
void print_effective_geometry(const EffectiveGeometry *geom);

void write_mode_spectrum(const ModeSpectrum *spectrum, const char *filename);
void write_dispersion_data(const DispersionData *dispersion, const char *filename);

#endif
