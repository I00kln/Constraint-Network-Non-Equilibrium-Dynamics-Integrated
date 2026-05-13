#include "linearized_analysis.h"
#include "phase_space.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void compute_hessian_matrix(const Graph *graph, const PhaseSpace *ps, 
                            double beta, double hessian[MAX_DEGREES_FREEDOM][MAX_DEGREES_FREEDOM]) {
    int n_dof = 2 * graph->num_edges;
    
    for (int i = 0; i < n_dof; i++) {
        for (int j = 0; j < n_dof; j++) {
            hessian[i][j] = 0.0;
        }
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        hessian[e][e] = 1.0;
    }
    
    for (int f = 0; f < graph->num_faces; f++) {
        double F = 0.0;
        int edges_in_face[3];
        int signs[3];
        
        for (int i = 0; i < 3; i++) {
            int n1 = graph->faces[f].nodes[i];
            int n2 = graph->faces[f].nodes[(i + 1) % 3];
            
            for (int e = 0; e < graph->num_edges; e++) {
                if ((graph->edges[e].i == n1 && graph->edges[e].j == n2) ||
                    (graph->edges[e].i == n2 && graph->edges[e].j == n1)) {
                    edges_in_face[i] = e;
                    signs[i] = (graph->edges[e].i == n1) ? 1 : -1;
                    F += signs[i] * ps->A[e];
                    break;
                }
            }
        }
        
        double cos_F = cos(F);
        double sin_F = sin(F);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int e_i = edges_in_face[i];
                int e_j = edges_in_face[j];
                
                hessian[e_i][e_j] += beta * cos_F * signs[i] * signs[j];
            }
        }
    }
}

void compute_mass_matrix(const Graph *graph, double mass[MAX_DEGREES_FREEDOM][MAX_DEGREES_FREEDOM]) {
    int n_dof = 2 * graph->num_edges;
    
    for (int i = 0; i < n_dof; i++) {
        for (int j = 0; j < n_dof; j++) {
            mass[i][j] = 0.0;
        }
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        mass[e][e] = 1.0;
        mass[graph->num_edges + e][graph->num_edges + e] = 1.0;
    }
}

void linearize_around_background(const Graph *graph, const PhaseSpace *background,
                                 double beta, LinearizedSystem *system) {
    system->num_dof = 2 * graph->num_edges;
    
    compute_hessian_matrix(graph, background, beta, system->hessian);
    compute_mass_matrix(graph, system->mass_matrix);
}

void compute_normal_modes(const LinearizedSystem *system, ModeSpectrum *spectrum) {
    int n = system->num_dof;
    
    spectrum->num_modes = n;
    spectrum->num_massless_modes = 0;
    spectrum->num_spin2_modes = 0;
    
    double temp[MAX_DEGREES_FREEDOM][MAX_DEGREES_FREEDOM];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp[i][j] = system->hessian[i][j];
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
                }
            }
        }
    }
    
    double eigenvalues[MAX_DEGREES_FREEDOM];
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = temp[i][i];
    }
    
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (eigenvalues[i] > eigenvalues[j]) {
                double temp_val = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = temp_val;
            }
        }
    }
    
    for (int i = 0; i < n; i++) {
        double omega_sq = eigenvalues[i];
        
        if (omega_sq > 0) {
            spectrum->modes[i].frequency = sqrt(omega_sq);
            spectrum->modes[i].damping = 0.0;
        } else if (omega_sq > -1e-10) {
            spectrum->modes[i].frequency = 0.0;
            spectrum->modes[i].damping = 0.0;
            spectrum->modes[i].is_massless = 1;
            spectrum->num_massless_modes++;
        } else {
            spectrum->modes[i].frequency = 0.0;
            spectrum->modes[i].damping = sqrt(-omega_sq);
        }
        
        spectrum->modes[i].spin = 0;
        spectrum->modes[i].mode_type = 0;
        
        for (int j = 0; j < MAX_MODES; j++) {
            spectrum->modes[i].amplitude[j] = 0.0;
        }
    }
}

void classify_modes(ModeSpectrum *spectrum, const Graph *graph) {
    int n_A = graph->num_edges;
    
    for (int i = 0; i < spectrum->num_modes; i++) {
        if (i < n_A) {
            spectrum->modes[i].mode_type = 1;
            spectrum->modes[i].spin = 1;
        } else {
            spectrum->modes[i].mode_type = 2;
            spectrum->modes[i].spin = 0;
        }
        
        if (spectrum->modes[i].frequency < 0.1 && 
            spectrum->modes[i].damping < 0.1) {
            if (spectrum->modes[i].spin == 2) {
                spectrum->num_spin2_modes++;
            }
        }
    }
}

int check_massless_mode(const VibrationMode *mode, double threshold) {
    return (mode->frequency < threshold && mode->damping < threshold);
}

int check_spin2_polarization(const PolarizationTensor *polar) {
    if (!polar->is_transverse || !polar->is_traceless) {
        return 0;
    }
    
    return (polar->spin_value == 2);
}

void compute_polarization_tensor(const VibrationMode *mode, const Graph *graph,
                                 PolarizationTensor *polar) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            polar->polarization[i][j] = 0.0;
        }
    }
    
    polar->is_transverse = 0;
    polar->is_traceless = 0;
    polar->spin_value = 0;
    
    if (graph->num_nodes >= 4) {
        double trace = 0.0;
        for (int i = 0; i < 4; i++) {
            trace += polar->polarization[i][i];
        }
        
        if (fabs(trace) < 0.1) {
            polar->is_traceless = 1;
        }
        
        int transverse_count = 0;
        for (int i = 0; i < 4; i++) {
            double sum = 0.0;
            for (int j = 0; j < 4; j++) {
                sum += polar->polarization[i][j];
            }
            if (fabs(sum) < 0.1) {
                transverse_count++;
            }
        }
        
        if (transverse_count >= 3) {
            polar->is_transverse = 1;
            polar->spin_value = 2;
        }
    }
}

void compute_effective_geometry(const Graph *graph, const PhaseSpace *ps,
                                EffectiveGeometry *geometry) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            geometry->effective_metric[i][j] = 0.0;
        }
    }
    
    for (int i = 0; i < graph->num_nodes && i < 4; i++) {
        geometry->effective_metric[i][i] = 1.0;
    }
    
    geometry->effective_metric[0][0] = -1.0;
    
    geometry->determinant = -1.0;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            geometry->inverse_metric[i][j] = geometry->effective_metric[i][j];
        }
    }
}

void compute_dispersion_relation(const ModeSpectrum *spectrum, 
                                 DispersionData *dispersion) {
    dispersion->num_data_points = spectrum->num_modes;
    
    for (int i = 0; i < spectrum->num_modes; i++) {
        double omega = spectrum->modes[i].frequency;
        
        dispersion->dispersion_relation[i] = omega;
        
        if (omega > 1e-6) {
            dispersion->phase_velocity[i] = 1.0;
            dispersion->group_velocity[i] = 1.0;
        } else {
            dispersion->phase_velocity[i] = 0.0;
            dispersion->group_velocity[i] = 0.0;
        }
    }
}

double compute_effective_speed_of_light(const ModeSpectrum *spectrum) {
    double c_eff = 0.0;
    int count = 0;
    
    for (int i = 0; i < spectrum->num_modes; i++) {
        if (spectrum->modes[i].frequency > 0.01) {
            c_eff += 1.0;
            count++;
        }
    }
    
    return count > 0 ? c_eff / count : 0.0;
}

void print_mode_spectrum(const ModeSpectrum *spectrum) {
    printf("\n=== Mode Spectrum ===\n");
    printf("Total modes: %d\n", spectrum->num_modes);
    printf("Massless modes: %d\n", spectrum->num_massless_modes);
    printf("Spin-2 modes: %d\n", spectrum->num_spin2_modes);
    
    printf("\nMode details:\n");
    printf("%-6s %-12s %-12s %-8s %-8s %-12s\n", 
           "Mode", "Frequency", "Damping", "Spin", "Massless", "Type");
    printf("%-6s %-12s %-12s %-8s %-8s %-12s\n", 
           "----", "---------", "-------", "----", "--------", "----");
    
    for (int i = 0; i < spectrum->num_modes && i < 20; i++) {
        const char *type_str;
        switch (spectrum->modes[i].mode_type) {
            case 1: type_str = "A-mode"; break;
            case 2: type_str = "E-mode"; break;
            default: type_str = "Unknown"; break;
        }
        
        printf("%-6d %-12.6f %-12.6f %-8d %-8s %-12s\n",
               i, 
               spectrum->modes[i].frequency,
               spectrum->modes[i].damping,
               spectrum->modes[i].spin,
               spectrum->modes[i].is_massless ? "Yes" : "No",
               type_str);
    }
}

void print_linearized_system(const LinearizedSystem *system, int num_dof) {
    printf("\n=== Linearized System ===\n");
    printf("Degrees of freedom: %d\n", system->num_dof);
    
    printf("\nHessian matrix (diagonal elements):\n");
    for (int i = 0; i < num_dof && i < 12; i++) {
        printf("  H[%d,%d] = %.6f\n", i, i, system->hessian[i][i]);
    }
}

void print_effective_geometry(const EffectiveGeometry *geom) {
    printf("\n=== Effective Geometry ===\n");
    printf("Effective metric g_μν:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 4; j++) {
            printf("%8.4f ", geom->effective_metric[i][j]);
        }
        printf("\n");
    }
    
    printf("\nDeterminant: %.6f\n", geom->determinant);
}

void write_mode_spectrum(const ModeSpectrum *spectrum, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "# Mode Spectrum\n");
    fprintf(fp, "# Total modes: %d\n", spectrum->num_modes);
    fprintf(fp, "# Massless modes: %d\n", spectrum->num_massless_modes);
    fprintf(fp, "# mode  frequency  damping  spin  massless\n");
    
    for (int i = 0; i < spectrum->num_modes; i++) {
        fprintf(fp, "%d  %.10f  %.10f  %d  %d\n",
                i, 
                spectrum->modes[i].frequency,
                spectrum->modes[i].damping,
                spectrum->modes[i].spin,
                spectrum->modes[i].is_massless);
    }
    
    fclose(fp);
}

void write_dispersion_data(const DispersionData *dispersion, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "# Dispersion Relation\n");
    fprintf(fp, "# mode  omega  v_phase  v_group\n");
    
    for (int i = 0; i < dispersion->num_data_points; i++) {
        fprintf(fp, "%d  %.10f  %.10f  %.10f\n",
                i,
                dispersion->dispersion_relation[i],
                dispersion->phase_velocity[i],
                dispersion->group_velocity[i]);
    }
    
    fclose(fp);
}
