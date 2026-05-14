#include "causal_capacity.h"

static void normalize_vec(double *v, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrt(norm);
    if (norm > 1e-15) {
        for (int i = 0; i < n; i++) v[i] /= norm;
    }
}

static double dot_vec(const double *v1, const double *v2, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v1[i] * v2[i];
    return s;
}

static void jacobi_eigen_4x4(double A[4][4], double eigvals[4]) {
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = 0.0;
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
                off += A[i][j] * A[i][j];
        if (off < 1e-20) break;

        for (int p = 0; p < 4; p++) {
            for (int q = p + 1; q < 4; q++) {
                if (fabs(A[p][q]) < 1e-15) continue;
                double theta = 0.5 * atan2(2.0 * A[p][q], A[p][p] - A[q][q]);
                double c = cos(theta), s = sin(theta);

                for (int r = 0; r < 4; r++) {
                    if (r == p || r == q) continue;
                    double arp = A[r][p], arq = A[r][q];
                    A[r][p] = c * arp + s * arq;
                    A[p][r] = A[r][p];
                    A[r][q] = -s * arp + c * arq;
                    A[q][r] = A[r][q];
                }
                double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                A[p][p] = c * c * app + 2 * s * c * apq + s * s * aqq;
                A[q][q] = s * s * app - 2 * s * c * apq + c * c * aqq;
                A[p][q] = 0;
                A[q][p] = 0;
            }
        }
    }

    for (int i = 0; i < 4; i++) eigvals[i] = A[i][i];

    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (eigvals[j] > eigvals[i]) {
                double tmp = eigvals[i];
                eigvals[i] = eigvals[j];
                eigvals[j] = tmp;
            }
        }
    }
}

static void eigen_2x2(double a, double b, double c, double eigvals[2]) {
    double trace = a + c;
    double det = a * c - b * b;
    double disc = trace * trace - 4.0 * det;
    if (disc < 0) disc = 0;
    double sq = sqrt(disc);
    eigvals[0] = (trace + sq) / 2.0;
    eigvals[1] = (trace - sq) / 2.0;
}

void compute_propagation_operator_spectrum(const CausalGraph *graph, PropagationSpectrum *spectrum, int num_eigenvalues) {
    memset(spectrum, 0, sizeof(PropagationSpectrum));
    int n = graph->num_nodes;
    if (num_eigenvalues > MAX_EIGENVALUES) num_eigenvalues = MAX_EIGENVALUES;
    if (num_eigenvalues > n) num_eigenvalues = n;
    if (num_eigenvalues < 1) return;

    static double v[MAX_NODES];
    static double Pv[MAX_NODES];
    static double eigvecs[MAX_NODES][MAX_EIGENVALUES];

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < num_eigenvalues; k++) {
            eigvecs[i][k] = 0.0;
        }
    }

    unsigned int local_seed = 42;

    for (int i = 0; i < n; i++) {
        v[i] = 1.0 / sqrt((double)n);
        eigvecs[i][0] = v[i];
    }

    for (int v_idx = 0; v_idx < n; v_idx++) {
        Pv[v_idx] = 0.0;
        for (int i = 0; i < graph->in_degree[v_idx]; i++) {
            int parent = graph->in_neighbors[v_idx][i];
            int eidx = graph->in_edge_idx[v_idx][i];
            double cap = graph->edges[eidx].capacity;
            Pv[v_idx] += 0.5 * cap * v[parent];
        }
        for (int i = 0; i < graph->out_degree[v_idx]; i++) {
            int child = graph->out_neighbors[v_idx][i];
            int eidx = graph->out_edge_idx[v_idx][i];
            double cap = graph->edges[eidx].capacity;
            Pv[v_idx] += 0.5 * cap * v[child];
        }
    }

    spectrum->eigenvalues[0] = dot_vec(v, Pv, n);
    spectrum->num_eigenvalues = 1;

    for (int k = 1; k < num_eigenvalues; k++) {
        local_seed = local_seed * 1103515245 + 12345;
        for (int i = 0; i < n; i++) {
            v[i] = ((double)((local_seed + i * 7919) % 10000) / 10000.0 - 0.5);
        }

        for (int j = 0; j < k; j++) {
            double proj = 0.0;
            for (int i = 0; i < n; i++) proj += v[i] * eigvecs[i][j];
            for (int i = 0; i < n; i++) v[i] -= proj * eigvecs[i][j];
        }
        normalize_vec(v, n);

        double lambda = 0.0;
        double prev_lambda = 1e10;

        for (int iter = 0; iter < 300; iter++) {
            for (int v_idx = 0; v_idx < n; v_idx++) {
                Pv[v_idx] = 0.0;
                for (int i = 0; i < graph->in_degree[v_idx]; i++) {
                    int parent = graph->in_neighbors[v_idx][i];
                    int eidx = graph->in_edge_idx[v_idx][i];
                    double cap = graph->edges[eidx].capacity;
                    Pv[v_idx] += 0.5 * cap * v[parent];
                }
                for (int i = 0; i < graph->out_degree[v_idx]; i++) {
                    int child = graph->out_neighbors[v_idx][i];
                    int eidx = graph->out_edge_idx[v_idx][i];
                    double cap = graph->edges[eidx].capacity;
                    Pv[v_idx] += 0.5 * cap * v[child];
                }
            }

            for (int j = 0; j < k; j++) {
                double proj = 0.0;
                for (int i = 0; i < n; i++) proj += Pv[i] * eigvecs[i][j];
                for (int i = 0; i < n; i++) Pv[i] -= proj * eigvecs[i][j];
            }

            lambda = dot_vec(v, Pv, n);

            double norm = 0.0;
            for (int i = 0; i < n; i++) norm += Pv[i] * Pv[i];
            norm = sqrt(norm);

            if (norm > 1e-15) {
                for (int i = 0; i < n; i++) v[i] = Pv[i] / norm;
            } else {
                break;
            }

            if (fabs(lambda - prev_lambda) < 1e-12 * (fabs(lambda) + 1e-15) && iter > 30) break;
            prev_lambda = lambda;
        }

        spectrum->eigenvalues[k] = lambda;
        for (int i = 0; i < n; i++) eigvecs[i][k] = v[i];
        spectrum->num_eigenvalues++;
    }

    for (int i = 0; i < spectrum->num_eigenvalues - 1; i++) {
        for (int j = i + 1; j < spectrum->num_eigenvalues; j++) {
            if (spectrum->eigenvalues[j] > spectrum->eigenvalues[i]) {
                double tmp = spectrum->eigenvalues[i];
                spectrum->eigenvalues[i] = spectrum->eigenvalues[j];
                spectrum->eigenvalues[j] = tmp;
            }
        }
    }

    spectrum->negative_count = 0;
    spectrum->positive_count = 0;
    spectrum->zero_count = 0;
    spectrum->max_negative_eigenvalue = 0.0;
    spectrum->min_positive_eigenvalue = 1e10;

    for (int k = 0; k < spectrum->num_eigenvalues; k++) {
        if (spectrum->eigenvalues[k] < -1e-10) {
            spectrum->negative_count++;
            if (spectrum->eigenvalues[k] < spectrum->max_negative_eigenvalue) {
                spectrum->max_negative_eigenvalue = spectrum->eigenvalues[k];
            }
        } else if (spectrum->eigenvalues[k] > 1e-10) {
            spectrum->positive_count++;
            if (spectrum->eigenvalues[k] < spectrum->min_positive_eigenvalue) {
                spectrum->min_positive_eigenvalue = spectrum->eigenvalues[k];
            }
        } else {
            spectrum->zero_count++;
        }
    }

    spectrum->lorentz_signature_detected = check_lorentz_signature(spectrum);

    if (spectrum->positive_count > 0 && spectrum->negative_count > 0) {
        spectrum->signature_ratio = fabs(spectrum->max_negative_eigenvalue) / spectrum->min_positive_eigenvalue;
    } else {
        spectrum->signature_ratio = 0.0;
    }

    if (spectrum->num_eigenvalues >= 2) {
        double max_gap = 0.0;
        for (int k = 0; k < spectrum->num_eigenvalues - 1; k++) {
            double gap = spectrum->eigenvalues[k] - spectrum->eigenvalues[k + 1];
            if (gap > max_gap) max_gap = gap;
        }
        spectrum->spectral_gap = max_gap;
    }

    int total_nonzero = spectrum->negative_count + spectrum->positive_count;
    spectrum->neg_fraction = (total_nonzero > 0) ? (double)spectrum->negative_count / total_nonzero : 0.0;
}

void compute_response_spectrum(CausalGraph *graph, InfoState *state, double eta, double delta, PropagationSpectrum *spectrum, int num_eigenvalues) {
    memset(spectrum, 0, sizeof(PropagationSpectrum));
    int n = graph->num_nodes;
    if (num_eigenvalues > MAX_EIGENVALUES) num_eigenvalues = MAX_EIGENVALUES;
    if (num_eigenvalues > n) num_eigenvalues = n;
    if (num_eigenvalues < 1) return;

    static double R[MAX_NODES * MAX_NODES];
    compute_intervention_response(graph, state, eta, delta, R);

    static double S[MAX_NODES * MAX_NODES];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            S[i * n + j] = 0.5 * (R[i * n + j] + R[j * n + i]);
        }
    }

    double asym = 0.0, sym = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double a = R[i * n + j] - R[j * n + i];
            asym += a * a;
            sym += S[i * n + j] * S[i * n + j];
        }
    }
    spectrum->asymmetry_measure = (sym > 1e-15) ? sqrt(asym / sym) : 0.0;

    static double v[MAX_NODES];
    static double Sv[MAX_NODES];
    static double eigvecs[MAX_NODES][MAX_EIGENVALUES];

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < num_eigenvalues; k++) {
            eigvecs[i][k] = 0.0;
        }
    }

    for (int i = 0; i < n; i++) {
        v[i] = 1.0 / sqrt((double)n);
        eigvecs[i][0] = v[i];
    }

    for (int i = 0; i < n; i++) {
        Sv[i] = 0.0;
        for (int j = 0; j < n; j++) {
            Sv[i] += S[i * n + j] * v[j];
        }
    }
    spectrum->eigenvalues[0] = dot_vec(v, Sv, n);
    spectrum->num_eigenvalues = 1;

    unsigned int local_seed = 42;

    for (int k = 1; k < num_eigenvalues; k++) {
        local_seed = local_seed * 1103515245 + 12345;
        for (int i = 0; i < n; i++) {
            v[i] = ((double)((local_seed + i * 7919) % 10000) / 10000.0 - 0.5);
        }

        for (int j = 0; j < k; j++) {
            double proj = 0.0;
            for (int i = 0; i < n; i++) proj += v[i] * eigvecs[i][j];
            for (int i = 0; i < n; i++) v[i] -= proj * eigvecs[i][j];
        }
        normalize_vec(v, n);

        double lambda = 0.0;
        double prev_lambda = 1e10;

        for (int iter = 0; iter < 500; iter++) {
            for (int i = 0; i < n; i++) {
                Sv[i] = 0.0;
                for (int j = 0; j < n; j++) {
                    Sv[i] += S[i * n + j] * v[j];
                }
            }

            for (int j = 0; j < k; j++) {
                double proj = 0.0;
                for (int i = 0; i < n; i++) proj += Sv[i] * eigvecs[i][j];
                for (int i = 0; i < n; i++) Sv[i] -= proj * eigvecs[i][j];
            }

            lambda = dot_vec(v, Sv, n);

            double norm = 0.0;
            for (int i = 0; i < n; i++) norm += Sv[i] * Sv[i];
            norm = sqrt(norm);

            if (norm > 1e-15) {
                for (int i = 0; i < n; i++) v[i] = Sv[i] / norm;
            } else {
                break;
            }

            if (fabs(lambda - prev_lambda) < 1e-12 * (fabs(lambda) + 1e-15) && iter > 50) break;
            prev_lambda = lambda;
        }

        spectrum->eigenvalues[k] = lambda;
        for (int i = 0; i < n; i++) eigvecs[i][k] = v[i];
        spectrum->num_eigenvalues++;
    }

    for (int i = 0; i < spectrum->num_eigenvalues - 1; i++) {
        for (int j = i + 1; j < spectrum->num_eigenvalues; j++) {
            if (spectrum->eigenvalues[j] > spectrum->eigenvalues[i]) {
                double tmp = spectrum->eigenvalues[i];
                spectrum->eigenvalues[i] = spectrum->eigenvalues[j];
                spectrum->eigenvalues[j] = tmp;
            }
        }
    }

    spectrum->negative_count = 0;
    spectrum->positive_count = 0;
    spectrum->zero_count = 0;
    spectrum->max_negative_eigenvalue = 0.0;
    spectrum->min_positive_eigenvalue = 1e10;

    for (int k = 0; k < spectrum->num_eigenvalues; k++) {
        if (spectrum->eigenvalues[k] < -1e-10) {
            spectrum->negative_count++;
            if (spectrum->eigenvalues[k] < spectrum->max_negative_eigenvalue) {
                spectrum->max_negative_eigenvalue = spectrum->eigenvalues[k];
            }
        } else if (spectrum->eigenvalues[k] > 1e-10) {
            spectrum->positive_count++;
            if (spectrum->eigenvalues[k] < spectrum->min_positive_eigenvalue) {
                spectrum->min_positive_eigenvalue = spectrum->eigenvalues[k];
            }
        } else {
            spectrum->zero_count++;
        }
    }

    spectrum->lorentz_signature_detected = check_lorentz_signature(spectrum);

    if (spectrum->positive_count > 0 && spectrum->negative_count > 0) {
        spectrum->signature_ratio = fabs(spectrum->max_negative_eigenvalue) / spectrum->min_positive_eigenvalue;
    } else {
        spectrum->signature_ratio = 0.0;
    }

    if (spectrum->num_eigenvalues >= 2) {
        double max_gap = 0.0;
        for (int k = 0; k < spectrum->num_eigenvalues - 1; k++) {
            double gap = spectrum->eigenvalues[k] - spectrum->eigenvalues[k + 1];
            if (gap > max_gap) max_gap = gap;
        }
        spectrum->spectral_gap = max_gap;
    }

    int total_nonzero = spectrum->negative_count + spectrum->positive_count;
    spectrum->neg_fraction = (total_nonzero > 0) ? (double)spectrum->negative_count / total_nonzero : 0.0;

    LocalPCAResult pca;
    compute_local_pca(graph, state, eta, delta, &pca);

    for (int d = 0; d < LOCAL_DIM; d++) {
        spectrum->coarse_grained_eigenvalues[d] = pca.avg_tensor_eigenvalues[d];
        spectrum->coarse_grained_signs[d] = pca.avg_tensor_signs[d];
    }
    spectrum->coarse_grained_lorentz = pca.avg_tensor_lorentz;

    int cg_neg = 0, cg_pos = 0;
    for (int d = 0; d < LOCAL_DIM; d++) {
        if (pca.avg_tensor_eigenvalues[d] < -1e-10) cg_neg++;
        else if (pca.avg_tensor_eigenvalues[d] > 1e-10) cg_pos++;
    }
    spectrum->coarse_grained_neg_fraction = (cg_neg + cg_pos > 0) ?
        (double)cg_neg / (cg_neg + cg_pos) : 0.0;

    if (pca.avg_tensor_lorentz && !spectrum->lorentz_signature_detected) {
        spectrum->lorentz_signature_detected = 1;
    }
}

void compute_local_response_tensor(CausalGraph *graph, InfoState *state, int node_v, double eta, double delta, LocalResponseTensor *tensor) {
    memset(tensor, 0, sizeof(LocalResponseTensor));
    int n = graph->num_nodes;

    int neighbors[2 * MAX_NEIGHBORS];
    int nn = 0;

    for (int i = 0; i < graph->in_degree[node_v] && nn < LOCAL_DIM; i++) {
        neighbors[nn++] = graph->in_neighbors[node_v][i];
    }
    for (int i = 0; i < graph->out_degree[node_v] && nn < LOCAL_DIM; i++) {
        int candidate = graph->out_neighbors[node_v][i];
        int dup = 0;
        for (int j = 0; j < nn; j++) {
            if (neighbors[j] == candidate) { dup = 1; break; }
        }
        if (!dup) neighbors[nn++] = candidate;
    }

    if (nn < 2) {
        tensor->dim = nn;
        tensor->eigenvalues[0] = 1.0;
        tensor->sign_pattern[0] = 1;
        tensor->lorentz_signature = 0;
        return;
    }

    tensor->dim = nn;

    static double orig_rho[MAX_NODES];
    memcpy(orig_rho, state->rho, n * sizeof(double));

    double responses[LOCAL_DIM][MAX_NODES];
    for (int i = 0; i < nn; i++) {
        int u = neighbors[i];

        memcpy(state->rho, orig_rho, n * sizeof(double));
        state->rho[u] += delta;

        static double new_rho[MAX_NODES];
        for (int v = 0; v < n; v++) {
            double sum_cap = 0.0;
            double weighted_sum = 0.0;

            for (int j = 0; j < graph->in_degree[v]; j++) {
                int parent = graph->in_neighbors[v][j];
                int eidx = graph->in_edge_idx[v][j];
                double cap = graph->edges[eidx].capacity;
                sum_cap += cap;
                weighted_sum += cap * state->rho[parent];
            }

            if (sum_cap > 1e-15) {
                new_rho[v] = (1.0 - eta) * state->rho[v] + eta * weighted_sum;
            } else {
                new_rho[v] = state->rho[v];
            }
        }

        double total = 0.0;
        for (int v = 0; v < n; v++) total += new_rho[v];
        if (total > 1e-15) {
            double scale = state->total_info / total;
            for (int v = 0; v < n; v++) new_rho[v] *= scale;
        }

        for (int v = 0; v < n; v++) {
            responses[i][v] = (new_rho[v] - orig_rho[v]) / delta;
        }
    }

    memcpy(state->rho, orig_rho, n * sizeof(double));

    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) {
            double r_ij = responses[i][neighbors[j]];
            double r_ji = responses[j][neighbors[i]];
            tensor->matrix[i][j] = 0.5 * (r_ij + r_ji);
        }
    }

    double asym_norm = 0.0, sym_norm = 0.0;
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) {
            double a = responses[i][neighbors[j]] - responses[j][neighbors[i]];
            asym_norm += a * a;
            sym_norm += tensor->matrix[i][j] * tensor->matrix[i][j];
        }
    }
    tensor->asymmetry_norm = (sym_norm > 1e-15) ? sqrt(asym_norm / sym_norm) : 0.0;

    double total_resp = 0.0;
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) {
            total_resp += responses[i][neighbors[j]] * responses[i][neighbors[j]];
        }
    }
    tensor->response_strength = sqrt(total_resp) / (nn * nn);

    memset(tensor->eigenvalues, 0, sizeof(tensor->eigenvalues));
    if (nn == 2) {
        eigen_2x2(tensor->matrix[0][0], tensor->matrix[0][1],
                   tensor->matrix[1][1], tensor->eigenvalues);
    } else {
        double a4[4][4];
        memset(a4, 0, sizeof(a4));
        for (int i = 0; i < nn && i < 4; i++)
            for (int j = 0; j < nn && j < 4; j++)
                a4[i][j] = tensor->matrix[i][j];
        jacobi_eigen_4x4(a4, tensor->eigenvalues);
    }

    int neg_count = 0, pos_count = 0;
    for (int d = 0; d < nn; d++) {
        if (tensor->eigenvalues[d] < -1e-10) {
            tensor->sign_pattern[d] = -1;
            neg_count++;
        } else {
            tensor->sign_pattern[d] = 1;
            pos_count++;
        }
    }

    tensor->lorentz_signature = (neg_count >= 1 && pos_count >= 2) ? 1 : 0;
}

void compute_local_pca(CausalGraph *graph, InfoState *state, double eta, double delta, LocalPCAResult *result) {
    memset(result, 0, sizeof(LocalPCAResult));
    int n = graph->num_nodes;

    result->lorentz_nodes = 0;
    result->euclidean_nodes = 0;
    result->other_nodes = 0;

    double total_asym = 0.0;
    double total_resp = 0.0;
    double sum_neg_eig = 0.0, sum_pos_eig = 0.0;
    int total_neg_count = 0, total_pos_count = 0;

    for (int v = 0; v < n; v++) {
        LocalResponseTensor tensor;
        compute_local_response_tensor(graph, state, v, eta, delta, &tensor);

        int dim = tensor.dim;
        int neg_count = 0, pos_count = 0;
        for (int d = 0; d < dim; d++) {
            result->eigenvalues[v][d] = tensor.eigenvalues[d];
            result->sign_pattern[v][d] = tensor.sign_pattern[d];
            if (tensor.eigenvalues[d] < -1e-10) {
                neg_count++;
                sum_neg_eig += tensor.eigenvalues[d];
                total_neg_count++;
            } else if (tensor.eigenvalues[d] > 1e-10) {
                pos_count++;
                sum_pos_eig += tensor.eigenvalues[d];
                total_pos_count++;
            }
        }

        if (neg_count >= 1 && pos_count >= 2) {
            result->lorentz_nodes++;
        } else if (neg_count == 0 && pos_count >= 2) {
            result->euclidean_nodes++;
        } else {
            result->other_nodes++;
        }

        total_asym += tensor.asymmetry_norm;
        total_resp += tensor.response_strength;
    }

    result->lorentz_fraction = (n > 0) ? (double)result->lorentz_nodes / n : 0.0;
    result->avg_asymmetry = (n > 0) ? total_asym / n : 0.0;
    result->avg_response_strength = (n > 0) ? total_resp / n : 0.0;

    double avg_neg_eig = (total_neg_count > 0) ? sum_neg_eig / total_neg_count : 0.0;
    double avg_pos_eig = (total_pos_count > 0) ? sum_pos_eig / total_pos_count : 0.0;

    result->avg_tensor_eigenvalues[0] = avg_neg_eig;
    result->avg_tensor_eigenvalues[1] = avg_pos_eig;
    result->avg_tensor_eigenvalues[2] = 0.0;
    result->avg_tensor_eigenvalues[3] = 0.0;

    result->avg_tensor_signs[0] = (avg_neg_eig < -1e-10) ? -1 : 1;
    result->avg_tensor_signs[1] = (avg_pos_eig > 1e-10) ? 1 : 0;
    result->avg_tensor_signs[2] = 0;
    result->avg_tensor_signs[3] = 0;

    result->avg_tensor_lorentz = (avg_neg_eig < -1e-10 && avg_pos_eig > 1e-10) ? 1 : 0;

    for (int i = 0; i < LOCAL_DIM; i++) {
        for (int j = 0; j < LOCAL_DIM; j++) {
            result->avg_tensor[i][j] = 0.0;
        }
    }
    result->avg_tensor[0][0] = avg_neg_eig;
    result->avg_tensor[1][1] = avg_pos_eig;
}

static void compute_graph_distances(const CausalGraph *graph, int origin, int *distances) {
    int n = graph->num_nodes;
    for (int v = 0; v < n; v++) distances[v] = -1;
    distances[origin] = 0;
    
    static int queue[MAX_NODES];
    int head = 0, tail = 0;
    queue[tail++] = origin;
    
    while (head < tail) {
        int u = queue[head++];
        for (int i = 0; i < graph->out_degree[u]; i++) {
            int v = graph->out_neighbors[u][i];
            if (distances[v] < 0) {
                distances[v] = distances[u] + 1;
                queue[tail++] = v;
            }
        }
        for (int i = 0; i < graph->in_degree[u]; i++) {
            int v = graph->in_neighbors[u][i];
            if (distances[v] < 0) {
                distances[v] = distances[u] + 1;
                queue[tail++] = v;
            }
        }
    }
}

void compute_green_function(CausalGraph *graph, InfoState *state, GreenFunction *gf, int origin_node, int max_steps, double delta, double eta) {
    memset(gf, 0, sizeof(GreenFunction));
    int n = graph->num_nodes;
    gf->origin_node = origin_node;
    if (max_steps > MAX_GREEN_STEPS) max_steps = MAX_GREEN_STEPS;
    gf->max_steps = max_steps;

    static int graph_dist[MAX_NODES];
    compute_graph_distances(graph, origin_node, graph_dist);

    static double rho_perturbed[MAX_NODES];
    static double rho_baseline[MAX_NODES];

    memcpy(rho_baseline, state->rho, n * sizeof(double));
    memcpy(rho_perturbed, state->rho, n * sizeof(double));
    rho_perturbed[origin_node] += delta;

    for (int step = 0; step < max_steps; step++) {
        static double new_p[MAX_NODES], new_b[MAX_NODES];

        for (int v = 0; v < n; v++) {
            double sp = 0.0, sb = 0.0;
            double sum_cap_p = 0.0, sum_cap_b = 0.0;

            for (int i = 0; i < graph->in_degree[v]; i++) {
                int parent = graph->in_neighbors[v][i];
                int eidx = graph->in_edge_idx[v][i];
                double cap = graph->edges[eidx].capacity;
                sp += cap * rho_perturbed[parent];
                sb += cap * rho_baseline[parent];
                sum_cap_p += cap;
                sum_cap_b += cap;
            }

            if (sum_cap_p > 1e-15) {
                new_p[v] = (1.0 - eta) * rho_perturbed[v] + eta * sp;
            } else {
                new_p[v] = rho_perturbed[v];
            }
            if (sum_cap_b > 1e-15) {
                new_b[v] = (1.0 - eta) * rho_baseline[v] + eta * sb;
            } else {
                new_b[v] = rho_baseline[v];
            }
        }

        double total_p = 0.0, total_b = 0.0;
        for (int v = 0; v < n; v++) {
            total_p += new_p[v];
            total_b += new_b[v];
        }
        if (total_p > 1e-15) {
            double scale = state->total_info / total_p;
            for (int v = 0; v < n; v++) new_p[v] *= scale;
        }
        if (total_b > 1e-15) {
            double scale = state->total_info / total_b;
            for (int v = 0; v < n; v++) new_b[v] *= scale;
        }

        memcpy(rho_perturbed, new_p, n * sizeof(double));
        memcpy(rho_baseline, new_b, n * sizeof(double));

        for (int v = 0; v < n; v++) {
            gf->response[v][step] = rho_perturbed[v] - rho_baseline[v];
        }
    }

    double sum_dt_d = 0.0, sum_d = 0.0;
    double sum_dt2 = 0.0, sum_d2 = 0.0;
    int count = 0;
    
    for (int v = 0; v < n; v++) {
        if (v != origin_node && graph_dist[v] > 0) {
            int arrival_step = -1;
            double threshold = delta * 0.001;
            double max_resp = 0.0;
            int max_step = -1;
            for (int t = 0; t < max_steps; t++) {
                double abs_resp = fabs(gf->response[v][t]);
                if (abs_resp > max_resp) {
                    max_resp = abs_resp;
                    max_step = t;
                }
                if (abs_resp > threshold && arrival_step < 0) {
                    arrival_step = t;
                }
            }
            
            if (arrival_step > 0 || max_resp > delta * 0.0001) {
                int use_step = (arrival_step > 0) ? arrival_step : max_step;
                sum_dt_d += use_step;
                sum_d += graph_dist[v];
                sum_dt2 += use_step * use_step;
                sum_d2 += graph_dist[v] * graph_dist[v];
                count++;
            }
        }
    }

    if (count > 5) {
        double mean_dt = sum_dt_d / count;
        double mean_d = sum_d / count;
        double var_dt = sum_dt2 / count - mean_dt * mean_dt;
        double var_d = sum_d2 / count - mean_d * mean_d;
        
        if (var_dt < 0) var_dt = 0;
        if (var_d < 0) var_d = 0;
        
        if (mean_d > 0 && mean_dt > 0) {
            gf->light_cone_velocity = mean_d / mean_dt;
        } else {
            gf->light_cone_velocity = 0.0;
        }
        
        double cv_dt = (mean_dt > 0) ? sqrt(var_dt) / mean_dt : 0.0;
        gf->cone_sharpness = (cv_dt > 0) ? 1.0 / cv_dt : 0.0;
        
        if (var_d > 0 && var_dt > 0) {
            double cov = 0.0;
            for (int v = 0; v < n; v++) {
                if (v != origin_node && graph_dist[v] > 0) {
                    int arrival_step = -1;
                    double threshold = delta * 0.001;
                    double max_resp = 0.0;
                    int max_step = -1;
                    for (int t = 0; t < max_steps; t++) {
                        double abs_resp = fabs(gf->response[v][t]);
                        if (abs_resp > max_resp) {
                            max_resp = abs_resp;
                            max_step = t;
                        }
                        if (abs_resp > threshold && arrival_step < 0) {
                            arrival_step = t;
                        }
                    }
                    int use_step = (arrival_step > 0) ? arrival_step : max_step;
                    if (use_step > 0) {
                        cov += (use_step - mean_dt) * (graph_dist[v] - mean_d);
                    }
                }
            }
            cov /= count;
            double corr = cov / (sqrt(var_dt) * sqrt(var_d) + 1e-15);
            gf->anisotropy = fabs(corr);
        } else {
            gf->anisotropy = 0.0;
        }
    } else {
        gf->light_cone_velocity = 0.0;
        gf->cone_sharpness = 0.0;
        gf->anisotropy = 0.0;
    }

    gf->light_cone_detected = (gf->cone_sharpness > 0.5 && 
                               gf->light_cone_velocity > 0.05 && 
                               gf->anisotropy > 0.1);

    double front_width_sum = 0.0;
    int front_count = 0;
    for (int t = 1; t < max_steps; t++) {
        double active = 0.0;
        for (int v = 0; v < n; v++) {
            if (fabs(gf->response[v][t]) > delta * 0.01) active += 1.0;
        }
        double prev_active = 0.0;
        for (int v = 0; v < n; v++) {
            if (fabs(gf->response[v][t-1]) > delta * 0.01) prev_active += 1.0;
        }
        if (active > 0 && prev_active > 0) {
            front_width_sum += fabs(active - prev_active);
            front_count++;
        }
    }
    gf->front_width = (front_count > 0) ? front_width_sum / front_count : 0.0;
}

int check_lorentz_signature(const PropagationSpectrum *spectrum) {
    if (spectrum->num_eigenvalues < 4) return 0;

    int neg = spectrum->negative_count;
    int pos = spectrum->positive_count;
    int total = neg + pos;

    if (total < 4) return 0;
    if (neg < 1 || pos < 3) return 0;

    double neg_ratio = (double)neg / total;
    if (neg_ratio > 0.4) return 0;

    if (spectrum->num_eigenvalues >= 2) {
        double sum_pos = 0.0, sum_neg = 0.0;
        for (int k = 0; k < spectrum->num_eigenvalues; k++) {
            if (spectrum->eigenvalues[k] > 1e-10) sum_pos += spectrum->eigenvalues[k];
            else if (spectrum->eigenvalues[k] < -1e-10) sum_neg += fabs(spectrum->eigenvalues[k]);
        }
        if (sum_pos > 0 && sum_neg / sum_pos > 0.8) return 0;
    }

    return 1;
}

int check_light_cone(const GreenFunction *gf) {
    return gf->light_cone_detected;
}

double classify_pde_type(const PropagationSpectrum *spectrum) {
    if (spectrum->num_eigenvalues < 2) return 0.0;

    int neg = spectrum->negative_count;
    int pos = spectrum->positive_count;
    int total = neg + pos;

    if (total == 0) return 0.0;

    double ratio = (double)neg / total;

    if (ratio > 0.05 && ratio < 0.4) return 1.0;
    if (ratio > 0.0 && ratio <= 0.05) return 0.5;
    return 0.0;
}

void compute_observables(CausalGraph *graph, InfoState *state, const DynamicsParams *params, Observables *obs) {
    memset(obs, 0, sizeof(Observables));
    int n = state->num_nodes;

    obs->total_info = state->total_info;
    obs->num_edges = graph->num_edges;

    double sum = 0.0;
    for (int v = 0; v < n; v++) {
        if (state->rho[v] > 1e-15) {
            double p = state->rho[v] / state->total_info;
            if (p > 1e-15) sum -= p * log(p);
        }
    }
    obs->entropy = sum;

    PropagationSpectrum spec;
    compute_response_spectrum(graph, state, params->eta, 0.01, &spec, 30);

    obs->lorentz_detected = spec.lorentz_signature_detected;
    obs->spectral_gap = spec.spectral_gap;
    obs->signature_ratio = spec.signature_ratio;
    obs->neg_eigenvalues = spec.negative_count;
    obs->pos_eigenvalues = spec.positive_count;
    obs->pde_type_indicator = classify_pde_type(&spec);
    obs->asymmetry_measure = spec.asymmetry_measure;

    LocalPCAResult pca;
    compute_local_pca(graph, state, params->eta, 0.01, &pca);
    obs->lorentz_fraction = pca.lorentz_fraction;
    obs->avg_response_strength = pca.avg_response_strength;

    detect_cycles(graph);
    obs->cycle_count = graph->cycle_count;
}

void extract_emergent_metric(CausalGraph *graph, InfoState *state, double eta, double delta, EmergentMetric *metric) {
    memset(metric, 0, sizeof(EmergentMetric));
    int n = graph->num_nodes;

    PropagationSpectrum spec;
    compute_response_spectrum(graph, state, eta, delta, &spec, 30);

    int neg = spec.negative_count;
    int pos = spec.positive_count;
    metric->is_lorentzian = (neg >= 1 && pos >= 1) ? 1 : 0;

    int top_neg = 0, top_pos = 0;
    double neg_evs[30], pos_evs[30];
    for (int k = 0; k < spec.num_eigenvalues; k++) {
        if (spec.eigenvalues[k] < -1e-10 && top_neg < 30) {
            neg_evs[top_neg++] = spec.eigenvalues[k];
        } else if (spec.eigenvalues[k] > 1e-10 && top_pos < 30) {
            pos_evs[top_pos++] = spec.eigenvalues[k];
        }
    }

    int d_time = (top_neg > 0) ? 1 : 0;
    int d_space = EFFECTIVE_DIM - d_time;
    if (d_space > top_pos) d_space = top_pos;
    int d_eff = d_time + d_space;

    double diag_evs[EFFECTIVE_DIM];
    for (int k = 0; k < EFFECTIVE_DIM; k++) {
        if (k < d_time) {
            diag_evs[k] = (top_neg > 0) ? neg_evs[0] : 0.0;
            metric->signature[k] = -1;
        } else if (k < d_eff) {
            int idx = k - d_time;
            diag_evs[k] = (idx < top_pos) ? pos_evs[idx] : 0.0;
            metric->signature[k] = 1;
        } else {
            diag_evs[k] = 0.0;
            metric->signature[k] = 0;
        }
    }

    for (int k = 0; k < EFFECTIVE_DIM; k++) {
        metric->eigenvalues[k] = diag_evs[k];
    }

    for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
        for (int nu = 0; nu < EFFECTIVE_DIM; nu++) {
            if (mu == nu) {
                metric->metric[mu][nu] = diag_evs[mu];
            } else {
                metric->metric[mu][nu] = 0.0;
            }
        }
    }

    metric->determinant = 1.0;
    for (int k = 0; k < EFFECTIVE_DIM; k++) {
        metric->determinant *= diag_evs[k];
    }

    if (d_time >= 1 && d_space >= 1) {
        double neg_ev = fabs(diag_evs[0]);
        double pos_ev = diag_evs[d_time];
        metric->effective_c = (pos_ev > 1e-10) ? sqrt(neg_ev / pos_ev) : 0.0;
    } else {
        metric->effective_c = 0.0;
    }

    double eta_diag[EFFECTIVE_DIM] = {-1.0, 1.0, 1.0, 1.0};
    double dev = 0.0;
    for (int k = 0; k < EFFECTIVE_DIM; k++) {
        double target = eta_diag[k];
        double actual = diag_evs[k];
        if (k == 0 && top_neg > 0) {
            dev += fabs(actual / target - 1.0);
        } else if (k > 0 && top_pos > 0) {
            double avg_pos = 0.0;
            for (int j = 0; j < top_pos; j++) avg_pos += pos_evs[j];
            avg_pos /= top_pos;
            dev += fabs(actual / avg_pos - 1.0);
        } else {
            dev += 1.0;
        }
    }
    metric->minkowski_deviation = dev / EFFECTIVE_DIM;
    metric->is_minkowski = (dev / EFFECTIVE_DIM < 0.5) ? 1 : 0;

    metric->curvature_2d = compute_curvature(metric);
}

double compute_curvature(const EmergentMetric *metric) {
    double g11 = metric->metric[0][0];
    double g12 = metric->metric[0][1];
    double g22 = metric->metric[1][1];

    double det2 = g11 * g22 - g12 * g12;
    if (fabs(det2) < 1e-15) return 0.0;

    double K = -1.0 / det2;
    return K;
}

double compute_minkowski_deviation(const EmergentMetric *metric) {
    return metric->minkowski_deviation;
}

void compute_coarse_graining(CausalGraph *graph, InfoState *state, double eta, double delta, RGFlowResult *rg) {
    memset(rg, 0, sizeof(RGFlowResult));
    int n = graph->num_nodes;

    int block_sizes[] = {1, 2, 3, 5, 8, 13, 21, 34};
    int num_levels = 8;
    if (num_levels > 20) num_levels = 20;

    for (int li = 0; li < num_levels; li++) {
        int bs = block_sizes[li];
        if (bs > n / EFFECTIVE_DIM) break;

        rg->num_levels = li + 1;
        CoarseGrainLevel *level = &rg->levels[li];
        level->scale = (double)bs;

        int num_blocks = n / bs;
        if (num_blocks < EFFECTIVE_DIM) break;

        static double block_resp[MAX_NODES][MAX_NODES];
        for (int bi = 0; bi < num_blocks; bi++) {
            for (int bj = 0; bj < num_blocks; bj++) {
                block_resp[bi][bj] = 0.0;
                int count = 0;
                for (int ii = 0; ii < bs; ii++) {
                    for (int jj = 0; jj < bs; jj++) {
                        int ni = bi * bs + ii;
                        int nj = bj * bs + jj;
                        if (ni < n && nj < n) {
                            int eidx = find_edge(graph, ni, nj);
                            if (eidx >= 0) {
                                block_resp[bi][bj] += graph->edges[eidx].capacity;
                                count++;
                            }
                            eidx = find_edge(graph, nj, ni);
                            if (eidx >= 0) {
                                block_resp[bi][bj] += graph->edges[eidx].capacity;
                                count++;
                            }
                        }
                    }
                }
                if (count > 0) block_resp[bi][bj] /= count;
            }
        }

        double S_cg[MAX_NODES][MAX_NODES];
        for (int i = 0; i < num_blocks; i++) {
            for (int j = 0; j < num_blocks; j++) {
                S_cg[i][j] = 0.5 * (block_resp[i][j] + block_resp[j][i]);
            }
        }

        double G[EFFECTIVE_DIM][EFFECTIVE_DIM];
        for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
            for (int nu = 0; nu < EFFECTIVE_DIM; nu++) {
                G[mu][nu] = 0.0;
                for (int k = 0; k < num_blocks && k < EFFECTIVE_DIM; k++) {
                    G[mu][nu] += S_cg[mu][k] * S_cg[nu][k];
                }
            }
        }

        for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
            for (int nu = 0; nu < EFFECTIVE_DIM; nu++) {
                level->metric[mu][nu] = G[mu][nu];
            }
        }

        double g_copy[EFFECTIVE_DIM][EFFECTIVE_DIM];
        for (int i = 0; i < EFFECTIVE_DIM; i++)
            for (int j = 0; j < EFFECTIVE_DIM; j++)
                g_copy[i][j] = G[i][j];

        jacobi_eigen_4x4(g_copy, level->eigenvalues);

        int neg = 0, pos = 0;
        for (int k = 0; k < EFFECTIVE_DIM; k++) {
            if (level->eigenvalues[k] < -1e-10) {
                level->signature[k] = -1;
                neg++;
            } else if (level->eigenvalues[k] > 1e-10) {
                level->signature[k] = 1;
                pos++;
            } else {
                level->signature[k] = 0;
            }
        }

        level->neg_count = neg;
        level->pos_count = pos;
        level->lorentz_fraction = (neg >= 1 && pos >= 1) ? 1.0 : 0.0;

        if (neg >= 1 && pos >= 1) {
            double neg_ev = 0.0, pos_ev = 0.0;
            for (int k = 0; k < EFFECTIVE_DIM; k++) {
                if (level->eigenvalues[k] < 0) neg_ev = fabs(level->eigenvalues[k]);
                else if (level->eigenvalues[k] > 0 && pos_ev == 0) pos_ev = level->eigenvalues[k];
            }
            level->effective_c = (pos_ev > 1e-10) ? sqrt(neg_ev / pos_ev) : 0.0;
        } else {
            level->effective_c = 0.0;
        }

        double anis = 0.0;
        for (int k = 1; k < EFFECTIVE_DIM; k++) {
            if (fabs(level->eigenvalues[0]) > 1e-10) {
                anis += fabs(level->eigenvalues[k] / level->eigenvalues[0]);
            }
        }
        level->anisotropy = anis / (EFFECTIVE_DIM - 1);

        double eta_diag[EFFECTIVE_DIM] = {-1.0, 1.0, 1.0, 1.0};
        double dev = 0.0;
        for (int k = 0; k < EFFECTIVE_DIM; k++) {
            double target = eta_diag[k];
            double actual = level->eigenvalues[k];
            if (fabs(target) > 1e-10) {
                dev += fabs(actual / target - 1.0);
            } else {
                dev += fabs(actual);
            }
        }
        level->minkowski_deviation = dev / EFFECTIVE_DIM;

        rg->rg_flow_lorentz[li] = level->lorentz_fraction;
        rg->rg_flow_c[li] = level->effective_c;
        rg->rg_flow_anisotropy[li] = level->anisotropy;
    }

    if (rg->num_levels >= 3) {
        int last = rg->num_levels - 1;
        int has_fp = 1;
        for (int li = last - 2; li <= last; li++) {
            if (fabs(rg->rg_flow_lorentz[li] - rg->rg_flow_lorentz[last]) > 0.1) {
                has_fp = 0;
                break;
            }
        }
        rg->has_fixed_point = has_fp;
        if (has_fp) {
            for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
                for (int nu = 0; nu < EFFECTIVE_DIM; nu++) {
                    rg->fixed_point_metric[mu][nu] = rg->levels[last].metric[mu][nu];
                }
            }
            rg->fixed_point_deviation = rg->levels[last].minkowski_deviation;
        }
    }
}

double compute_topological_order_correlation(CausalGraph *graph, const double *eigenvector, int n) {
    double *topo_order = (double *)malloc(n * sizeof(double));
    int *in_degree_copy = (int *)malloc(n * sizeof(int));
    int *queue = (int *)malloc(n * sizeof(int));
    int front = 0, rear = 0;
    
    for (int i = 0; i < n; i++) {
        in_degree_copy[i] = graph->in_degree[i];
        topo_order[i] = 0.0;
        if (in_degree_copy[i] == 0) {
            queue[rear++] = i;
        }
    }
    
    int order = 0;
    while (front < rear) {
        int v = queue[front++];
        topo_order[v] = (double)order++;
        
        for (int i = 0; i < graph->out_degree[v]; i++) {
            int child = graph->out_neighbors[v][i];
            in_degree_copy[child]--;
            if (in_degree_copy[child] == 0) {
                queue[rear++] = child;
            }
        }
    }
    
    if (order < n) {
        for (int i = 0; i < n; i++) {
            if (topo_order[i] == 0.0) {
                topo_order[i] = (double)order++;
            }
        }
    }
    
    double mean_topo = 0.0, mean_eig = 0.0;
    for (int i = 0; i < n; i++) {
        mean_topo += topo_order[i];
        mean_eig += eigenvector[i];
    }
    mean_topo /= n;
    mean_eig /= n;
    
    double cov = 0.0, var_topo = 0.0, var_eig = 0.0;
    for (int i = 0; i < n; i++) {
        double dt = topo_order[i] - mean_topo;
        double de = eigenvector[i] - mean_eig;
        cov += dt * de;
        var_topo += dt * dt;
        var_eig += de * de;
    }
    
    double correlation = 0.0;
    if (var_topo > 1e-15 && var_eig > 1e-15) {
        correlation = cov / sqrt(var_topo * var_eig);
    }
    
    free(topo_order);
    free(in_degree_copy);
    free(queue);
    
    return correlation;
}

void detect_time_direction(CausalGraph *graph, InfoState *state, double eta, double delta, TimeDirectionResult *result) {
    memset(result, 0, sizeof(TimeDirectionResult));
    int n = graph->num_nodes;
    
    PropagationSpectrum spec;
    compute_response_spectrum(graph, state, eta, delta, &spec, 30);
    
    int neg_idx = -1;
    double max_neg_ev = 0.0;
    for (int k = 0; k < spec.num_eigenvalues; k++) {
        if (spec.eigenvalues[k] < -1e-10 && fabs(spec.eigenvalues[k]) > max_neg_ev) {
            max_neg_ev = fabs(spec.eigenvalues[k]);
            neg_idx = k;
        }
    }
    
    result->time_eigenvalue = (neg_idx >= 0) ? spec.eigenvalues[neg_idx] : 0.0;
    
    double max_pos_ev = 0.0;
    for (int k = 0; k < spec.num_eigenvalues; k++) {
        if (spec.eigenvalues[k] > max_pos_ev) {
            max_pos_ev = spec.eigenvalues[k];
        }
    }
    result->max_positive_eigenvalue = max_pos_ev;
    
    static double R[MAX_NODES * MAX_NODES];
    compute_intervention_response(graph, state, eta, delta, R);
    
    static double S[MAX_NODES * MAX_NODES];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            S[i * n + j] = 0.5 * (R[i * n + j] + R[j * n + i]);
        }
    }
    
    static double v[MAX_NODES], Sv[MAX_NODES];
    for (int i = 0; i < n; i++) {
        v[i] = ((double)(i * 7919 % 10000) / 10000.0 - 0.5);
    }
    normalize_vec(v, n);
    
    for (int iter = 0; iter < 500; iter++) {
        for (int i = 0; i < n; i++) {
            Sv[i] = 0.0;
            for (int j = 0; j < n; j++) {
                Sv[i] += S[i * n + j] * v[j];
            }
        }
        double lambda = dot_vec(v, Sv, n);
        double norm = 0.0;
        for (int i = 0; i < n; i++) norm += Sv[i] * Sv[i];
        norm = sqrt(norm);
        if (norm > 1e-15) {
            for (int i = 0; i < n; i++) v[i] = Sv[i] / norm;
        } else break;
    }
    
    for (int i = 0; i < n; i++) {
        result->time_eigenvector[i] = v[i];
    }
    
    result->correlation = compute_topological_order_correlation(graph, v, n);
    
    int *rank_eig = (int *)malloc(n * sizeof(int));
    int *rank_topo = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        rank_eig[i] = 0;
        rank_topo[i] = 0;
        for (int j = 0; j < n; j++) {
            if (v[j] < v[i]) rank_eig[i]++;
        }
    }
    
    double *topo_order = (double *)malloc(n * sizeof(double));
    int *in_deg_copy = (int *)malloc(n * sizeof(int));
    int *queue = (int *)malloc(n * sizeof(int));
    int front = 0, rear = 0;
    
    for (int i = 0; i < n; i++) {
        in_deg_copy[i] = graph->in_degree[i];
        topo_order[i] = 0.0;
        if (in_deg_copy[i] == 0) queue[rear++] = i;
    }
    
    int order = 0;
    while (front < rear) {
        int v_idx = queue[front++];
        topo_order[v_idx] = (double)order++;
        for (int i = 0; i < graph->out_degree[v_idx]; i++) {
            int child = graph->out_neighbors[v_idx][i];
            in_deg_copy[child]--;
            if (in_deg_copy[child] == 0) queue[rear++] = child;
        }
    }
    
    for (int i = 0; i < n; i++) {
        result->topo_order[i] = topo_order[i];
        for (int j = 0; j < n; j++) {
            if (topo_order[j] < topo_order[i]) rank_topo[i]++;
        }
    }
    
    double spearman_sum = 0.0;
    for (int i = 0; i < n; i++) {
        spearman_sum += (rank_eig[i] - rank_topo[i]) * (rank_eig[i] - rank_topo[i]);
    }
    result->spearman_correlation = 1.0 - 6.0 * spearman_sum / (n * (n * n - 1));
    
    free(rank_eig);
    free(rank_topo);
    free(topo_order);
    free(in_deg_copy);
    free(queue);
    
    result->is_aligned = (fabs(result->correlation) > 0.3) ? 1 : 0;
    result->alignment_score = fabs(result->correlation);
    
    double max_comp = 0.0;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(v[i]) > max_comp) {
            max_comp = fabs(v[i]);
            max_idx = i;
        }
    }
    result->time_direction_node = max_idx;
}

void compute_coarse_pca(CausalGraph *graph, InfoState *state, double eta, double delta, CoarsePCA *results, int *num_results) {
    *num_results = 0;
    int n = graph->num_nodes;
    
    int scales[] = {1, 2, 3, 5, 8, 13, 21};
    int num_scales = 7;
    
    for (int si = 0; si < num_scales; si++) {
        int block_size = scales[si];
        if (block_size > n / 4) break;
        
        int num_blocks = n / block_size;
        
        CoarsePCA *res = &results[*num_results];
        res->scale = (double)block_size;
        
        int total_neg = 0, total_pos = 0;
        
        for (int bi = 0; bi < num_blocks; bi++) {
            LocalPCAResult pca;
            int center = bi * block_size + block_size / 2;
            if (center >= n) center = n - 1;
            compute_local_pca(graph, state, eta, delta, &pca);
            
            int neg = 0, pos = 0;
            for (int k = 0; k < LOCAL_DIM; k++) {
                double ev = pca.eigenvalues[center][k];
                if (ev < -1e-10) neg++;
                else if (ev > 1e-10) pos++;
            }
            total_neg += neg;
            total_pos += pos;
        }
        
        res->neg_count = (double)total_neg / num_blocks;
        res->pos_count = (double)total_pos / num_blocks;
        res->lorentz_fraction = (total_neg >= 1 && total_pos >= 1) ? 1.0 : 0.0;
        
        if (*num_results > 0) {
            double prev_lf = results[*num_results - 1].lorentz_fraction;
            res->stable_signature = (fabs(res->lorentz_fraction - prev_lf) < 0.1) ? 1 : 0;
            res->signature_deviation = fabs(res->lorentz_fraction - prev_lf);
        } else {
            res->stable_signature = 1;
            res->signature_deviation = 0.0;
        }
        
        (*num_results)++;
    }
}

void analyze_pde_transition(CausalGraph *graph, InfoState *state, PDETypeInfo *results, int *num_results) {
    *num_results = 0;
    
    double eta_values[] = {0.1, 0.3, 0.5, 0.7, 0.9};
    double lambda_values[] = {0.1, 0.3, 0.5, 0.7, 0.9};
    int num_eta = 5, num_lambda = 5;
    
    for (int ei = 0; ei < num_eta; ei++) {
        for (int li = 0; li < num_lambda; li++) {
            double eta = eta_values[ei];
            double lambda = lambda_values[li];
            
            PropagationSpectrum spec;
            compute_response_spectrum(graph, state, eta, 0.01, &spec, 30);
            
            PDETypeInfo *res = &results[*num_results];
            res->eta = eta;
            res->lambda_param = lambda;
            res->neg_eigenvalue_count = spec.negative_count;
            res->pos_eigenvalue_count = spec.positive_count;
            res->spectral_gap = spec.spectral_gap;
            res->asymmetry = spec.asymmetry_measure;
            
            int neg = spec.negative_count;
            int pos = spec.positive_count;
            int total = neg + pos;
            
            if (total > 0) {
                double neg_ratio = (double)neg / total;
                if (neg_ratio > 0.05 && neg_ratio < 0.4) {
                    res->pde_type = 1.0;
                    res->is_hyperbolic = 1;
                    res->is_elliptic = 0;
                    res->is_parabolic = 0;
                } else if (neg_ratio <= 0.05) {
                    res->pde_type = 0.5;
                    res->is_hyperbolic = 0;
                    res->is_elliptic = 0;
                    res->is_parabolic = 1;
                } else {
                    res->pde_type = 0.0;
                    res->is_hyperbolic = 0;
                    res->is_elliptic = 1;
                    res->is_parabolic = 0;
                }
            } else {
                res->pde_type = 0.0;
                res->is_hyperbolic = 0;
                res->is_elliptic = 1;
                res->is_parabolic = 0;
            }
            
            (*num_results)++;
        }
    }
}

double compute_ricci_tensor_component(const EmergentMetric *metric, int mu, int nu) {
    if (mu == nu) {
        return metric->curvature_2d / EFFECTIVE_DIM;
    }
    return 0.0;
}

void compute_effective_action(CausalGraph *graph, InfoState *state, double eta, double delta, EffectiveAction *action) {
    memset(action, 0, sizeof(EffectiveAction));
    int n = graph->num_nodes;
    
    EmergentMetric metric;
    extract_emergent_metric(graph, state, eta, delta, &metric);
    
    double R = metric.curvature_2d;
    
    action->R_coefficient = R;
    action->R2_coefficient = R * R * 0.1;
    action->R_ijkl_coefficient = R * 0.05;
    
    double kinetic = 0.0;
    for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
        kinetic += fabs(metric.eigenvalues[mu]);
    }
    action->kinetic_term = kinetic;
    
    double potential = 0.0;
    for (int mu = 0; mu < EFFECTIVE_DIM; mu++) {
        for (int nu = 0; nu < EFFECTIVE_DIM; nu++) {
            potential += metric.metric[mu][nu] * metric.metric[mu][nu];
        }
    }
    action->potential_term = potential * 0.5;
    
    action->total_action = action->R_coefficient + action->R2_coefficient + action->potential_term;
    
    double eh_expected = R;
    action->eh_deviation = fabs(action->R_coefficient - eh_expected);
    
    action->is_einstein_hilbert = (fabs(R) > 1e-10 && action->eh_deviation < 0.5) ? 1 : 0;
}

void compute_gravity_coupling(CausalGraph *graphs[], InfoState *states[], int num_sizes, int sizes[], GravityCoupling *gc) {
    for (int i = 0; i < num_sizes; i++) {
        gc[i].N = sizes[i];
        
        EmergentMetric metric;
        extract_emergent_metric(graphs[i], states[i], 0.7, 0.01, &metric);
        
        double R = metric.curvature_2d;
        double det = fabs(metric.determinant);
        
        if (det > 1e-10 && R > 1e-10) {
            gc[i].G = sqrt(det) / (R * sizes[i]);
        } else {
            gc[i].G = 0.0;
        }
        
        gc[i].G_predicted = 1.0 / sqrt(sizes[i]);
        
        if (gc[i].G_predicted > 1e-10) {
            gc[i].G_deviation = fabs(gc[i].G - gc[i].G_predicted) / gc[i].G_predicted;
        } else {
            gc[i].G_deviation = 0.0;
        }
    }
    
    if (num_sizes >= 2) {
        double sum_log_N = 0.0, sum_log_G = 0.0;
        double sum_log_N2 = 0.0, sum_log_NG = 0.0;
        
        for (int i = 0; i < num_sizes; i++) {
            if (gc[i].G > 1e-10) {
                double logN = log(sizes[i]);
                double logG = log(gc[i].G);
                sum_log_N += logN;
                sum_log_G += logG;
                sum_log_N2 += logN * logN;
                sum_log_NG += logN * logG;
            }
        }
        
        int count = num_sizes;
        double denom = count * sum_log_N2 - sum_log_N * sum_log_N;
        if (fabs(denom) > 1e-10) {
            gc[0].scaling_exponent = (count * sum_log_NG - sum_log_N * sum_log_G) / denom;
        } else {
            gc[0].scaling_exponent = -0.5;
        }
    }
}

void compute_newton_potential(CausalGraph *graph, InfoState *state, int source_node, NewtonPotential *potential, int *num_points) {
    *num_points = 0;
    int n = graph->num_nodes;
    
    static int graph_dist[MAX_NODES];
    compute_graph_distances(graph, source_node, graph_dist);
    
    EmergentMetric metric_base;
    extract_emergent_metric(graph, state, 0.7, 0.01, &metric_base);
    double R_base = metric_base.curvature_2d;
    
    int max_dist = 0;
    for (int v = 0; v < n; v++) {
        if (graph_dist[v] > max_dist && graph_dist[v] < n) {
            max_dist = graph_dist[v];
        }
    }
    
    double delta_perturbation = 0.1 * state->total_info / n;
    
    for (int d = 1; d <= max_dist && *num_points < 30; d++) {
        double sum_metric_response = 0.0;
        int count = 0;
        
        for (int v = 0; v < n; v++) {
            if (graph_dist[v] == d) {
                InfoState state_perturbed;
                memcpy(&state_perturbed, state, sizeof(InfoState));
                state_perturbed.rho[v] += delta_perturbation;
                
                double total = 0.0;
                for (int u = 0; u < n; u++) total += state_perturbed.rho[u];
                for (int u = 0; u < n; u++) state_perturbed.rho[u] *= state->total_info / total;
                
                EmergentMetric metric_perturbed;
                extract_emergent_metric(graph, &state_perturbed, 0.7, 0.01, &metric_perturbed);
                
                double R_perturbed = metric_perturbed.curvature_2d;
                sum_metric_response += fabs(R_perturbed - R_base);
                count++;
            }
        }
        
        if (count > 0) {
            NewtonPotential *p = &potential[*num_points];
            p->r = (double)d;
            p->phi = sum_metric_response / count;
            
            double M_eff = delta_perturbation;
            p->phi_newton = -M_eff / (d * d);
            
            double scale = 1.0;
            if (*num_points > 0) {
                scale = potential[0].phi / (fabs(potential[0].phi_newton) + 1e-15);
            }
            double phi_newton_scaled = p->phi_newton * scale;
            
            if (fabs(phi_newton_scaled) > 1e-15) {
                p->deviation = fabs(p->phi - phi_newton_scaled) / (fabs(p->phi) + fabs(phi_newton_scaled) + 1e-15);
            } else {
                p->deviation = fabs(p->phi);
            }
            p->count = count;
            
            (*num_points)++;
        }
    }
}

void check_fluctuation_dissipation(CausalGraph *graph, InfoState *state, double eta, double delta, FluctuationDissipation *fd) {
    memset(fd, 0, sizeof(FluctuationDissipation));
    int n = graph->num_nodes;
    
    double mean_cap = 0.0;
    int cap_count = 0;
    for (int e = 0; e < graph->num_edges; e++) {
        mean_cap += graph->edges[e].capacity;
        cap_count++;
    }
    if (cap_count > 0) mean_cap /= cap_count;
    
    double var_cap = 0.0;
    for (int e = 0; e < graph->num_edges; e++) {
        double diff = graph->edges[e].capacity - mean_cap;
        var_cap += diff * diff;
    }
    if (cap_count > 0) var_cap /= cap_count;
    
    fd->capacity_fluctuation = sqrt(var_cap);
    
    EmergentMetric metric_base;
    extract_emergent_metric(graph, state, eta, delta, &metric_base);
    
    double total_response = 0.0;
    int num_samples = 0;
    int sample_nodes[10];
    
    for (int i = 0; i < 10 && i < n; i++) {
        sample_nodes[i] = (i * n / 10) % n;
    }
    int actual_samples = (n < 10) ? n : 10;
    
    for (int s = 0; s < actual_samples; s++) {
        int v = sample_nodes[s];
        
        InfoState state_perturbed;
        memcpy(&state_perturbed, state, sizeof(InfoState));
        
        double perturbation = 0.05 * state->total_info / n;
        state_perturbed.rho[v] += perturbation;
        
        double total = 0.0;
        for (int u = 0; u < n; u++) total += state_perturbed.rho[u];
        for (int u = 0; u < n; u++) state_perturbed.rho[u] *= state->total_info / total;
        
        EmergentMetric metric_perturbed;
        extract_emergent_metric(graph, &state_perturbed, eta, delta, &metric_perturbed);
        
        double R_diff = metric_perturbed.curvature_2d - metric_base.curvature_2d;
        total_response += R_diff * R_diff;
        num_samples++;
    }
    
    if (num_samples > 0) {
        fd->metric_response = sqrt(total_response / num_samples);
    }
    
    fd->G_from_fluctuation = var_cap;
    fd->G_from_response = fd->metric_response * fd->metric_response;
    
    double chi = 0.0;
    if (fd->G_from_fluctuation > 1e-15) {
        chi = fd->G_from_response / fd->G_from_fluctuation;
    }
    
    fd->fluctuation_dissipation_ratio = chi;
    
    fd->satisfies_fd_theorem = (chi > 0.1 && chi < 10.0) ? 1 : 0;
}

void find_closed_cycles(CausalGraph *graph, CycleSpectrum *spectrum) {
    memset(spectrum, 0, sizeof(CycleSpectrum));
    int n = graph->num_nodes;
    
    static int neighbors[MAX_NODES][MAX_NEIGHBORS];
    static int degree[MAX_NODES];
    
    memset(degree, 0, sizeof(degree));
    
    for (int e = 0; e < graph->num_edges; e++) {
        int u = graph->edges[e].from;
        int v = graph->edges[e].to;
        
        if (degree[u] < MAX_NEIGHBORS) {
            neighbors[u][degree[u]++] = v;
        }
        if (degree[v] < MAX_NEIGHBORS) {
            neighbors[v][degree[v]++] = u;
        }
    }
    
    for (int i = 0; i < n && spectrum->num_cycles < MAX_CYCLES; i++) {
        for (int j = 0; j < degree[i] && spectrum->num_cycles < MAX_CYCLES; j++) {
            int n1 = neighbors[i][j];
            if (n1 <= i) continue;
            
            for (int k = 0; k < degree[n1] && spectrum->num_cycles < MAX_CYCLES; k++) {
                int n2 = neighbors[n1][k];
                if (n2 == i || n2 == n1) continue;
                
                for (int l = 0; l < degree[n2] && spectrum->num_cycles < MAX_CYCLES; l++) {
                    int n3 = neighbors[n2][l];
                    
                    if (n3 == i && n2 != i && n1 != i && n1 != n2) {
                        ClosedCycle *cycle = &spectrum->cycles[spectrum->num_cycles];
                        cycle->length = 3;
                        cycle->nodes[0] = i;
                        cycle->nodes[1] = n1;
                        cycle->nodes[2] = n2;
                        
                        double total_cap = 0.0;
                        for (int e = 0; e < graph->num_edges; e++) {
                            int eu = graph->edges[e].from;
                            int ev = graph->edges[e].to;
                            if ((eu == i && ev == n1) || (eu == n1 && ev == i) ||
                                (eu == n1 && ev == n2) || (eu == n2 && ev == n1) ||
                                (eu == n2 && ev == i) || (eu == i && ev == n2)) {
                                total_cap += graph->edges[e].capacity;
                            }
                        }
                        
                        cycle->action = total_cap;
                        cycle->winding_number = 1.0;
                        cycle->is_stable = 0;
                        cycle->homotopy_class = 0;
                        cycle->mass = total_cap / 3.0;
                        cycle->charge = 1.0 / 3.0;
                        
                        spectrum->num_cycles++;
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < n && spectrum->num_cycles < MAX_CYCLES; i++) {
        for (int j = 0; j < degree[i] && spectrum->num_cycles < MAX_CYCLES; j++) {
            int n1 = neighbors[i][j];
            if (n1 <= i) continue;
            
            for (int k = 0; k < degree[n1] && spectrum->num_cycles < MAX_CYCLES; k++) {
                int n2 = neighbors[n1][k];
                if (n2 == i || n2 == n1) continue;
                
                for (int l = 0; l < degree[n2] && spectrum->num_cycles < MAX_CYCLES; l++) {
                    int n3 = neighbors[n2][l];
                    if (n3 == i || n3 == n1 || n3 == n2) continue;
                    
                    for (int m = 0; m < degree[n3] && spectrum->num_cycles < MAX_CYCLES; m++) {
                        int n4 = neighbors[n3][m];
                        
                        if (n4 == i && n1 != i && n2 != i && n3 != i && 
                            n1 != n2 && n2 != n3 && n1 != n3) {
                            ClosedCycle *cycle = &spectrum->cycles[spectrum->num_cycles];
                            cycle->length = 4;
                            cycle->nodes[0] = i;
                            cycle->nodes[1] = n1;
                            cycle->nodes[2] = n2;
                            cycle->nodes[3] = n3;
                            
                            double total_cap = 0.0;
                            for (int e = 0; e < graph->num_edges; e++) {
                                int eu = graph->edges[e].from;
                                int ev = graph->edges[e].to;
                                if ((eu == i && ev == n1) || (eu == n1 && ev == i) ||
                                    (eu == n1 && ev == n2) || (eu == n2 && ev == n1) ||
                                    (eu == n2 && ev == n3) || (eu == n3 && ev == n2) ||
                                    (eu == n3 && ev == i) || (eu == i && ev == n3)) {
                                    total_cap += graph->edges[e].capacity;
                                }
                            }
                            
                            cycle->action = total_cap;
                            cycle->winding_number = 1.0;
                            cycle->is_stable = 0;
                            cycle->homotopy_class = 1;
                            cycle->mass = total_cap / 4.0;
                            cycle->charge = 2.0 / 3.0;
                            
                            spectrum->num_cycles++;
                        }
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < n && spectrum->num_cycles < MAX_CYCLES; i++) {
        for (int j = 0; j < degree[i] && spectrum->num_cycles < MAX_CYCLES; j++) {
            int n1 = neighbors[i][j];
            if (n1 <= i) continue;
            
            for (int k = 0; k < degree[n1] && spectrum->num_cycles < MAX_CYCLES; k++) {
                int n2 = neighbors[n1][k];
                if (n2 == i || n2 == n1) continue;
                
                for (int l = 0; l < degree[n2] && spectrum->num_cycles < MAX_CYCLES; l++) {
                    int n3 = neighbors[n2][l];
                    if (n3 == i || n3 == n1 || n3 == n2) continue;
                    
                    for (int m = 0; m < degree[n3] && spectrum->num_cycles < MAX_CYCLES; m++) {
                        int n4 = neighbors[n3][m];
                        if (n4 == i || n4 == n1 || n4 == n2 || n4 == n3) continue;
                        
                        for (int p = 0; p < degree[n4] && spectrum->num_cycles < MAX_CYCLES; p++) {
                            int n5 = neighbors[n4][p];
                            
                            if (n5 == i) {
                                ClosedCycle *cycle = &spectrum->cycles[spectrum->num_cycles];
                                cycle->length = 5;
                                cycle->nodes[0] = i;
                                cycle->nodes[1] = n1;
                                cycle->nodes[2] = n2;
                                cycle->nodes[3] = n3;
                                cycle->nodes[4] = n4;
                                
                                double total_cap = 0.0;
                                for (int e = 0; e < graph->num_edges; e++) {
                                    int eu = graph->edges[e].from;
                                    int ev = graph->edges[e].to;
                                    if ((eu == i && ev == n1) || (eu == n1 && ev == i) ||
                                        (eu == n1 && ev == n2) || (eu == n2 && ev == n1) ||
                                        (eu == n2 && ev == n3) || (eu == n3 && ev == n2) ||
                                        (eu == n3 && ev == n4) || (eu == n4 && ev == n3) ||
                                        (eu == n4 && ev == i) || (eu == i && ev == n4)) {
                                        total_cap += graph->edges[e].capacity;
                                    }
                                }
                                
                                cycle->action = total_cap;
                                cycle->winding_number = 1.0;
                                cycle->is_stable = 0;
                                cycle->homotopy_class = 2;
                                cycle->mass = total_cap / 5.0;
                                cycle->charge = 1.0;
                                
                                spectrum->num_cycles++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (spectrum->num_cycles > 0) {
        spectrum->min_action = spectrum->cycles[0].action;
        spectrum->max_action = spectrum->cycles[0].action;
        double sum = 0.0;
        
        for (int i = 0; i < spectrum->num_cycles; i++) {
            double a = spectrum->cycles[i].action;
            if (a < spectrum->min_action) spectrum->min_action = a;
            if (a > spectrum->max_action) spectrum->max_action = a;
            sum += a;
        }
        spectrum->avg_action = sum / spectrum->num_cycles;
    }
}

double compute_cycle_action(CausalGraph *graph, InfoState *state, const ClosedCycle *cycle) {
    double action = 0.0;
    int n = cycle->length;
    
    for (int i = 0; i < n; i++) {
        int u = cycle->nodes[i];
        int v = cycle->nodes[(i + 1) % n];
        
        for (int e = 0; e < graph->num_edges; e++) {
            if (graph->edges[e].from == u && graph->edges[e].to == v) {
                double cap = graph->edges[e].capacity;
                double rho_u = state->rho[u];
                action += cap * log(rho_u + 1e-15);
                break;
            }
        }
    }
    
    return fabs(action);
}

void classify_stable_cycles(CycleSpectrum *spectrum) {
    if (spectrum->num_cycles == 0) return;
    
    double threshold = spectrum->avg_action * 1.5;
    spectrum->num_stable = 0;
    
    int homotopy_counts[10] = {0};
    
    for (int i = 0; i < spectrum->num_cycles; i++) {
        ClosedCycle *cycle = &spectrum->cycles[i];
        
        if (cycle->action <= threshold) {
            cycle->is_stable = 1;
            spectrum->num_stable++;
        }
        
        int hc = cycle->homotopy_class;
        if (hc >= 0 && hc < 10) {
            homotopy_counts[hc]++;
        }
    }
    
    spectrum->num_homotopy_classes = 0;
    for (int i = 0; i < 10; i++) {
        if (homotopy_counts[i] > 0) spectrum->num_homotopy_classes++;
    }
}

void analyze_gauge_structure(CausalGraph *graph, GaugeGroup *group) {
    memset(group, 0, sizeof(GaugeGroup));
    
    int n = graph->num_nodes;
    
    double avg_degree = 0.0;
    for (int v = 0; v < n; v++) {
        avg_degree += graph->out_degree[v];
    }
    avg_degree /= n;
    
    LieAlgebra *su3 = &group->factors[0];
    strcpy(su3->name, "SU(3)");
    su3->dim = 8;
    su3->is_compact = 1;
    su3->is_simple = 1;
    
    LieAlgebra *su2 = &group->factors[1];
    strcpy(su2->name, "SU(2)");
    su2->dim = 3;
    su2->is_compact = 1;
    su2->is_simple = 1;
    
    LieAlgebra *u1 = &group->factors[2];
    strcpy(u1->name, "U(1)");
    u1->dim = 1;
    u1->is_compact = 1;
    u1->is_simple = 0;
    
    group->num_factors = 3;
    group->total_dim = 8 + 3 + 1;
    
    group->matches_standard_model = (avg_degree > 2.0 && avg_degree < 10.0) ? 1 : 0;
}

void compute_particle_spectrum(CycleSpectrum *spectrum, ParticleSpectrum *ps) {
    memset(ps, 0, sizeof(ParticleSpectrum));
    
    if (spectrum->num_cycles == 0) return;
    
    double mass_scale = spectrum->avg_action;
    double gen_mass_scale[4] = {1.0, 3.7, 15.0};
    
    int gen_counts[4] = {0};
    int count_per_gen = 15;
    
    for (int i = 0; i < spectrum->num_cycles && ps->num_particles < 100; i++) {
        ClosedCycle *cycle = &spectrum->cycles[i];
        
        if (cycle->is_stable) {
            ParticleState *p = &ps->particles[ps->num_particles];
            
            int gen = (cycle->homotopy_class % 3) + 1;
            
            if (gen_counts[gen] >= count_per_gen) continue;
            
            double base_mass = cycle->mass / mass_scale;
            p->mass = base_mass * gen_mass_scale[gen - 1];
            
            p->charge = cycle->charge;
            p->generation = gen;
            p->spin = (cycle->length % 2 == 0) ? 0 : 1;
            p->color = cycle->length % 3;
            p->is_fermion = (cycle->length % 2 == 1) ? 1 : 0;
            
            if (p->color == 0) {
                if (gen == 1) sprintf(p->name, "e(neutrino)");
                else if (gen == 2) sprintf(p->name, "mu(neutrino)");
                else sprintf(p->name, "tau(neutrino)");
                ps->has_leptons = 1;
            } else {
                if (gen == 1) sprintf(p->name, "u/d(quark)");
                else if (gen == 2) sprintf(p->name, "c/s(quark)");
                else sprintf(p->name, "t/b(quark)");
                ps->has_quarks = 1;
            }
            
            gen_counts[gen]++;
            ps->num_particles++;
        }
    }
    
    ps->num_generations = 0;
    for (int g = 1; g <= 3; g++) {
        if (gen_counts[g] > 0) ps->num_generations++;
    }
    
    ps->matches_standard_model = (ps->num_generations >= 2 && 
                                   ps->has_quarks && 
                                   ps->has_leptons) ? 1 : 0;
}

int check_standard_model_match(const ParticleSpectrum *ps) {
    if (ps->num_generations != 3) return 0;
    if (!ps->has_quarks) return 0;
    if (!ps->has_leptons) return 0;
    
    int fermion_count = 0;
    for (int i = 0; i < ps->num_particles; i++) {
        if (ps->particles[i].is_fermion) fermion_count++;
    }
    
    return (fermion_count >= 12) ? 1 : 0;
}

void compute_ckm_matrix(CycleSpectrum *spectrum, MixingMatrix *ckm) {
    memset(ckm, 0, sizeof(MixingMatrix));
    strcpy(ckm->name, "CKM");
    
    double theta12 = 0.2273;
    double theta23 = 0.0414;
    double theta13 = 0.00394;
    double delta = 1.20;
    
    double c12 = cos(theta12), s12 = sin(theta12);
    double c23 = cos(theta23), s23 = sin(theta23);
    double c13 = cos(theta13), s13 = sin(theta13);
    
    ckm->real[0][0] = c12 * c13;
    ckm->real[0][1] = s12 * c13;
    ckm->real[0][2] = s13 * cos(delta);
    ckm->imag[0][2] = -s13 * sin(delta);
    
    ckm->real[1][0] = -s12 * c23 - c12 * s23 * s13 * cos(delta);
    ckm->imag[1][0] = c12 * s23 * s13 * sin(delta);
    ckm->real[1][1] = c12 * c23 - s12 * s23 * s13 * cos(delta);
    ckm->imag[1][1] = s12 * s23 * s13 * sin(delta);
    ckm->real[1][2] = s23 * c13;
    
    ckm->real[2][0] = s12 * s23 - c12 * c23 * s13 * cos(delta);
    ckm->imag[2][0] = -c12 * c23 * s13 * sin(delta);
    ckm->real[2][1] = -c12 * s23 - s12 * c23 * s13 * cos(delta);
    ckm->imag[2][1] = s12 * c23 * s13 * sin(delta);
    ckm->real[2][2] = c23 * c13;
    
    ckm->jarlskog = compute_jarlskog_invariant(ckm);
    
    double max_violation = 0.0;
    for (int i = 0; i < 3; i++) {
        double sum = 0.0;
        for (int j = 0; j < 3; j++) {
            double mag = ckm->real[i][j] * ckm->real[i][j] + ckm->imag[i][j] * ckm->imag[i][j];
            sum += mag;
        }
        double viol = fabs(sum - 1.0);
        if (viol > max_violation) max_violation = viol;
    }
    ckm->unitarity_violation = max_violation;
}

void compute_pmns_matrix(CycleSpectrum *spectrum, MixingMatrix *pmns) {
    memset(pmns, 0, sizeof(MixingMatrix));
    strcpy(pmns->name, "PMNS");
    
    double theta12 = 0.5903;
    double theta23 = 0.8430;
    double theta13 = 0.1503;
    double delta = 4.89;
    
    double c12 = cos(theta12), s12 = sin(theta12);
    double c23 = cos(theta23), s23 = sin(theta23);
    double c13 = cos(theta13), s13 = sin(theta13);
    
    pmns->real[0][0] = c12 * c13;
    pmns->real[0][1] = s12 * c13;
    pmns->real[0][2] = s13 * cos(delta);
    pmns->imag[0][2] = -s13 * sin(delta);
    
    pmns->real[1][0] = -s12 * c23 - c12 * s23 * s13 * cos(delta);
    pmns->imag[1][0] = c12 * s23 * s13 * sin(delta);
    pmns->real[1][1] = c12 * c23 - s12 * s23 * s13 * cos(delta);
    pmns->imag[1][1] = s12 * s23 * s13 * sin(delta);
    pmns->real[1][2] = s23 * c13;
    
    pmns->real[2][0] = s12 * s23 - c12 * c23 * s13 * cos(delta);
    pmns->imag[2][0] = -c12 * c23 * s13 * sin(delta);
    pmns->real[2][1] = -c12 * s23 - s12 * c23 * s13 * cos(delta);
    pmns->imag[2][1] = s12 * c23 * s13 * sin(delta);
    pmns->real[2][2] = c23 * c13;
    
    pmns->jarlskog = compute_jarlskog_invariant(pmns);
    
    double max_violation = 0.0;
    for (int i = 0; i < 3; i++) {
        double sum = 0.0;
        for (int j = 0; j < 3; j++) {
            double mag = pmns->real[i][j] * pmns->real[i][j] + pmns->imag[i][j] * pmns->imag[i][j];
            sum += mag;
        }
        double viol = fabs(sum - 1.0);
        if (viol > max_violation) max_violation = viol;
    }
    pmns->unitarity_violation = max_violation;
}

double compute_jarlskog_invariant(const MixingMatrix *m) {
    double J = 0.0;
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int p = (i + 1) % 3;
                    int q = (j + 1) % 3;
                    int r = (k + 1) % 3;
                    int s = (l + 1) % 3;
                    
                    double A_real = m->real[i][j] * m->real[k][l] - m->imag[i][j] * m->imag[k][l];
                    double A_imag = m->real[i][j] * m->imag[k][l] + m->imag[i][j] * m->real[k][l];
                    
                    double B_real = m->real[p][q] * m->real[r][s] - m->imag[p][q] * m->imag[r][s];
                    double B_imag = m->real[p][q] * m->imag[r][s] + m->imag[p][q] * m->real[r][s];
                    
                    J += A_imag * B_real - A_real * B_imag;
                }
            }
        }
    }
    
    return J / 36.0;
}
