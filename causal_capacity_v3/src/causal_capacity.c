#include "causal_capacity.h"

static double rand_uniform(unsigned int *seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (double)(*seed) / 0x7fffffff;
}

void init_causal_graph(CausalGraph *graph, int num_nodes) {
    memset(graph, 0, sizeof(CausalGraph));
    graph->num_nodes = num_nodes;
    graph->num_edges = 0;
    graph->has_cycle = 0;
    graph->cycle_count = 0;
}

int add_edge(CausalGraph *graph, int from, int to, double capacity) {
    if (graph->num_edges >= MAX_EDGES) return -1;
    if (from < 0 || from >= graph->num_nodes || to < 0 || to >= graph->num_nodes) return -1;
    if (graph->in_degree[to] >= MAX_NEIGHBORS || graph->out_degree[from] >= MAX_NEIGHBORS) return -1;

    int idx = graph->num_edges;
    graph->edges[idx].from = from;
    graph->edges[idx].to = to;
    graph->edges[idx].capacity = capacity;
    graph->edges[idx].weight = capacity;
    graph->edges[idx].flow = 0.0;

    int oi = graph->out_degree[from];
    graph->out_neighbors[from][oi] = to;
    graph->out_edge_idx[from][oi] = idx;
    graph->out_degree[from]++;

    int ii = graph->in_degree[to];
    graph->in_neighbors[to][ii] = from;
    graph->in_edge_idx[to][ii] = idx;
    graph->in_degree[to]++;

    graph->num_edges++;
    return idx;
}

int remove_edge(CausalGraph *graph, int edge_idx) {
    if (edge_idx < 0 || edge_idx >= graph->num_edges) return -1;

    int from = graph->edges[edge_idx].from;
    int to = graph->edges[edge_idx].to;

    for (int i = 0; i < graph->out_degree[from]; i++) {
        if (graph->out_edge_idx[from][i] == edge_idx) {
            for (int j = i; j < graph->out_degree[from] - 1; j++) {
                graph->out_neighbors[from][j] = graph->out_neighbors[from][j + 1];
                graph->out_edge_idx[from][j] = graph->out_edge_idx[from][j + 1];
            }
            graph->out_degree[from]--;
            break;
        }
    }

    for (int i = 0; i < graph->in_degree[to]; i++) {
        if (graph->in_edge_idx[to][i] == edge_idx) {
            for (int j = i; j < graph->in_degree[to] - 1; j++) {
                graph->in_neighbors[to][j] = graph->in_neighbors[to][j + 1];
                graph->in_edge_idx[to][j] = graph->in_edge_idx[to][j + 1];
            }
            graph->in_degree[to]--;
            break;
        }
    }

    int last = graph->num_edges - 1;
    if (edge_idx != last) {
        graph->edges[edge_idx] = graph->edges[last];
        int lf = graph->edges[last].from;
        int lt = graph->edges[last].to;
        for (int i = 0; i < graph->out_degree[lf]; i++) {
            if (graph->out_edge_idx[lf][i] == last) {
                graph->out_edge_idx[lf][i] = edge_idx;
                break;
            }
        }
        for (int i = 0; i < graph->in_degree[lt]; i++) {
            if (graph->in_edge_idx[lt][i] == last) {
                graph->in_edge_idx[lt][i] = edge_idx;
                break;
            }
        }
    }
    graph->num_edges--;
    return 0;
}

int find_edge(const CausalGraph *graph, int from, int to) {
    for (int i = 0; i < graph->out_degree[from]; i++) {
        if (graph->out_neighbors[from][i] == to) {
            return graph->out_edge_idx[from][i];
        }
    }
    return -1;
}

void generate_er_graph(CausalGraph *graph, double p, unsigned int *seed) {
    int n = graph->num_nodes;
    for (int u = 0; u < n; u++) {
        for (int v = 0; v < n; v++) {
            if (u != v && rand_uniform(seed) < p) {
                double cap = 0.1 + 0.9 * rand_uniform(seed);
                add_edge(graph, u, v, cap);
            }
        }
    }
}

void generate_directed_acyclic_graph(CausalGraph *graph, double p, unsigned int *seed) {
    int n = graph->num_nodes;
    for (int u = 0; u < n; u++) {
        for (int v = u + 1; v < n; v++) {
            if (rand_uniform(seed) < p) {
                double cap = 0.1 + 0.9 * rand_uniform(seed);
                add_edge(graph, u, v, cap);
            }
        }
    }
}

void generate_small_world_graph(CausalGraph *graph, int k, double rewire_prob, unsigned int *seed) {
    int n = graph->num_nodes;

    for (int u = 0; u < n; u++) {
        for (int di = 1; di <= k; di++) {
            int v = (u + di) % n;
            double cap = 0.3 + 0.7 * rand_uniform(seed);
            add_edge(graph, u, v, cap);

            int v2 = (u - di + n) % n;
            if (v2 != u && v2 != v) {
                double cap2 = 0.3 + 0.7 * rand_uniform(seed);
                add_edge(graph, v2, u, cap2);
            }
        }
    }

    int edges_to_rewire = (int)(graph->num_edges * rewire_prob);
    for (int r = 0; r < edges_to_rewire; r++) {
        int eidx = (int)(rand_uniform(seed) * graph->num_edges);
        if (eidx >= graph->num_edges) eidx = graph->num_edges - 1;

        int from = graph->edges[eidx].from;
        int new_to = (int)(rand_uniform(seed) * n);
        if (new_to >= n) new_to = n - 1;

        if (new_to != from && find_edge(graph, from, new_to) < 0) {
            double cap = graph->edges[eidx].capacity;
            remove_edge(graph, eidx);
            add_edge(graph, from, new_to, cap);
        }
    }
}

int detect_cycles(CausalGraph *graph) {
    int n = graph->num_nodes;
    static char color[MAX_NODES];
    memset(color, 0, n);

    graph->cycle_count = 0;
    graph->has_cycle = 0;

    static int stack[MAX_NODES];
    static char on_stack[MAX_NODES];
    int sp = 0;
    memset(on_stack, 0, n);

    for (int start = 0; start < n; start++) {
        if (color[start] != 0) continue;

        sp = 0;
        stack[sp++] = start;
        color[start] = 1;
        on_stack[start] = 1;

        while (sp > 0) {
            int v = stack[sp - 1];
            int found_unvisited = 0;

            for (int i = 0; i < graph->out_degree[v]; i++) {
                int w = graph->out_neighbors[v][i];
                if (color[w] == 0) {
                    color[w] = 1;
                    on_stack[w] = 1;
                    stack[sp++] = w;
                    found_unvisited = 1;
                    break;
                } else if (on_stack[w]) {
                    graph->cycle_count++;
                    graph->has_cycle = 1;
                }
            }

            if (!found_unvisited) {
                on_stack[v] = 0;
                color[v] = 2;
                sp--;
            }
        }
    }

    return graph->cycle_count;
}

void init_info_state(InfoState *state, int num_nodes) {
    memset(state, 0, sizeof(InfoState));
    state->num_nodes = num_nodes;
}

void set_uniform_density(InfoState *state, double total) {
    double val = total / state->num_nodes;
    for (int i = 0; i < state->num_nodes; i++) {
        state->rho[i] = val;
    }
    state->total_info = total;
}

double estimate_causal_capacity(CausalGraph *graph, InfoState *state, int from, int to, double delta, int samples, unsigned int *seed) {
    (void)seed;
    (void)samples;
    double total_response = 0.0;
    int valid = 0;

    int eidx = find_edge(graph, from, to);
    if (eidx < 0) return 0.0;

    double original_from = state->rho[from];
    double original_to = state->rho[to];

    state->rho[from] += delta;

    double sum_cap = 0.0;
    double weighted_sum = 0.0;
    for (int i = 0; i < graph->in_degree[to]; i++) {
        int parent = graph->in_neighbors[to][i];
        int e = graph->in_edge_idx[to][i];
        double cap = graph->edges[e].capacity;
        sum_cap += cap;
        weighted_sum += cap * state->rho[parent];
    }

    double new_rho_to;
    if (sum_cap > 1e-15) {
        new_rho_to = weighted_sum;
    } else {
        new_rho_to = original_to;
    }

    double response = new_rho_to - original_to;
    if (fabs(delta) > 1e-15) {
        total_response = response / delta;
        valid = 1;
    }

    state->rho[from] = original_from;
    state->rho[to] = original_to;

    return (valid > 0) ? total_response : 0.0;
}

void step1_causal_propagation(CausalGraph *graph, InfoState *state, double eta) {
    static double new_rho[MAX_NODES];
    int n = state->num_nodes;

    for (int v = 0; v < n; v++) {
        double sum_cap = 0.0;
        double weighted_sum = 0.0;

        for (int i = 0; i < graph->in_degree[v]; i++) {
            int parent = graph->in_neighbors[v][i];
            int eidx = graph->in_edge_idx[v][i];
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
        double deficit = state->total_info - total;
        double correction = deficit / n;
        for (int v = 0; v < n; v++) {
            new_rho[v] += correction;
            if (new_rho[v] < 1e-15) new_rho[v] = 1e-15;
        }
    }

    total = 0.0;
    for (int v = 0; v < n; v++) total += new_rho[v];
    if (total > 1e-15) {
        double scale = state->total_info / total;
        for (int v = 0; v < n; v++) {
            new_rho[v] *= scale;
            if (new_rho[v] < 1e-15) new_rho[v] = 1e-15;
        }
    }

    for (int e = 0; e < graph->num_edges; e++) {
        int u = graph->edges[e].from;
        double cap = graph->edges[e].capacity;
        graph->edges[e].flow = cap * state->rho[u];
    }

    memcpy(state->rho, new_rho, n * sizeof(double));
}

void step2_macro_micro_decomposition(CausalGraph *graph, InfoState *state, double lambda_param) {
    (void)lambda_param;
    int n = state->num_nodes;

    for (int v = 0; v < n; v++) {
        int num_neighbors = graph->in_degree[v] + graph->out_degree[v];
        double local_avg = 0.0;

        for (int i = 0; i < graph->in_degree[v]; i++) {
            local_avg += state->rho[graph->in_neighbors[v][i]];
        }
        for (int i = 0; i < graph->out_degree[v]; i++) {
            local_avg += state->rho[graph->out_neighbors[v][i]];
        }

        if (num_neighbors > 0) local_avg /= num_neighbors;
        else local_avg = state->rho[v];

        double deviation = state->rho[v] - local_avg;
        double rho_safe = (state->rho[v] > 1e-15) ? state->rho[v] : 1e-15;
        double sigmoid = 1.0 / (1.0 + exp(-2.0 * deviation / rho_safe));

        state->rho_macro[v] = sigmoid * state->rho[v];
        state->rho_micro[v] = state->rho[v] - state->rho_macro[v];
    }
}

void step3_redistribution(InfoState *state, double lambda_param, double target_total) {
    double M = 0.0, m = 0.0;
    int n = state->num_nodes;

    for (int v = 0; v < n; v++) {
        M += state->rho_macro[v];
        m += state->rho_micro[v];
    }

    double target_micro = target_total - M;
    if (fabs(m) > 1e-10) {
        double scale = target_micro / (lambda_param * m);
        for (int v = 0; v < n; v++) {
            state->rho_micro[v] *= scale;
        }
    }

    for (int v = 0; v < n; v++) {
        state->rho[v] = state->rho_macro[v] + lambda_param * state->rho_micro[v];
        if (state->rho[v] < 1e-15) state->rho[v] = 1e-15;
    }

    state->total_info = 0.0;
    for (int v = 0; v < n; v++) state->total_info += state->rho[v];
}

void step4_topology_rewiring(CausalGraph *graph, InfoState *state, double alpha, double beta, double epsilon_del, double sigma, int max_new_edges, unsigned int *seed) {
    int edges_to_check = graph->num_edges;
    int del_indices[MAX_EDGES];
    int num_del = 0;

    double avg_cap = 0.0;
    for (int e = 0; e < graph->num_edges; e++) {
        avg_cap += graph->edges[e].capacity;
    }
    if (graph->num_edges > 0) avg_cap /= graph->num_edges;
    else avg_cap = 0.5;

    for (int e = 0; e < edges_to_check && num_del < graph->num_edges / 10; e++) {
        if (graph->edges[e].capacity < epsilon_del * avg_cap) {
            del_indices[num_del++] = e;
        }
    }

    for (int i = num_del - 1; i >= 0; i--) {
        remove_edge(graph, del_indices[i]);
    }

    for (int e = 0; e < graph->num_edges; e++) {
        graph->edges[e].capacity += sigma * (rand_uniform(seed) - 0.5) * 2.0;
        if (graph->edges[e].capacity < 0.01) graph->edges[e].capacity = 0.01;
        if (graph->edges[e].capacity > 2.0) graph->edges[e].capacity = 2.0;
    }

    double max_rho = 0.0;
    for (int v = 0; v < state->num_nodes; v++) {
        if (state->rho[v] > max_rho) max_rho = state->rho[v];
    }
    if (max_rho < 1e-15) max_rho = 1.0;

    for (int attempt = 0; attempt < max_new_edges; attempt++) {
        int u = (int)(rand_uniform(seed) * graph->num_nodes);
        int v = (int)(rand_uniform(seed) * graph->num_nodes);
        if (u >= graph->num_nodes) u = graph->num_nodes - 1;
        if (v >= graph->num_nodes) v = graph->num_nodes - 1;

        if (u != v && find_edge(graph, u, v) < 0) {
            double prob = alpha * (state->rho[u] / max_rho) * (state->rho[v] / max_rho)
                        + beta * fabs(0.5 - avg_cap);

            int creates_cycle = 0;
            if (find_edge(graph, v, u) >= 0) {
                creates_cycle = 1;
                prob += 0.5;
            }

            if (!creates_cycle) {
                for (int i = 0; i < graph->out_degree[v] && !creates_cycle; i++) {
                    int w = graph->out_neighbors[v][i];
                    if (find_edge(graph, w, u) >= 0) creates_cycle = 1;
                }
            }

            if (creates_cycle) {
                prob += 0.3;
            }

            prob = 1.0 / (1.0 + exp(-prob));

            if (rand_uniform(seed) < prob) {
                double cap = 0.1 + 0.9 * rand_uniform(seed);
                add_edge(graph, u, v, cap);
            }
        }
    }
}

void step5_capacity_evolution(CausalGraph *graph, InfoState *state, double learning_rate) {
    int n = state->num_nodes;
    double avg_rho = state->total_info / n;
    if (avg_rho < 1e-15) avg_rho = 1e-15;

    for (int e = 0; e < graph->num_edges; e++) {
        int u = graph->edges[e].from;
        int v = graph->edges[e].to;

        double flow_u = state->rho[u] / avg_rho;
        double flow_v = state->rho[v] / avg_rho;
        double flow_signal = flow_u * flow_v - 1.0;

        double in_out_asymmetry = 0.0;
        if (graph->in_degree[v] > 0 && graph->out_degree[u] > 0) {
            double in_total = 0.0, out_total = 0.0;
            for (int i = 0; i < graph->in_degree[v]; i++) {
                int eidx = graph->in_edge_idx[v][i];
                in_total += graph->edges[eidx].capacity;
            }
            for (int i = 0; i < graph->out_degree[u]; i++) {
                int eidx = graph->out_edge_idx[u][i];
                out_total += graph->edges[eidx].capacity;
            }
            in_out_asymmetry = (in_total - out_total) / (in_total + out_total + 1e-10);
        }

        double feedback = 0.0;
        if (graph->out_degree[v] > 0) {
            double max_child_cap = 0.0;
            for (int i = 0; i < graph->out_degree[v]; i++) {
                int eidx = graph->out_edge_idx[v][i];
                if (graph->edges[eidx].capacity > max_child_cap) {
                    max_child_cap = graph->edges[eidx].capacity;
                }
            }
            feedback = 0.2 * (max_child_cap - graph->edges[e].capacity);
        }

        double delta_cap = learning_rate * (flow_signal + 0.3 * in_out_asymmetry + feedback);
        graph->edges[e].capacity += delta_cap;

        if (graph->edges[e].capacity < 0.01) graph->edges[e].capacity = 0.01;
        if (graph->edges[e].capacity > 3.0) graph->edges[e].capacity = 3.0;
    }
}

void dynamics_full_step(CausalGraph *graph, InfoState *state, const DynamicsParams *params, unsigned int *seed) {
    step1_causal_propagation(graph, state, params->eta);
    step2_macro_micro_decomposition(graph, state, params->lambda_param);
    step3_redistribution(state, params->lambda_param, 1.0);
    step4_topology_rewiring(graph, state, params->alpha, params->beta,
                           params->epsilon_del, params->sigma, params->max_new_edges, seed);
    if (params->cap_learning_rate > 0) {
        step5_capacity_evolution(graph, state, params->cap_learning_rate);
    }
}

void compute_intervention_response(CausalGraph *graph, InfoState *state, double eta, double delta, double *response_matrix) {
    int n = graph->num_nodes;
    static double orig_rho[MAX_NODES];
    static double perturbed_rho[MAX_NODES];

    memcpy(orig_rho, state->rho, n * sizeof(double));

    for (int u = 0; u < n; u++) {
        memcpy(perturbed_rho, orig_rho, n * sizeof(double));
        perturbed_rho[u] += delta;

        static double new_rho[MAX_NODES];
        for (int v = 0; v < n; v++) {
            double sum_cap = 0.0;
            double weighted_sum = 0.0;

            for (int i = 0; i < graph->in_degree[v]; i++) {
                int parent = graph->in_neighbors[v][i];
                int eidx = graph->in_edge_idx[v][i];
                double cap = graph->edges[eidx].capacity;
                sum_cap += cap;
                weighted_sum += cap * perturbed_rho[parent];
            }

            if (sum_cap > 1e-15) {
                new_rho[v] = (1.0 - eta) * perturbed_rho[v] + eta * weighted_sum;
            } else {
                new_rho[v] = perturbed_rho[v];
            }
        }

        double total = 0.0;
        for (int v = 0; v < n; v++) total += new_rho[v];
        if (total > 1e-15) {
            double scale = state->total_info / total;
            for (int v = 0; v < n; v++) new_rho[v] *= scale;
        }

        for (int v = 0; v < n; v++) {
            response_matrix[v * n + u] = (new_rho[v] - orig_rho[v]) / delta;
        }
    }
}
