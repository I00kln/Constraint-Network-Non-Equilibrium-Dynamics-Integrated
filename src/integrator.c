#include "integrator.h"
#include "phase_space.h"
#include <string.h>

static double alpha_param = 1.0;

void set_alpha(double alpha) {
    alpha_param = alpha;
}

void normalize_angles(PhaseSpace *ps, int num_edges) {
    for (int e = 0; e < num_edges; e++) {
        while (ps->A[e] > PI) ps->A[e] -= TWO_PI;
        while (ps->A[e] < -PI) ps->A[e] += TWO_PI;
    }
}

void enforce_q_positive(PhaseSpace *ps, int num_edges) {
    for (int e = 0; e < num_edges; e++) {
        if (ps->q[e] < 0.01) ps->q[e] = 0.01;
    }
}

void project_diffusion_constraint(PhaseSpace *ps, int num_edges, double hbar_eff) {
    for (int e = 0; e < num_edges; e++) {
        ps->p[e] = hbar_eff / ps->q[e];
    }
}

void verlet_step(Graph *graph, PhaseSpace *ps, double dt, double beta) {
    static double dH_dA[MAX_EDGES], dH_dE[MAX_EDGES];
    static double dH_dq[MAX_EDGES], dH_dp[MAX_EDGES];
    
    compute_hamiltonian_derivatives(graph, ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->E[e] -= 0.5 * dt * dH_dA[e];
        ps->p[e] -= 0.5 * dt * dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->A[e] += dt * dH_dE[e];
        ps->q[e] += dt * dH_dp[e];
    }
    
    compute_hamiltonian_derivatives(graph, ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->E[e] -= 0.5 * dt * dH_dA[e];
        ps->p[e] -= 0.5 * dt * dH_dq[e];
    }
    
    enforce_q_positive(ps, graph->num_edges);
}

void symplectic_rk4_step_with_projection(Graph *graph, PhaseSpace *ps, double dt, 
                                          double beta, double hbar_eff) {
    symplectic_rk4_step(graph, ps, dt, beta);
    project_diffusion_constraint(ps, graph->num_edges, hbar_eff);
}

void symplectic_rk4_step(Graph *graph, PhaseSpace *ps, double dt, double beta) {
    static PhaseSpace temp_ps;
    static double k1_A[MAX_EDGES], k1_E[MAX_EDGES], k1_q[MAX_EDGES], k1_p[MAX_EDGES];
    static double k2_A[MAX_EDGES], k2_E[MAX_EDGES], k2_q[MAX_EDGES], k2_p[MAX_EDGES];
    static double k3_A[MAX_EDGES], k3_E[MAX_EDGES], k3_q[MAX_EDGES], k3_p[MAX_EDGES];
    static double k4_A[MAX_EDGES], k4_E[MAX_EDGES], k4_q[MAX_EDGES], k4_p[MAX_EDGES];
    static double dH_dA[MAX_EDGES], dH_dE[MAX_EDGES], dH_dq[MAX_EDGES], dH_dp[MAX_EDGES];
    
    memcpy(&temp_ps, ps, sizeof(PhaseSpace));
    compute_hamiltonian_derivatives(graph, &temp_ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        k1_A[e] = dH_dE[e];
        k1_E[e] = -dH_dA[e];
        k1_q[e] = dH_dp[e];
        k1_p[e] = -dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        temp_ps.A[e] = ps->A[e] + 0.5 * dt * k1_A[e];
        temp_ps.E[e] = ps->E[e] + 0.5 * dt * k1_E[e];
        temp_ps.q[e] = ps->q[e] + 0.5 * dt * k1_q[e];
        temp_ps.p[e] = ps->p[e] + 0.5 * dt * k1_p[e];
    }
    compute_hamiltonian_derivatives(graph, &temp_ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        k2_A[e] = dH_dE[e];
        k2_E[e] = -dH_dA[e];
        k2_q[e] = dH_dp[e];
        k2_p[e] = -dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        temp_ps.A[e] = ps->A[e] + 0.5 * dt * k2_A[e];
        temp_ps.E[e] = ps->E[e] + 0.5 * dt * k2_E[e];
        temp_ps.q[e] = ps->q[e] + 0.5 * dt * k2_q[e];
        temp_ps.p[e] = ps->p[e] + 0.5 * dt * k2_p[e];
    }
    compute_hamiltonian_derivatives(graph, &temp_ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        k3_A[e] = dH_dE[e];
        k3_E[e] = -dH_dA[e];
        k3_q[e] = dH_dp[e];
        k3_p[e] = -dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        temp_ps.A[e] = ps->A[e] + dt * k3_A[e];
        temp_ps.E[e] = ps->E[e] + dt * k3_E[e];
        temp_ps.q[e] = ps->q[e] + dt * k3_q[e];
        temp_ps.p[e] = ps->p[e] + dt * k3_p[e];
    }
    compute_hamiltonian_derivatives(graph, &temp_ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        k4_A[e] = dH_dE[e];
        k4_E[e] = -dH_dA[e];
        k4_q[e] = dH_dp[e];
        k4_p[e] = -dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->A[e] += dt * (k1_A[e] + 2.0*k2_A[e] + 2.0*k3_A[e] + k4_A[e]) / 6.0;
        ps->E[e] += dt * (k1_E[e] + 2.0*k2_E[e] + 2.0*k3_E[e] + k4_E[e]) / 6.0;
        ps->q[e] += dt * (k1_q[e] + 2.0*k2_q[e] + 2.0*k3_q[e] + k4_q[e]) / 6.0;
        ps->p[e] += dt * (k1_p[e] + 2.0*k2_p[e] + 2.0*k3_p[e] + k4_p[e]) / 6.0;
    }
    
    enforce_q_positive(ps, graph->num_edges);
}

void forest_ruth_step(Graph *graph, PhaseSpace *ps, double dt, double beta) {
    static double dH_dA[MAX_EDGES], dH_dE[MAX_EDGES], dH_dq[MAX_EDGES], dH_dp[MAX_EDGES];
    
    const double theta = 1.0 / (2.0 - pow(2.0, 1.0/3.0));
    const double lambda = 1.0 - 2.0 * theta;
    
    compute_hamiltonian_derivatives(graph, ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        ps->E[e] -= 0.5 * theta * dt * dH_dA[e];
        ps->p[e] -= 0.5 * theta * dt * dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->A[e] += theta * dt * dH_dE[e];
        ps->q[e] += theta * dt * dH_dp[e];
    }
    
    compute_hamiltonian_derivatives(graph, ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        ps->E[e] -= 0.5 * lambda * dt * dH_dA[e];
        ps->p[e] -= 0.5 * lambda * dt * dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->A[e] += lambda * dt * dH_dE[e];
        ps->q[e] += lambda * dt * dH_dp[e];
    }
    
    compute_hamiltonian_derivatives(graph, ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        ps->E[e] -= 0.5 * theta * dt * dH_dA[e];
        ps->p[e] -= 0.5 * theta * dt * dH_dq[e];
    }
    
    for (int e = 0; e < graph->num_edges; e++) {
        ps->A[e] += theta * dt * dH_dE[e];
        ps->q[e] += theta * dt * dH_dp[e];
    }
    
    compute_hamiltonian_derivatives(graph, ps, dH_dA, dH_dE, dH_dq, dH_dp, beta, alpha_param);
    for (int e = 0; e < graph->num_edges; e++) {
        ps->E[e] -= 0.5 * theta * dt * dH_dA[e];
        ps->p[e] -= 0.5 * theta * dt * dH_dq[e];
    }
    
    enforce_q_positive(ps, graph->num_edges);
}
