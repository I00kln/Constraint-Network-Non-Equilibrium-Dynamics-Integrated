#ifndef CONSTRAINT_NETWORK_H
#define CONSTRAINT_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_NODES 256
#define MAX_EDGES 512
#define MAX_FACES 512
#define PI 3.14159265358979323846
#define TWO_PI (2.0 * PI)

typedef struct {
    int i;
    int j;
} Edge;

typedef struct {
    int edges[3];
    int nodes[3];
} Face;

typedef struct {
    int num_nodes;
    int num_edges;
    int num_faces;
    Edge edges[MAX_EDGES];
    Face faces[MAX_FACES];
    int node_edges[MAX_NODES][MAX_EDGES];
    int node_edge_count[MAX_NODES];
} Graph;

typedef struct {
    double A[MAX_EDGES];
    double E[MAX_EDGES];
    double q[MAX_EDGES];
    double p[MAX_EDGES];
} PhaseSpace;

typedef struct {
    double G[MAX_NODES];
    double S[MAX_EDGES];
    double H_total;
    double H_curv;
    double H_kin;
    double H_qp;
    double tau[MAX_NODES];
    double Ric[MAX_FACES];
} Constraints;

typedef struct {
    double beta;
    double hbar_eff;
    double gamma;
    double dt;
    int num_steps;
    int output_interval;
} SimulationParams;

typedef struct {
    double time;
    double gauss_violation;
    double diffusion_violation;
    double energy;
    double energy_drift;
    double max_gauss;
    double max_diffusion;
    double causal_depth_variance;
} SimulationState;

#endif
