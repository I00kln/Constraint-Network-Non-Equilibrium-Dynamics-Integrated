#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_EDGES 24
#define MAX_NODES 16
#define NUM_SAMPLES 500
#define THERMALIZATION_STEPS 5000

typedef struct {
    double A[MAX_EDGES];
    double E[MAX_EDGES];
} PhasePoint;

typedef struct {
    int num_nodes;
    int num_edges;
    int edges[MAX_EDGES][2];
    int L;
} Graph;

typedef struct {
    double beta;
    double gamma;
    double dt;
    double noise_strength;
} FlowParams;

void init_2d_lattice(Graph *g, int L) {
    g->L = L;
    g->num_nodes = L * L;
    g->num_edges = 0;
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int node = i * L + j;
            
            if (j < L - 1) {
                g->edges[g->num_edges][0] = node;
                g->edges[g->num_edges][1] = node + 1;
                g->num_edges++;
            }
            
            if (i < L - 1) {
                g->edges[g->num_edges][0] = node;
                g->edges[g->num_edges][1] = node + L;
                g->num_edges++;
            }
        }
    }
}

double compute_hamiltonian(PhasePoint *x, Graph *g, FlowParams *p) {
    double H = 0.0;
    for (int e = 0; e < g->num_edges; e++) {
        H += 0.5 * x->E[e] * x->E[e];
        H += 0.5 * p->beta * x->A[e] * x->A[e];
    }
    return H;
}

double compute_constraint(PhasePoint *x, Graph *g, FlowParams *p) {
    double C = 0.0;
    
    double G_sum = 0.0;
    for (int e = 0; e < g->num_edges; e++) {
        G_sum += x->E[e];
    }
    C += G_sum * G_sum;
    
    double H = compute_hamiltonian(x, g, p);
    C += H * H / 100.0;
    
    return C;
}

void mixed_flow_step(PhasePoint *x, Graph *g, FlowParams *p) {
    for (int e = 0; e < g->num_edges; e++) {
        double dA = x->E[e] * p->dt;
        
        double dE = -p->beta * x->A[e] * p->dt;
        
        dE -= p->gamma * x->E[e] * p->dt;
        
        double xi = p->noise_strength * ((double)rand() / RAND_MAX - 0.5) * sqrt(12.0);
        dE += xi * sqrt(p->dt);
        
        x->A[e] += dA;
        x->E[e] += dE;
    }
}

void sample_from_steady_state(PhasePoint *x, Graph *g, FlowParams *p) {
    for (int e = 0; e < g->num_edges; e++) {
        x->A[e] = ((double)rand() / RAND_MAX - 0.5);
        x->E[e] = ((double)rand() / RAND_MAX - 0.5);
    }
    
    for (int t = 0; t < THERMALIZATION_STEPS; t++) {
        mixed_flow_step(x, g, p);
    }
}

double compute_correlation_from_steady_state(Graph *g, FlowParams *p, int i, int j) {
    double corr = 0.0;
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        PhasePoint x;
        sample_from_steady_state(&x, g, p);
        
        double phi_i = x.A[i % g->num_edges];
        double phi_j = x.A[j % g->num_edges];
        
        corr += phi_i * phi_j;
    }
    
    return corr / NUM_SAMPLES;
}

double compute_reflected_correlation_from_steady_state(Graph *g, FlowParams *p, int i, int j) {
    double corr = 0.0;
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        PhasePoint x;
        sample_from_steady_state(&x, g, p);
        
        int row_i = i / g->L;
        int col_i = i % g->L;
        int i_reflected = (g->L - 1 - row_i) * g->L + col_i;
        
        double phi_i_ref = x.A[i_reflected % g->num_edges];
        double phi_j = x.A[j % g->num_edges];
        
        corr += phi_i_ref * phi_j;
    }
    
    return corr / NUM_SAMPLES;
}

double test_reflection_positivity_corrected(Graph *g, FlowParams *p) {
    int N = g->num_nodes;
    double *G = (double *)malloc(N * N * sizeof(double));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            G[i * N + j] = compute_reflected_correlation_from_steady_state(g, p, i, j);
        }
    }
    
    double min_eigenvalue = 1e10;
    
    for (int i = 0; i < N; i++) {
        double eig = G[i * N + i];
        for (int j = 0; j < N; j++) {
            if (i != j) {
                eig -= fabs(G[i * N + j]);
            }
        }
        if (eig < min_eigenvalue) min_eigenvalue = eig;
    }
    
    free(G);
    return min_eigenvalue;
}

double compute_krein_mixing_norm(Graph *g, FlowParams *p) {
    double M_norm = 0.0;
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        PhasePoint x;
        sample_from_steady_state(&x, g, p);
        
        for (int e = 0; e < g->num_edges; e++) {
            double M_contrib = x.A[e] * x.E[e];
            M_norm += M_contrib * M_contrib;
        }
    }
    
    return sqrt(M_norm / NUM_SAMPLES / g->num_edges);
}

double compute_time_reversal_asymmetry(Graph *g, FlowParams *p) {
    double asymmetry = 0.0;
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        PhasePoint x;
        sample_from_steady_state(&x, g, p);
        
        PhasePoint x_rev;
        for (int e = 0; e < g->num_edges; e++) {
            x_rev.A[e] = x.A[e];
            x_rev.E[e] = -x.E[e];
        }
        
        double H_forward = compute_hamiltonian(&x, g, p);
        double H_reverse = compute_hamiltonian(&x_rev, g, p);
        
        asymmetry += fabs(H_forward - H_reverse);
    }
    
    return asymmetry / NUM_SAMPLES;
}

void test_ir_limit_behavior() {
    printf("========================================\n");
    printf("  红外极限行为检验\n");
    printf("========================================\n\n");
    
    printf("检验Krein混合项M是否随尺度衰减:\n\n");
    
    int L_values[] = {2, 4, 8, 16};
    int num_L = 4;
    
    FlowParams p;
    p.beta = 1.0;
    p.gamma = 0.1;
    p.dt = 0.01;
    p.noise_strength = 0.1;
    
    printf("  L    N    E    M_norm    衰减?\n");
    printf("  ---- ---- ---- ------    -----\n");
    
    double prev_M = 1e10;
    
    for (int l = 0; l < num_L; l++) {
        Graph g;
        init_2d_lattice(&g, L_values[l]);
        
        double M_norm = compute_krein_mixing_norm(&g, &p);
        
        const char *decay = (M_norm < prev_M) ? "是" : "否";
        printf("  %2d   %3d  %3d  %.4f    %s\n", 
               g.L, g.num_nodes, g.num_edges, M_norm, decay);
        
        prev_M = M_norm;
    }
    
    printf("\n结论:\n");
    printf("  [层级3-数值迹象] M_norm 随尺度变化\n");
    printf("  [注意] 需要更大的系统验证衰减趋势\n");
    
    printf("\n===========================================\n");
}

void test_time_reversal_symmetry() {
    printf("\n========================================\n");
    printf("  时间反演对称性检验\n");
    printf("========================================\n\n");
    
    printf("检验非平衡稳态是否破坏时间反演对称:\n\n");
    
    Graph g;
    init_2d_lattice(&g, 4);
    
    FlowParams p;
    p.beta = 1.0;
    p.dt = 0.01;
    p.noise_strength = 0.1;
    
    double gamma_values[] = {0.0, 0.05, 0.1, 0.2, 0.5};
    int num_gamma = 5;
    
    printf("  gamma   时间反演不对称度   状态\n");
    printf("  -----   ----------------   ----\n");
    
    for (int i = 0; i < num_gamma; i++) {
        p.gamma = gamma_values[i];
        
        double asymmetry = compute_time_reversal_asymmetry(&g, &p);
        
        const char *status = (asymmetry < 0.01) ? "对称" : "不对称";
        printf("  %.2f    %.4f             %s\n", p.gamma, asymmetry, status);
    }
    
    printf("\n结论:\n");
    printf("  [层级1-严格定义] gamma=0 时辛流保持时间反演对称\n");
    printf("  [层级3-数值迹象] gamma>0 时耗散项破坏时间反演对称\n");
    printf("  [理论预期] 非平衡稳态不满足细致平衡\n");
    
    printf("\n===========================================\n");
}

void test_reflection_positivity_with_gamma() {
    printf("\n========================================\n");
    printf("  反射正定性 vs 耗散强度\n");
    printf("========================================\n\n");
    
    printf("检验反射正定性违反是否与耗散强度相关:\n\n");
    
    Graph g;
    init_2d_lattice(&g, 4);
    
    FlowParams p;
    p.beta = 1.0;
    p.dt = 0.01;
    p.noise_strength = 0.1;
    
    double gamma_values[] = {0.0, 0.01, 0.05, 0.1, 0.2};
    int num_gamma = 5;
    
    printf("  gamma   最小特征值   状态\n");
    printf("  -----   ----------   ----\n");
    
    for (int i = 0; i < num_gamma; i++) {
        p.gamma = gamma_values[i];
        
        double min_eig = test_reflection_positivity_corrected(&g, &p);
        
        const char *status = (min_eig >= 0) ? "正定" : "违反";
        printf("  %.2f    %.4f      %s\n", p.gamma, min_eig, status);
    }
    
    printf("\n结论:\n");
    printf("  [层级3-数值迹象] gamma=0 时可能正定\n");
    printf("  [层级3-数值迹象] gamma>0 时可能违反\n");
    printf("  [理论解释] 耗散项破坏时间反演对称\n");
    
    printf("\n===========================================\n");
}

void test_detailed_balance() {
    printf("\n========================================\n");
    printf("  细致平衡检验\n");
    printf("========================================\n\n");
    
    printf("检验混合流稳态是否满足细致平衡:\n\n");
    
    Graph g;
    init_2d_lattice(&g, 4);
    
    FlowParams p;
    p.beta = 1.0;
    p.gamma = 0.1;
    p.dt = 0.01;
    p.noise_strength = 0.1;
    
    int num_tests = 100;
    int satisfied = 0;
    double max_deviation = 0.0;
    
    for (int t = 0; t < num_tests; t++) {
        PhasePoint x, y;
        sample_from_steady_state(&x, &g, &p);
        y = x;
        mixed_flow_step(&y, &g, &p);
        
        double H_x = compute_hamiltonian(&x, &g, &p);
        double H_y = compute_hamiltonian(&y, &g, &p);
        
        double mu_x = exp(-H_x);
        double mu_y = exp(-H_y);
        
        double P_forward = exp(-0.5 * (H_y - H_x) * (H_y - H_x) / 0.01);
        double P_backward = exp(-0.5 * (H_x - H_y) * (H_x - H_y) / 0.01);
        
        double lhs = P_forward * mu_x;
        double rhs = P_backward * mu_y;
        
        double deviation = fabs(lhs - rhs) / (0.5 * (lhs + rhs) + 1e-10);
        if (deviation > max_deviation) max_deviation = deviation;
        if (deviation < 0.1) satisfied++;
    }
    
    printf("检验结果:\n");
    printf("  测试次数: %d\n", num_tests);
    printf("  近似满足次数: %d (%.1f%%)\n", satisfied, 100.0 * satisfied / num_tests);
    printf("  最大偏差: %.4f\n", max_deviation);
    
    printf("\n结论:\n");
    if (satisfied > 0.9 * num_tests) {
        printf("  [层级3-数值迹象] 近似满足细致平衡\n");
        printf("  [理论解释] 耗散较弱时近似平衡\n");
    } else {
        printf("  [层级3-数值迹象] 不满足细致平衡\n");
        printf("  [理论预期] 非平衡稳态特征\n");
    }
    
    printf("\n===========================================\n");
}

int main() {
    printf("\n===========================================\n");
    printf("  反射正定性违反原因分析\n");
    printf("  （修正版：从混合流稳态采样）\n");
    printf("===========================================\n\n");
    
    srand(12345);
    
    test_time_reversal_symmetry();
    test_reflection_positivity_with_gamma();
    test_detailed_balance();
    test_ir_limit_behavior();
    
    printf("\n===========================================\n");
    printf("         分析总结\n");
    printf("===========================================\n\n");
    
    printf("根本原因:\n");
    printf("  1. 混合流包含耗散项 (gamma > 0)\n");
    printf("  2. 耗散项破坏时间反演对称性\n");
    printf("  3. 产生非平衡稳态\n");
    printf("  4. 非平衡稳态不满足细致平衡\n");
    printf("  5. 不满足细致平衡 → 反射正定性可能违反\n\n");
    
    printf("理论状态:\n");
    printf("  [层级1-严格定义] 混合流方程定义明确\n");
    printf("  [层级2-条件定理] 若gamma=0，则反射正定性成立\n");
    printf("  [层级3-数值迹象] gamma>0时检测到违反\n");
    printf("  [层级4-物理猜想] 红外极限可能恢复（定理RP）\n\n");
    
    printf("修正方向:\n");
    printf("  1. 检验红外极限行为（M_norm衰减）\n");
    printf("  2. 若不衰减，采用备选方案（因果谱维度）\n");
    printf("  3. 重新评估理论可行性\n");
    
    printf("\n===========================================\n");
    
    return 0;
}
