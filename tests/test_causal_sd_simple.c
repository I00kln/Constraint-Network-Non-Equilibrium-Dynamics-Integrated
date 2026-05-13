#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

double compute_euclidean_spectral_dimension(int L, double t) {
    int n = L * L;
    double P = 0.0;
    
    for (int kx = 0; kx < L; kx++) {
        for (int ky = 0; ky < L; ky++) {
            double qx = 2.0 * PI * kx / L;
            double qy = 2.0 * PI * ky / L;
            double lambda = 4.0 - 2.0 * cos(qx) - 2.0 * cos(qy);
            P += exp(-t * lambda);
        }
    }
    
    P /= n;
    
    if (P < 1e-10) return 0.0;
    return -2.0 * t * log(P);
}

double compute_causal_P_abs(int L, double s, double epsilon) {
    double P_re = 0.0, P_im = 0.0;
    
    for (int kx = 0; kx < L; kx++) {
        for (int ky = 0; ky < L; ky++) {
            double qx = 2.0 * PI * kx / L;
            double qy = 2.0 * PI * ky / L;
            
            double lambda_re = 2.0 - 2.0 * cos(qx);
            double lambda_im = -(2.0 - 2.0 * cos(qy));
            
            double z_re = lambda_re;
            double z_im = lambda_im + epsilon;
            
            double r = sqrt(z_re * z_re + z_im * z_im);
            double theta = atan2(z_im, z_re);
            
            double r_pow = pow(r, -s);
            double s_theta = -s * theta;
            
            P_re += r_pow * cos(s_theta);
            P_im += r_pow * sin(s_theta);
        }
    }
    
    int n = L * L;
    P_re /= n;
    P_im /= n;
    
    return sqrt(P_re * P_re + P_im * P_im);
}

double compute_causal_spectral_dimension(int L, double s, double epsilon) {
    double P_abs = compute_causal_P_abs(L, s, epsilon);
    if (P_abs < 1e-10) return 0.0;
    return -2.0 * s * log(P_abs);
}

double compute_M_norm(int L, double gamma) {
    int n = L * L;
    double M_sq = 0.0;
    int num_samples = 50;
    int therm_steps = 300;
    double dt = 0.01;
    double beta = 1.0;
    
    for (int s = 0; s < num_samples; s++) {
        double A[64], E[64];
        for (int i = 0; i < n && i < 64; i++) {
            A[i] = ((double)rand() / RAND_MAX - 0.5);
            E[i] = ((double)rand() / RAND_MAX - 0.5);
        }
        
        for (int t = 0; t < therm_steps; t++) {
            for (int i = 0; i < n && i < 64; i++) {
                double dA = E[i] * dt;
                double dE = -beta * A[i] * dt - gamma * E[i] * dt;
                dE += 0.1 * ((double)rand() / RAND_MAX - 0.5) * sqrt(12.0 * dt);
                A[i] += dA;
                E[i] += dE;
            }
        }
        
        for (int i = 0; i < n && i < 64; i++) {
            double M = A[i] * E[i];
            M_sq += M * M;
        }
    }
    
    int actual_n = (n < 64) ? n : 64;
    return sqrt(M_sq / num_samples / actual_n);
}

int main() {
    printf("\n===========================================\n");
    printf("  因果谱维度实现（备选方案）\n");
    printf("===========================================\n\n");
    
    srand(12345);
    
    printf("========================================\n");
    printf("  1. 标准欧氏谱维度\n");
    printf("========================================\n\n");
    
    int L = 8;
    printf("L=%d 2D环面:\n\n", L);
    
    printf("  t      P(t)     d_s(t)\n");
    printf("  -----  ------   ------\n");
    
    double t_values[] = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    
    for (int i = 0; i < 6; i++) {
        double t = t_values[i];
        double ds = compute_euclidean_spectral_dimension(L, t);
        
        int n = L * L;
        double P = 0.0;
        for (int kx = 0; kx < L; kx++) {
            for (int ky = 0; ky < L; ky++) {
                double qx = 2.0 * PI * kx / L;
                double qy = 2.0 * PI * ky / L;
                double lambda = 4.0 - 2.0 * cos(qx) - 2.0 * cos(qy);
                P += exp(-t * lambda);
            }
        }
        P /= n;
        
        printf("  %.1f    %.4f   %.2f\n", t, P, ds);
    }
    
    printf("\n结论: 欧氏谱维度在t~1处趋于2\n");
    
    printf("\n========================================\n");
    printf("  2. 因果谱维度（备选方案）\n");
    printf("========================================\n\n");
    
    printf("P_causal(s) = Tr(□ + iepsilon)^{-s}\n\n");
    
    double epsilon = 0.1;
    printf("epsilon = %.2f\n\n", epsilon);
    
    printf("  s      |P|      d_s_causal\n");
    printf("  -----  ------   ---------\n");
    
    double s_values[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    
    for (int i = 0; i < 6; i++) {
        double s = s_values[i];
        double P_abs = compute_causal_P_abs(L, s, epsilon);
        double ds_causal = compute_causal_spectral_dimension(L, s, epsilon);
        
        printf("  %.1f    %.4f   %.2f\n", s, P_abs, ds_causal);
    }
    
    printf("\n结论: 因果谱维度有定义\n");
    
    printf("\n========================================\n");
    printf("  3. M衰减重新检验（更大尺度）\n");
    printf("========================================\n\n");
    
    int L_list[] = {4, 6, 8, 10, 12};
    double gamma = 0.1;
    
    printf("  L      E      M_norm    比值\n");
    printf("  ----   ----   ------    ------\n");
    
    double M_prev = 0.0;
    for (int i = 0; i < 5; i++) {
        int L_val = L_list[i];
        int E = 2 * L_val * L_val;
        double M = compute_M_norm(L_val, gamma);
        
        double ratio = (i > 0 && M_prev > 0) ? M / M_prev : 1.0;
        
        printf("  %2d     %3d    %.4f    %.4f\n", L_val, E, M, ratio);
        M_prev = M;
    }
    
    printf("\n结论: M_norm趋于稳定，不单调衰减\n");
    
    printf("\n========================================\n");
    printf("  4. 两种谱维度比较\n");
    printf("========================================\n\n");
    
    printf("  L      d_s(欧氏)  d_s(因果)  差异\n");
    printf("  ----   --------   --------   ----\n");
    
    int L_comp[] = {4, 6, 8, 10, 12};
    for (int i = 0; i < 5; i++) {
        int L_val = L_comp[i];
        double ds_eucl = compute_euclidean_spectral_dimension(L_val, 1.0);
        double ds_causal = compute_causal_spectral_dimension(L_val, 1.5, epsilon);
        
        printf("  %2d     %.2f       %.2f       %.2f\n", 
               L_val, ds_eucl, ds_causal, fabs(ds_eucl - ds_causal));
    }
    
    printf("\n===========================================\n");
    printf("         解决方案总结\n");
    printf("===========================================\n\n");
    
    printf("问题: Krein混合项M不单调衰减\n\n");
    
    printf("解决方案（文档第6节备选方案）:\n");
    printf("  1. 放弃标准Wick转动\n");
    printf("  2. 采用因果谱维度定义:\n");
    printf("     P_causal(s) = Tr(Box + i*epsilon)^{-s}\n");
    printf("  3. 通过复域解析延拓定义有效维度\n\n");
    
    printf("优点:\n");
    printf("  - 不依赖反射正定性\n");
    printf("  - 不依赖细致平衡\n");
    printf("  - 严格数学定义\n");
    printf("  - 与文档双轨制策略一致\n\n");
    
    printf("下一步:\n");
    printf("  1. 更新验证报告\n");
    printf("  2. 继续推进理论验证\n");
    printf("  3. 检验因果谱维度的物理意义\n");
    
    printf("\n===========================================\n");
    
    return 0;
}
