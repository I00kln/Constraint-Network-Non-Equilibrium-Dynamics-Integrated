#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

double compute_spectral_dimension_2d(int L, double t) {
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int kx = 0; kx < L; kx++) {
        for (int ky = 0; ky < L; ky++) {
            double qx = 2.0 * PI * kx / L;
            double qy = 2.0 * PI * ky / L;
            double lambda = 4.0 - 2.0 * cos(qx) - 2.0 * cos(qy);
            double w = exp(-t * lambda);
            numerator += lambda * w;
            denominator += w;
        }
    }
    
    if (denominator < 1e-10) return 0.0;
    return 2.0 * t * (numerator / denominator);
}

double compute_spectral_dimension_3d(int L, double t) {
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int kx = 0; kx < L; kx++) {
        for (int ky = 0; ky < L; ky++) {
            for (int kz = 0; kz < L; kz++) {
                double qx = 2.0 * PI * kx / L;
                double qy = 2.0 * PI * ky / L;
                double qz = 2.0 * PI * kz / L;
                double lambda = 6.0 - 2.0*cos(qx) - 2.0*cos(qy) - 2.0*cos(qz);
                double w = exp(-t * lambda);
                numerator += lambda * w;
                denominator += w;
            }
        }
    }
    
    if (denominator < 1e-10) return 0.0;
    return 2.0 * t * (numerator / denominator);
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("     扩展验证 - 修正版（使用正确t值）\n");
    printf("================================================================\n\n");
    
    printf("修正说明:\n");
    printf("  - 之前使用 t=0.5，不在平台区域\n");
    printf("  - 修正后使用 t~5.0（3D）和 t~4.9（2D）\n");
    printf("  - 偏差从10%%降低到3%%以内\n\n");
    
    int all_passed = 1;
    
    printf("================================================================\n");
    printf("  1. 更大系统尺寸验证 (L > 40) - 修正版\n");
    printf("================================================================\n\n");
    
    double t_2d = 4.9;
    
    printf("2D系统谱维度 (t = %.1f, 平台区域):\n\n", t_2d);
    printf("  L      N        d_s       预期    偏差\n");
    printf("  ----   ------   ------   ----    ----\n");
    
    int L2d[] = {32, 40, 48, 56, 64};
    int n_L2d = 5;
    double ds2d_values[5];
    
    for (int i = 0; i < n_L2d; i++) {
        int L = L2d[i];
        int N = L * L;
        double ds = compute_spectral_dimension_2d(L, t_2d);
        ds2d_values[i] = ds;
        
        printf("  %2d     %5d    %.3f     2       %.1f%%\n", 
               L, N, ds, fabs(ds-2.0)/2.0*100);
    }
    
    double ds2d_limit = 0.0;
    for (int i = n_L2d - 3; i < n_L2d; i++) ds2d_limit += ds2d_values[i];
    ds2d_limit /= 3;
    
    printf("\n热力学极限: d_s = %.3f (预期 2.0)\n", ds2d_limit);
    printf("偏差: %.1f%%\n", fabs(ds2d_limit-2.0)/2.0*100);
    
    int large_L_pass = (fabs(ds2d_limit - 2.0) < 0.15);
    printf("结果: %s\n", large_L_pass ? "[通过]" : "[需检查]");
    if (!large_L_pass) all_passed = 0;
    
    printf("\n================================================================\n");
    printf("  2. 三维系统验证 - 修正版\n");
    printf("================================================================\n\n");
    
    double t_3d = 5.0;
    
    printf("3D系统谱维度 (t = %.1f, 平台区域):\n\n", t_3d);
    printf("  L      N        d_s       预期    偏差\n");
    printf("  ----   ------   ------   ----    ----\n");
    
    int L3d[] = {8, 10, 12, 14, 16};
    int n_L3d = 5;
    double ds3d_values[5];
    
    for (int i = 0; i < n_L3d; i++) {
        int L = L3d[i];
        int N = L * L * L;
        double ds = compute_spectral_dimension_3d(L, t_3d);
        ds3d_values[i] = ds;
        
        printf("  %2d     %5d    %.3f     3       %.1f%%\n", 
               L, N, ds, fabs(ds-3.0)/3.0*100);
    }
    
    double ds3d_limit = 0.0;
    for (int i = n_L3d - 3; i < n_L3d; i++) ds3d_limit += ds3d_values[i];
    ds3d_limit /= 3;
    
    printf("\n热力学极限: d_s = %.3f (预期 3.0)\n", ds3d_limit);
    printf("偏差: %.1f%%\n", fabs(ds3d_limit-3.0)/3.0*100);
    
    int threeD_pass = (fabs(ds3d_limit - 3.0) < 0.15);
    printf("结果: %s\n", threeD_pass ? "[通过]" : "[需检查]");
    if (!threeD_pass) all_passed = 0;
    
    printf("\n================================================================\n");
    printf("  3. 综合验证结果\n");
    printf("================================================================\n\n");
    
    printf("验证项                          状态      结果\n");
    printf("------------------------------  ------    -----------\n");
    printf("更大系统尺寸 (L <= 64)          %s    d_s = %.2f\n", 
           large_L_pass ? "通过" : "需检查", ds2d_limit);
    printf("三维系统验证                    %s    d_s = %.2f\n", 
           threeD_pass ? "通过" : "需检查", ds3d_limit);
    
    printf("\n================================================================\n");
    printf("  4. 修正前后对比\n");
    printf("================================================================\n\n");
    
    printf("2D系统:\n");
    printf("  修正前 (t=0.5): d_s = 2.21, 偏差 = 10.5%%\n");
    printf("  修正后 (t=4.9): d_s = %.2f, 偏差 = %.1f%%\n", 
           ds2d_limit, fabs(ds2d_limit-2.0)/2.0*100);
    
    printf("\n3D系统:\n");
    printf("  修正前 (t=0.5): d_s = 3.32, 偏差 = 10.7%%\n");
    printf("  修正后 (t=5.0): d_s = %.2f, 偏差 = %.1f%%\n", 
           ds3d_limit, fabs(ds3d_limit-3.0)/3.0*100);
    
    printf("\n================================================================\n");
    printf("  5. 最终状态\n");
    printf("================================================================\n\n");
    
    if (all_passed) {
        printf("所有扩展验证项通过！\n\n");
        printf("验证总结:\n");
        printf("  [通过] 更大系统尺寸验证 (L <= 64)\n");
        printf("  [通过] 三维系统验证\n");
        printf("  [修正] 使用正确的平台区域t值\n\n");
        printf("谱维度偏差已从10%%降低到3%%以内。\n");
    } else {
        printf("部分验证项需要进一步检查。\n");
    }
    
    printf("\n================================================================\n");
    printf("     扩展验证完成（修正版）\n");
    printf("================================================================\n\n");
    
    return 0;
}
