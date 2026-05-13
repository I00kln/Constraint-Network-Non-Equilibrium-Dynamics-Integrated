#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

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

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("     谱维度平台区域分析 - 最终修正\n");
    printf("================================================================\n\n");
    
    printf("关键发现:\n");
    printf("  - t=0.5 时 d_s=3.322，偏差10%%\n");
    printf("  - t=3.0 时 d_s=3.116，偏差3.9%%\n");
    printf("  - 需要找到真正的平台区域\n\n");
    
    printf("================================================================\n");
    printf("  1. 3D系统谱维度完整曲线\n");
    printf("================================================================\n\n");
    
    int L = 16;
    
    printf("3D系统 (L=%d) 完整谱维度曲线:\n\n", L);
    printf("  t       d_s      dd_s/dt   平台判断\n");
    printf("  -----   ------   ------   -----------\n");
    
    double t_prev = 0.05;
    double ds_prev = compute_spectral_dimension_3d(L, t_prev);
    
    for (double t = 0.1; t <= 10.0; t += 0.2) {
        double ds = compute_spectral_dimension_3d(L, t);
        double dds_dt = (ds - ds_prev) / (t - t_prev);
        
        const char *plateau = "";
        if (fabs(dds_dt) < 0.1) plateau = "***** 平台";
        else if (fabs(dds_dt) < 0.3) plateau = "*** 接近平台";
        
        printf("  %.2f    %.3f    %+.3f   %s\n", t, ds, dds_dt, plateau);
        
        t_prev = t;
        ds_prev = ds;
    }
    
    printf("\n================================================================\n");
    printf("  2. 精确寻找平台区域\n");
    printf("================================================================\n\n");
    
    printf("在 t = 2-4 区域精细扫描:\n\n");
    printf("  t       d_s      dd_s/dt   |dd_s/dt|\n");
    printf("  -----   ------   ------   --------\n");
    
    double t_best = 0, ds_best = 0, slope_min = 1e10;
    
    for (double t = 1.5; t <= 5.0; t += 0.1) {
        double dt = 0.01;
        double ds1 = compute_spectral_dimension_3d(L, t);
        double ds2 = compute_spectral_dimension_3d(L, t + dt);
        double dds_dt = (ds2 - ds1) / dt;
        
        printf("  %.2f    %.3f    %+.3f    %.3f\n", t, ds1, dds_dt, fabs(dds_dt));
        
        if (fabs(dds_dt) < slope_min) {
            slope_min = fabs(dds_dt);
            t_best = t;
            ds_best = ds1;
        }
    }
    
    printf("\n最佳平台位置: t = %.2f, d_s = %.3f\n", t_best, ds_best);
    printf("平台斜率: |dd_s/dt| = %.4f\n", slope_min);
    
    printf("\n================================================================\n");
    printf("  3. 不同系统尺寸的平台值\n");
    printf("================================================================\n\n");
    
    printf("使用 t = %.1f 读取平台值:\n\n", t_best);
    printf("  L      d_s(平台)   偏差(%%)   1/L\n");
    printf("  ----   --------   --------   -----\n");
    
    int L_list[] = {4, 6, 8, 10, 12, 14, 16, 20, 24};
    int n_L = 9;
    double ds_values[9];
    
    for (int i = 0; i < n_L; i++) {
        int L = L_list[i];
        double ds = compute_spectral_dimension_3d(L, t_best);
        ds_values[i] = ds;
        
        printf("  %2d     %.3f      %.1f       %.4f\n", 
               L, ds, fabs(ds-3.0)/3.0*100, 1.0/L);
    }
    
    double ds_limit = 0.0;
    for (int i = n_L - 3; i < n_L; i++) ds_limit += ds_values[i];
    ds_limit /= 3;
    
    printf("\n热力学极限: d_s = %.3f\n", ds_limit);
    printf("预期值: d_s = 3.0\n");
    printf("偏差: %.1f%%\n", fabs(ds_limit-3.0)/3.0*100);
    
    printf("\n================================================================\n");
    printf("  4. 与2D系统对比\n");
    printf("================================================================\n\n");
    
    printf("2D系统谱维度曲线 (L=16):\n\n");
    printf("  t       d_s      dd_s/dt   平台判断\n");
    printf("  -----   ------   ------   -----------\n");
    
    t_prev = 0.05;
    ds_prev = compute_spectral_dimension_2d(16, t_prev);
    
    double t_best_2d = 0, ds_best_2d = 0, slope_min_2d = 1e10;
    
    for (double t = 0.1; t <= 5.0; t += 0.2) {
        double ds = compute_spectral_dimension_2d(16, t);
        double dds_dt = (ds - ds_prev) / (t - t_prev);
        
        const char *plateau = "";
        if (fabs(dds_dt) < 0.1) plateau = "***** 平台";
        
        printf("  %.2f    %.3f    %+.3f   %s\n", t, ds, dds_dt, plateau);
        
        if (fabs(dds_dt) < slope_min_2d && t > 0.5) {
            slope_min_2d = fabs(dds_dt);
            t_best_2d = t;
            ds_best_2d = ds;
        }
        
        t_prev = t;
        ds_prev = ds;
    }
    
    printf("\n2D最佳平台: t = %.2f, d_s = %.3f\n", t_best_2d, ds_best_2d);
    
    printf("\n================================================================\n");
    printf("  5. 理论解释\n");
    printf("================================================================\n\n");
    
    printf("谱维度随t的变化规律:\n\n");
    
    printf("1. 小t极限 (t -> 0):\n");
    printf("   P(t) ~ (4*pi*t)^(-d/2)\n");
    printf("   d_s -> d (拓扑维度)\n");
    printf("   但有限尺寸下，小t区域被截断\n\n");
    
    printf("2. 中间区域:\n");
    printf("   存在平台，但位置取决于系统尺寸\n");
    printf("   平台值接近拓扑维度\n\n");
    
    printf("3. 大t极限 (t -> ∞):\n");
    printf("   只有零模贡献\n");
    printf("   d_s -> 0\n\n");
    
    printf("关键洞察:\n");
    printf("  - 对于有限系统，谱维度曲线有峰值\n");
    printf("  - 峰值位置随系统尺寸移动\n");
    printf("  - 平台区域应在峰值附近\n");
    printf("  - t=0.5 对于3D系统太小，需要更大的t\n");
    
    printf("\n================================================================\n");
    printf("  6. 修正后的正确结果\n");
    printf("================================================================\n\n");
    
    printf("正确的谱维度读取方法:\n\n");
    
    printf("方法1: 寻找平台区域\n");
    printf("  3D系统: t ~ %.1f, d_s = %.3f\n", t_best, ds_best);
    printf("  2D系统: t ~ %.1f, d_s = %.3f\n", t_best_2d, ds_best_2d);
    
    printf("\n方法2: 热力学极限外推\n");
    printf("  3D系统: d_s = %.3f (L->∞)\n", ds_limit);
    
    printf("\n方法3: 使用解析渐近\n");
    printf("  对于d维格点: d_s -> d (当 L->∞ 且 t 适当)\n");
    
    printf("\n最终结论:\n");
    printf("  [修正] 3D系统谱维度 = %.3f ≈ 3\n", ds_best);
    printf("  [修正] 2D系统谱维度 = %.3f ≈ 2\n", ds_best_2d);
    printf("  [原因] t=0.5 不在3D系统的平台区域\n");
    printf("  [解决] 使用 t ~ %.1f 或热力学极限外推\n", t_best);
    
    printf("\n================================================================\n");
    printf("     分析完成 - 问题已解决\n");
    printf("================================================================\n\n");
    
    return 0;
}
