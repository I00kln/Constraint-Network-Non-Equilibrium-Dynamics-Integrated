# 代码与文档一致性分析报告

**日期**: 2026年5月14日  
**分析对象**: causal_dynamics_v2 vs 理论文档

---

## 一、文档要求总结

### 理论框架（关于时空观的一种新猜想-v5.md）

**核心概念**:
1. 因果图基底 Γ=(V,E)
2. 信息密度 ρ: V→R⁺
3. 宏观-微观分解：ρ = Φ + Ψ
4. 谱粗化：使用拉普拉斯本征模分解

**三代谱定义**（猜想3）:
> 按递归深度分层：第0层（无回返）→ 第一代，第1层（一次回返）→ 第二代，第2层（双重自反馈）→ 第三代。

**关键预测**:
- 本征值分布应出现**三个分离的簇**
- 簇间距随N增大趋于常数
- 质量比 ~ 1 : e^a : e^2a

### 数值仿真计划（数值仿真计划v5.md）

**动力学步骤**:
1. 因果传播
2. 谱宏微观分解（**无sigmoid**）
3. 总量重分配
4. 拓扑与相位重连
5. 监测与输出

**谱粗化要求**:
```
- 计算图拉普拉斯本征值 λ_i 和本征向量 ψ_i
- 选择低通截止 k_c
- 宏观场：Φ = P_{k<k_c} ρ
- 微观场：Ψ = ρ - Φ
```

**三代谱检测**:
```
统计本征值分布，寻找分立谱簇（三代候选）。
预期在低能区（小λ）出现三个分离的簇。
```

---

## 二、代码实现分析

### 当前实现

**文件**: `dynamics_v2.c`

**步骤实现**:
```c
void dynamics_full_step_v2(...) {
    step1_causal_propagation_v2(...);      // ✅ 步骤1
    step2_spectral_decomposition(...);     // ✅ 步骤2
    step3_total_redistribution_v2(...);    // ✅ 步骤3
    step4_topology_phase_rewiring(...);    // ✅ 步骤4
    // ❌ 缺少步骤5：监测与输出
}
```

**谱分解实现**:
```c
void step2_spectral_decomposition(...) {
    int num_eigenvalues = (int)(kc_fraction * state->num_nodes);
    compute_laplacian_spectrum(graph, spectrum, num_eigenvalues);
    
    int cutoff = (int)(kc_fraction * spectrum->num_eigenvalues);
    project_to_macro(spectrum, state->rho, state->macro, ...);
    project_to_micro(spectrum, state->rho, state->micro, ...);
}
```

**谱簇检测**:
```c
void analyze_eigenvalue_clusters(...) {
    // 使用间隙检测
    if (gaps[i] > threshold * mean_gap) {
        // 新簇
    }
}
```

---

## 三、问题识别

### 问题1：本征值数量不足 ⚠️

**文档要求**: 计算完整的拉普拉斯本征谱

**当前实现**:
```c
int num_eigenvalues = (int)(kc_fraction * state->num_nodes);
// kc_fraction = 0.1 → 只计算10%的本征值
```

**问题**:
- N=600, kc=0.1 → 只计算60个本征值
- 可能漏掉重要的本征值结构
- 三代谱可能在更高本征值区域

**影响**: ⭐⭐⭐⭐⭐ 严重

---

### 问题2：谱簇检测算法不当 ⚠️

**文档要求**: 寻找本征值分布中的**三个分离簇**

**当前实现**: 
1. 间隙检测（简单阈值）
2. K-means聚类（但效果不佳）

**问题**:
- 间隙检测对阈值敏感
- K-means假设球形簇，不适合本征值分布
- 未考虑**簇的物理意义**（递归深度）

**影响**: ⭐⭐⭐⭐⭐ 严重

---

### 问题3：cutoff设置不当 ⚠️

**文档要求**: 选择低通截止k_c（本征值<0.1对应的特征模数目）

**当前实现**:
```c
int cutoff = (int)(kc_fraction * spectrum->num_eigenvalues);
// kc_fraction = 0.1 → cutoff = 6 (num_eigenvalues=60)
```

**问题**:
- cutoff太小，宏观场信息不足
- 应该基于本征值大小，而非数量
- 文档要求：λ < 0.1，而非前10%

**影响**: ⭐⭐⭐⭐ 重要

---

### 问题4：缺少本征值间隙分析 ⚠️

**文档要求**: 
> 本征值位于间隙（gap）中的模式，且参与比高 → 局域稳定结构

**当前实现**: 未计算参与比（IPR）

**问题**:
- 无法识别稳定模
- 无法区分物理簇和噪声

**影响**: ⭐⭐⭐ 中等

---

### 问题5：缺少递归深度分析 ⚠️

**文档要求**:
> 按递归深度分层：第0层→第一代，第1层→第二代，第2层→第三代

**当前实现**: 未实现递归深度分析

**问题**:
- 三代谱的物理基础未实现
- 无法验证猜想3

**影响**: ⭐⭐⭐⭐⭐ 严重

---

## 四、改进方案

### 改进1：增加本征值计算数量

```c
// 修改前
int num_eigenvalues = (int)(kc_fraction * state->num_nodes);

// 修改后
int num_eigenvalues = (int)(0.5 * state->num_nodes);  // 计算50%
if (num_eigenvalues > MAX_EIGENVECTORS) {
    num_eigenvalues = MAX_EIGENVECTORS;
}
```

**理由**: 需要足够的本征值来识别簇结构

---

### 改进2：基于本征值大小的cutoff

```c
// 修改前
int cutoff = (int)(kc_fraction * spectrum->num_eigenvalues);

// 修改后
int cutoff = 0;
double lambda_threshold = 0.1;  // 文档要求
for (int i = 0; i < spectrum->num_eigenvalues; i++) {
    if (spectrum->eigenvalues[i] < lambda_threshold) {
        cutoff = i + 1;
    }
}
```

**理由**: 符合文档"本征值<0.1"的要求

---

### 改进3：改进谱簇检测算法

**新算法**: 基于间隙统计的层次聚类

```c
void analyze_eigenvalue_clusters_hierarchical(...) {
    // 1. 计算所有间隙
    double gaps[MAX_EIGENVECTORS];
    for (int i = 0; i < n-1; i++) {
        gaps[i] = eigenvalues[i+1] - eigenvalues[i];
    }
    
    // 2. 识别显著间隙（> mean + 2*std）
    double mean_gap = ..., std_gap = ...;
    double threshold = mean_gap + 2.0 * std_gap;
    
    // 3. 基于显著间隙分割簇
    int num_clusters = 1;
    for (int i = 0; i < n-1; i++) {
        if (gaps[i] > threshold) {
            num_clusters++;
        }
    }
    
    // 4. 检验是否为三代
    result->three_generation_detected = (num_clusters >= 3);
}
```

---

### 改进4：实现参与比计算

```c
double compute_ipr(const double *eigenvector, int n) {
    double sum_sq = 0.0, sum_quad = 0.0;
    for (int i = 0; i < n; i++) {
        double psi_sq = eigenvector[i] * eigenvector[i];
        sum_sq += psi_sq;
        sum_quad += psi_sq * psi_sq;
    }
    return sum_quad / (sum_sq * sum_sq);  // IPR
}
```

**物理意义**: IPR高 → 局域化模式 → 稳定结构

---

### 改进5：实现递归深度分析

```c
void analyze_recursive_depth(const CausalGraphV2 *graph, 
                              int *depth_distribution) {
    // 计算每个节点的递归深度
    for (int v = 0; v < graph->num_nodes; v++) {
        depth_distribution[v] = compute_recursive_depth(graph, v);
    }
    
    // 统计三代分布
    int gen1 = 0, gen2 = 0, gen3 = 0;
    for (int v = 0; v < graph->num_nodes; v++) {
        if (depth_distribution[v] == 0) gen1++;
        else if (depth_distribution[v] == 1) gen2++;
        else if (depth_distribution[v] == 2) gen3++;
    }
}
```

---

## 五、预期效果

### 改进前

```
N=600, kc=0.1:
- 本征值数量: 60
- cutoff: 6
- 谱簇数: 2
- 三代涌现: No
```

### 改进后（预期）

```
N=600, kc=0.1:
- 本征值数量: 300
- cutoff: ~30 (基于λ<0.1)
- 谱簇数: 3 (预期)
- 三代涌现: Yes (预期)
```

---

## 六、优先级排序

| 问题 | 优先级 | 难度 | 影响 |
|------|--------|------|------|
| 本征值数量不足 | P0 | 低 | ⭐⭐⭐⭐⭐ |
| cutoff设置不当 | P0 | 低 | ⭐⭐⭐⭐ |
| 谱簇检测算法 | P0 | 中 | ⭐⭐⭐⭐⭐ |
| 缺少参与比 | P1 | 低 | ⭐⭐⭐ |
| 缺少递归深度 | P2 | 高 | ⭐⭐⭐⭐⭐ |

---

## 七、结论

### 关键发现

1. **本征值计算不足**: 只计算10%，可能漏掉重要结构
2. **cutoff设置错误**: 基于数量而非本征值大小
3. **谱簇检测不当**: 未考虑物理意义和间隙统计
4. **缺少关键分析**: 参与比、递归深度

### 根本原因

**算法实现与理论文档存在偏差**:
- 文档要求基于本征值大小选择cutoff
- 代码实现基于数量比例
- 导致谱结构分析不完整

### 改进方向

1. **立即修复**: 本征值数量、cutoff设置
2. **算法改进**: 谱簇检测、参与比计算
3. **长期实现**: 递归深度分析

---

**分析时间**: 2026年5月14日 16:30  
**状态**: 问题识别完成 ✅  
**下一步**: 实施改进方案
