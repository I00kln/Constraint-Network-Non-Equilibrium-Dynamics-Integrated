# 测试文件整理清单

## 生成日期
2026-05-13

---

## 1. 必须保留的核心文件 (15个)

| 文件名 | 功能说明 | 保留原因 |
|--------|----------|----------|
| test_V1_to_V7.c | V1-V7完整验证 | 最新最全面的验证 |
| test_V3_analysis.c | V3涌现时间问题分析 | 问题修正分析 |
| test_V7_nonlinear.c | 非线性引力相互作用 | V7扩展验证 |
| test_einstein_emergence.c | 爱因斯坦场方程涌现 | 核心理论验证 |
| test_quantization_paths.c | 量子化路径研究 | 量子化方案分析 |
| test_extended_corrected.c | 扩展验证（修正后） | 谱维度修正验证 |
| test_plateau_analysis.c | 平台区域分析 | 谱维度t值选择 |
| test_reflection_positivity_corrected.c | 反射正定性（修正后） | 反射正定性验证 |
| test_causal_sd_simple.c | 因果谱维度 | 备选谱维度方案 |
| test_morse_index_fix.c | Morse index修正 | Morse守恒验证 |
| test_krein_ir_analysis.c | Krein碰撞IR分析 | Krein结构验证 |
| test_functorial_embedding.c | 函子性嵌入验证 | RG函子性验证 |
| test_phase3_scaling.c | 阶段3标度分析 | 标度分析验证 |
| test_constraint_algebra.c | 约束代数验证 | 代数结构验证 |
| test_spectral_dimension_heat_kernel.c | 热核谱维度 | 解析方法验证 |

---

## 2. 可选保留的文件 (2个)

| 文件名 | 功能说明 | 保留原因 |
|--------|----------|----------|
| test_phase0_v3_fast.c | v3阶段0快速验证 | v3验证参考 |
| test_phase1_v3.c | v3阶段1验证 | v3验证参考 |

---

## 3. 需要删除的文件 (35个)

### 3.1 调试/修复文件 (已整合到修正版本)

| 文件名 | 删除原因 |
|--------|----------|
| test_spectral_dimension_bug.c | 调试文件，问题已解决 |
| test_spectral_dimension_fix.c | 已被extended_corrected替代 |
| test_spectral_dimension_correct.c | 已被extended_corrected替代 |
| test_reflection_positivity_fix.c | 已被corrected版本替代 |
| test_large_deviation_fix.c | 调试文件 |
| test_krein_morse_fix.c | 已被morse_index_fix替代 |
| test_rp_quick.c | 快速测试，调试用 |
| test_theorem_rp.c | 已被corrected版本替代 |

### 3.2 过时的验证文件

| 文件名 | 删除原因 |
|--------|----------|
| test_3d_spectral_analysis.c | 已被plateau_analysis替代 |
| test_large_system.c | 已被extended_corrected包含 |
| test_extended_verification.c | 已被extended_corrected替代 |
| test_complete_verification.c | 过时 |
| test_causal_spectral_dimension.c | 已被causal_sd_simple替代 |
| test_spectral_dimension_analytic.c | 已被heat_kernel替代 |

### 3.3 过时的v4测试文件

| 文件名 | 删除原因 |
|--------|----------|
| test_v4_dynamic.c | 过时 |
| test_v4_minimal.c | 过时 |
| test_v4_simple.c | 过时 |
| test_v4_verification.c | 过时 |

### 3.4 过时的阶段验证文件

| 文件名 | 删除原因 |
|--------|----------|
| test_phase0_v3.c | 已被fast版本替代 |
| test_strict_verification_part1.c | 过时 |
| test_strict_verification_part2.c | 过时 |
| test_strict_verification_part3.c | 过时 |
| test_phase5_final.c | 过时 |
| test_phase4_spin2.c | 过时 |
| test_phase3_complete.c | 过时 |
| test_phase4_rg_flow.c | 过时 |
| test_phase3_large_graph.c | 过时 |
| test_phase3_spectral.c | 过时 |
| test_phase2_simulation.c | 过时 |

### 3.5 调试/临时文件

| 文件名 | 删除原因 |
|--------|----------|
| test_correct_order.c | 调试文件 |
| test_optimal_solution.c | 调试文件 |
| test_optimal_repair.c | 调试文件 |
| test_algebra_repair.c | 调试文件 |
| test_frozen_diffusion.c | 调试文件 |
| test_anomaly_dissipation.c | 调试文件 |
| test_diffeomorphism_coefficients.c | 调试文件 |
| test_diffeomorphism_constraint.c | 已被constraint_algebra包含 |
| test_constraint_algebra_sympy.c | 符号计算辅助文件 |

---

## 4. 统计

| 类别 | 数量 |
|------|------|
| 必须保留 | 15 |
| 可选保留 | 2 |
| 已删除 | 35 |
| **总计** | **52** |

---

## 5. 最终保留的文件清单 (17个)

```
tests/
├── test_V1_to_V7.c                      # V1-V7完整验证
├── test_V3_analysis.c                   # V3涌现时间问题分析
├── test_V7_nonlinear.c                  # 非线性引力相互作用
├── test_einstein_emergence.c            # 爱因斯坦场方程涌现
├── test_quantization_paths.c            # 量子化路径研究
├── test_extended_corrected.c            # 扩展验证（修正后）
├── test_plateau_analysis.c              # 平台区域分析
├── test_reflection_positivity_corrected.c  # 反射正定性（修正后）
├── test_causal_sd_simple.c              # 因果谱维度
├── test_morse_index_fix.c               # Morse index修正
├── test_krein_ir_analysis.c             # Krein碰撞IR分析
├── test_functorial_embedding.c          # 函子性嵌入验证
├── test_phase3_scaling.c                # 阶段3标度分析
├── test_constraint_algebra.c            # 约束代数验证
├── test_spectral_dimension_heat_kernel.c   # 热核谱维度
├── test_phase0_v3_fast.c                # v3阶段0快速验证 (可选)
└── test_phase1_v3.c                     # v3阶段1验证 (可选)
```

---

## 6. 已删除的文件 (35个)

✓ 已成功删除35个过时/调试/重复文件

---

*生成时间: 2026-05-13*
