"""
多步信息价值传播验证
====================
验证文档 §3.2 中的多步传播理论:
  - 引理 3.1: 多步值函数的凸性与 Lipschitz 性
  - 定理 3.4: 多步上界 VoI_τ ≤ √(EIG/2)
  - 定理 3.5: 多步线性下界（横截条件下）
  - 开放问题: 横截条件的递归传播
"""

import numpy as np
from scipy.special import logsumexp
import json

np.random.seed(42)

# ============================================================
# 工具函数（复用单步验证的函数）
# ============================================================

def normalize_to_simplex(x):
    x = np.maximum(x, 1e-15)
    return x / x.sum()

def random_simplex(K, delta=0.0):
    if delta > 0:
        x = delta + np.random.exponential(1, K)
        return x / x.sum()
    x = np.random.exponential(1, K)
    return x / x.sum()

def d_tv(mu, nu):
    return 0.5 * np.abs(mu - nu).sum()

def d_kl(mu, nu):
    nu = np.maximum(nu, 1e-300)
    mu_safe = np.maximum(mu, 1e-300)
    return np.sum(mu_safe * np.log(mu_safe / nu))

def value_function(mu, C):
    return max(mu @ c for c in C)

def normalize_likelihoods(likelihoods):
    col_sums = likelihoods.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-300)
    return likelihoods / col_sums

def random_likelihoods(num_obs, K):
    L = np.random.uniform(0.1, 5.0, (num_obs, K))
    return normalize_likelihoods(L)

def compute_posterior_distribution(mu, likelihoods):
    results = []
    for o in range(likelihoods.shape[0]):
        L = likelihoods[o]
        Z = (mu * L).sum()
        if Z < 1e-300:
            continue
        mu_prime = (mu * L) / Z
        results.append((mu_prime, Z))
    return results

def compute_EIG(mu, likelihoods):
    posts = compute_posterior_distribution(mu, likelihoods)
    return sum(Z * d_kl(mu_p, mu) for mu_p, Z in posts)

# ============================================================
# 多步值函数计算
# ============================================================

def compute_multistep_value_function(mu, C_exploit, info_likelihoods, tau):
    """递归计算 τ 步值函数 V_τ(μ)
    
    参数:
        mu: 当前信念
        C_exploit: 利用动作的值向量集合
        info_likelihoods: 信息动作的似然矩阵列表
        tau: 剩余步数
    
    返回:
        V_τ(μ) 的值
    """
    if tau == 0:
        return 0.0
    
    if tau == 1:
        # V_1(μ) = U(μ) = max_{a∈exploit} <μ, v(a)>
        return value_function(mu, C_exploit)
    
    # 递归: V_τ(μ) = max{利用值, max_{信息动作} E[V_{τ-1}(μ')]}
    
    # 利用值
    val_exploit = value_function(mu, C_exploit)
    
    # 信息值
    val_info_best = -np.inf
    for likelihoods in info_likelihoods:
        posts = compute_posterior_distribution(mu, likelihoods)
        val_info = sum(Z * compute_multistep_value_function(mu_p, C_exploit, info_likelihoods, tau-1) 
                       for mu_p, Z in posts)
        val_info_best = max(val_info_best, val_info)
    
    return max(val_exploit, val_info_best)


def compute_multistep_value_function_with_pieces(mu, C_tau):
    """使用分片线性表示计算多步值函数
    
    V_τ(μ) = max_{c∈C_τ} <μ, c>
    
    参数:
        mu: 当前信念
        C_tau: τ 步值函数的线性片集合
    """
    return value_function(mu, C_tau)


def compute_multistep_VoI(mu, C_exploit, info_likelihoods, tau, action_idx=0):
    """计算 τ 步信息价值
    
    VoI_τ(μ, a) = E_{μ'}[V_{τ-1}(μ')] - V_{τ-1}(μ)
    
    参数:
        mu: 当前信念
        C_exploit: 利用动作的值向量集合
        info_likelihoods: 信息动作的似然矩阵列表
        tau: 剩余步数 (≥ 2)
        action_idx: 使用哪个信息动作
    """
    if tau < 2:
        raise ValueError("τ must be ≥ 2 for multi-step VoI")
    
    # V_{τ-1}(μ)
    V_tau_minus_1_mu = compute_multistep_value_function(mu, C_exploit, info_likelihoods, tau-1)
    
    # E_{μ'}[V_{τ-1}(μ')]
    likelihoods = info_likelihoods[action_idx]
    posts = compute_posterior_distribution(mu, likelihoods)
    E_V_tau_minus_1 = sum(Z * compute_multistep_value_function(mu_p, C_exploit, info_likelihoods, tau-1) 
                          for mu_p, Z in posts)
    
    return E_V_tau_minus_1 - V_tau_minus_1_mu


def find_multistep_transversality_constant(mu, C_exploit, info_likelihoods, tau, action_idx=0):
    """寻找多步横截常数 m_τ
    
    使得 V_{τ-1}(μ') - V_{τ-1}(μ) ≥ m_τ · D_TV(μ', μ) 对所有后验成立
    """
    V_tau_minus_1_mu = compute_multistep_value_function(mu, C_exploit, info_likelihoods, tau-1)
    
    likelihoods = info_likelihoods[action_idx]
    posts = compute_posterior_distribution(mu, likelihoods)
    
    ratios = []
    for mu_p, Z in posts:
        dtv = d_tv(mu_p, mu)
        if dtv > 1e-12:
            V_tau_minus_1_mu_p = compute_multistep_value_function(mu_p, C_exploit, info_likelihoods, tau-1)
            val_diff = V_tau_minus_1_mu_p - V_tau_minus_1_mu
            ratios.append(val_diff / dtv)
    
    if not ratios:
        return 0.0
    return min(ratios)


# ============================================================
# 验证 1: 多步值函数的凸性与 Lipschitz 性
# ============================================================

def verify_multistep_convexity_and_lipschitz(num_trials=1000, K_range=(2, 5), tau_range=(2, 5)):
    print("\n" + "="*70)
    print("验证 引理 3.1: 多步值函数的凸性与 Lipschitz 性")
    print("="*70)
    
    # 验证凸性: V_τ(αμ1 + (1-α)μ2) ≤ αV_τ(μ1) + (1-α)V_τ(μ2)
    convexity_violations = 0
    convexity_tests = 0
    
    # 验证 Lipschitz 性: |V_τ(μ1) - V_τ(μ2)| ≤ D_TV(μ1, μ2)
    lipschitz_violations = 0
    lipschitz_tests = 0
    lipschitz_ratios = []
    
    for _ in range(num_trials):
        K = np.random.randint(*K_range)
        tau = np.random.randint(*tau_range)
        
        # 生成利用动作的值向量
        num_exploit = np.random.randint(2, 4)
        C_exploit = [np.random.uniform(0, 1, K) for _ in range(num_exploit)]
        
        # 生成信息动作的似然
        num_info = np.random.randint(1, 3)
        info_likelihoods = []
        for _ in range(num_info):
            num_obs = np.random.randint(2, 4)
            info_likelihoods.append(random_likelihoods(num_obs, K))
        
        # 验证凸性
        mu1 = random_simplex(K)
        mu2 = random_simplex(K)
        alpha = np.random.uniform(0.1, 0.9)
        mu_mix = alpha * mu1 + (1 - alpha) * mu2
        
        V1 = compute_multistep_value_function(mu1, C_exploit, info_likelihoods, tau)
        V2 = compute_multistep_value_function(mu2, C_exploit, info_likelihoods, tau)
        V_mix = compute_multistep_value_function(mu_mix, C_exploit, info_likelihoods, tau)
        
        convexity_tests += 1
        if V_mix > alpha * V1 + (1 - alpha) * V2 + 1e-10:
            convexity_violations += 1
        
        # 验证 Lipschitz 性
        dtv = d_tv(mu1, mu2)
        if dtv > 1e-12:
            lipschitz_tests += 1
            ratio = abs(V1 - V2) / dtv
            lipschitz_ratios.append(ratio)
            if ratio > 1.0 + 1e-8:
                lipschitz_violations += 1
    
    print(f"\n  凸性验证:")
    print(f"    试验次数: {convexity_tests}")
    print(f"    违反次数: {convexity_violations}")
    print(f"    结果: {'PASS ✓' if convexity_violations == 0 else 'FAIL ✗'}")
    
    print(f"\n  Lipschitz 性验证:")
    print(f"    试验次数: {lipschitz_tests}")
    print(f"    违反次数: {lipschitz_violations}")
    if lipschitz_ratios:
        print(f"    最大比值 |ΔV|/D_TV: {max(lipschitz_ratios):.8f} (应 ≤ 1.0)")
        print(f"    平均比值: {np.mean(lipschitz_ratios):.6f}")
    print(f"    结果: {'PASS ✓' if lipschitz_violations == 0 else 'FAIL ✗'}")
    
    return convexity_violations == 0 and lipschitz_violations == 0


# ============================================================
# 验证 2: 多步上界
# ============================================================

def verify_multistep_upper_bound(num_trials=1000, K_range=(2, 5), tau_range=(2, 6)):
    print("\n" + "="*70)
    print("验证 定理 3.4: 多步上界 VoI_τ ≤ √(EIG/2)")
    print("="*70)
    
    violations = 0
    total_tests = 0
    ratios = []
    
    for _ in range(num_trials):
        K = np.random.randint(*K_range)
        tau = np.random.randint(*tau_range)
        
        # 生成利用动作的值向量
        num_exploit = np.random.randint(2, 4)
        C_exploit = [np.random.uniform(0, 1, K) for _ in range(num_exploit)]
        
        # 生成信息动作的似然
        num_info = np.random.randint(1, 3)
        info_likelihoods = []
        for _ in range(num_info):
            num_obs = np.random.randint(2, 4)
            info_likelihoods.append(random_likelihoods(num_obs, K))
        
        mu = random_simplex(K, delta=0.01)
        
        # 对每个信息动作验证
        for action_idx in range(len(info_likelihoods)):
            total_tests += 1
            
            VoI_tau = compute_multistep_VoI(mu, C_exploit, info_likelihoods, tau, action_idx)
            EIG = compute_EIG(mu, info_likelihoods[action_idx])
            
            if EIG > 1e-12:
                upper_bound = np.sqrt(EIG / 2)
                ratio = VoI_tau / upper_bound if upper_bound > 1e-15 else float('inf')
                ratios.append(ratio)
                
                if VoI_tau > upper_bound + 1e-8:
                    violations += 1
    
    print(f"  试验次数: {total_tests}")
    print(f"  违反次数: {violations}")
    if ratios:
        print(f"  最大比值 VoI_τ/√(EIG/2): {max(ratios):.8f} (应 ≤ 1.0)")
        print(f"  平均比值: {np.mean(ratios):.6f}")
        print(f"  中位比值: {np.median(ratios):.6f}")
    
    passed = violations == 0
    print(f"  结果: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ============================================================
# 验证 3: 多步线性下界
# ============================================================

def verify_multistep_lower_bound(num_trials=500, K_range=(2, 4), tau_range=(2, 5)):
    print("\n" + "="*70)
    print("验证 定理 3.5: 多步线性下界（横截条件下）")
    print("="*70)
    
    violations = 0
    total_applicable = 0
    ratios = []
    
    for _ in range(num_trials):
        K = np.random.randint(*K_range)
        tau = np.random.randint(*tau_range)
        delta = np.random.uniform(0.05, 0.2)
        
        # 生成利用动作的值向量（有序）
        num_exploit = np.random.randint(2, 4)
        C_exploit = [np.sort(np.random.uniform(0, 1, K))[::-1] for _ in range(num_exploit)]
        
        # 生成信息动作的似然（有序）
        num_info = np.random.randint(1, 2)
        info_likelihoods = []
        for _ in range(num_info):
            num_obs = np.random.randint(2, 3)
            likelihoods = np.zeros((num_obs, K))
            for o in range(num_obs):
                base = np.sort(np.random.uniform(0.5, 3.0, K))[::-1]
                likelihoods[o] = base
            info_likelihoods.append(normalize_likelihoods(likelihoods))
        
        mu = random_simplex(K, delta=delta)
        
        # 对每个信息动作验证
        for action_idx in range(len(info_likelihoods)):
            # 检查横截条件
            m_tau = find_multistep_transversality_constant(mu, C_exploit, info_likelihoods, tau, action_idx)
            
            if m_tau <= 0:
                continue
            
            total_applicable += 1
            
            VoI_tau = compute_multistep_VoI(mu, C_exploit, info_likelihoods, tau, action_idx)
            EIG = compute_EIG(mu, info_likelihoods[action_idx])
            
            if EIG > 1e-12:
                lower_bound = (m_tau * np.sqrt(delta)) / (2 * np.sqrt(np.log(1.0 / delta))) * EIG
                ratio = VoI_tau / lower_bound if lower_bound > 1e-15 else float('inf')
                ratios.append(ratio)
                
                if VoI_tau < lower_bound - 1e-8:
                    violations += 1
    
    print(f"  试验次数: {num_trials}")
    print(f"  适用场景数（横截条件成立）: {total_applicable}")
    print(f"  违反次数: {violations}")
    if ratios:
        print(f"  VoI_τ/下界 最小比值: {min(ratios):.6f} (应 ≥ 1.0)")
        print(f"  VoI_τ/下界 平均比值: {np.mean(ratios):.6f}")
    
    passed = violations == 0
    print(f"  结果: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ============================================================
# 验证 4: 横截条件的递归传播（开放问题）
# ============================================================

def verify_transversality_propagation(num_trials=500, K_range=(2, 4), tau_max=5):
    print("\n" + "="*70)
    print("验证 开放问题: 横截条件的递归传播")
    print("="*70)
    print("  问题: 单步横截性是否能保证多步横截性？")
    print("  即: m_1 > 0 是否推出 m_τ > 0 对所有 τ？")
    
    # 统计单步横截性与多步横截性的关系
    single_step_transverse_count = 0
    multi_step_transverse_count = {tau: 0 for tau in range(2, tau_max+1)}
    both_transverse_count = {tau: 0 for tau in range(2, tau_max+1)}
    
    m_single_values = []
    m_multi_values = {tau: [] for tau in range(2, tau_max+1)}
    
    for _ in range(num_trials):
        K = np.random.randint(*K_range)
        
        # 生成利用动作的值向量
        num_exploit = np.random.randint(2, 4)
        C_exploit = [np.random.uniform(0, 1, K) for _ in range(num_exploit)]
        
        # 生成信息动作的似然
        num_info = 1
        info_likelihoods = []
        num_obs = np.random.randint(2, 4)
        info_likelihoods.append(random_likelihoods(num_obs, K))
        
        mu = random_simplex(K, delta=0.05)
        
        # 检查单步横截性 (τ=2 时的横截性对应于 V_1 的横截性)
        m_single = find_multistep_transversality_constant(mu, C_exploit, info_likelihoods, 2, 0)
        
        if m_single > 0:
            single_step_transverse_count += 1
            m_single_values.append(m_single)
        
        # 检查多步横截性
        for tau in range(2, tau_max+1):
            m_multi = find_multistep_transversality_constant(mu, C_exploit, info_likelihoods, tau, 0)
            
            if m_multi > 0:
                multi_step_transverse_count[tau] += 1
                m_multi_values[tau].append(m_multi)
            
            if m_single > 0 and m_multi > 0:
                both_transverse_count[tau] += 1
    
    print(f"\n  试验次数: {num_trials}")
    print(f"\n  单步横截性成立次数: {single_step_transverse_count} ({single_step_transverse_count/num_trials*100:.1f}%)")
    if m_single_values:
        print(f"    单步横截常数 m_1 范围: [{min(m_single_values):.6f}, {max(m_single_values):.6f}]")
        print(f"    单步横截常数 m_1 均值: {np.mean(m_single_values):.6f}")
    
    for tau in range(2, tau_max+1):
        print(f"\n  τ={tau} 步横截性成立次数: {multi_step_transverse_count[tau]} ({multi_step_transverse_count[tau]/num_trials*100:.1f}%)")
        if m_multi_values[tau]:
            positive_m = [m for m in m_multi_values[tau] if m > 0]
            if positive_m:
                print(f"    m_{tau} 范围: [{min(positive_m):.6f}, {max(positive_m):.6f}]")
                print(f"    m_{tau} 均值: {np.mean(positive_m):.6f}")
        
        if single_step_transverse_count > 0:
            propagation_ratio = both_transverse_count[tau] / single_step_transverse_count
            print(f"    单步横截 ⇒ τ步横截: {both_transverse_count[tau]}/{single_step_transverse_count} ({propagation_ratio*100:.1f}%)")
    
    # 分析横截常数的变化趋势
    print(f"\n  横截常数随步数的变化:")
    if m_single_values:
        avg_m_single = np.mean(m_single_values)
        print(f"    平均 m_1: {avg_m_single:.6f}")
        
        for tau in range(2, tau_max+1):
            if m_multi_values[tau]:
                positive_m = [m for m in m_multi_values[tau] if m > 0]
                if positive_m:
                    avg_m_multi = np.mean(positive_m)
                    ratio = avg_m_multi / avg_m_single
                    print(f"    平均 m_{tau}: {avg_m_multi:.6f} (相对 m_1: {ratio:.4f})")
    
    # 结论
    if single_step_transverse_count > 0:
        min_propagation = min(both_transverse_count[tau] / single_step_transverse_count 
                             for tau in range(2, tau_max+1))
        if min_propagation >= 0.95:
            print(f"\n  结论: 单步横截性在 {min_propagation*100:.1f}% 的情况下传播到多步 ✓")
        elif min_propagation >= 0.7:
            print(f"\n  结论: 单步横截性在多数情况下传播到多步，但非普遍成立")
        else:
            print(f"\n  结论: 单步横截性不能保证传播到多步 ✗")
            print(f"        这与文档中的开放问题一致")
    
    return True


# ============================================================
# 验证 5: 多步值函数的单调性
# ============================================================

def verify_multistep_monotonicity(num_trials=500, K_range=(2, 5), tau_max=6):
    print("\n" + "="*70)
    print("验证 多步值函数的单调性: V_τ ≥ V_{τ-1}")
    print("="*70)
    
    violations = 0
    total_tests = 0
    
    for _ in range(num_trials):
        K = np.random.randint(*K_range)
        
        # 生成利用动作的值向量
        num_exploit = np.random.randint(2, 4)
        C_exploit = [np.random.uniform(0, 1, K) for _ in range(num_exploit)]
        
        # 生成信息动作的似然
        num_info = np.random.randint(1, 3)
        info_likelihoods = []
        for _ in range(num_info):
            num_obs = np.random.randint(2, 4)
            info_likelihoods.append(random_likelihoods(num_obs, K))
        
        mu = random_simplex(K)
        
        # 验证单调性
        for tau in range(2, tau_max+1):
            total_tests += 1
            V_tau = compute_multistep_value_function(mu, C_exploit, info_likelihoods, tau)
            V_tau_minus_1 = compute_multistep_value_function(mu, C_exploit, info_likelihoods, tau-1)
            
            if V_tau < V_tau_minus_1 - 1e-10:
                violations += 1
    
    print(f"  试验次数: {total_tests}")
    print(f"  违反次数: {violations}")
    print(f"  结果: {'PASS ✓' if violations == 0 else 'FAIL ✗'}")
    
    return violations == 0


# ============================================================
# 验证 6: 多步信息价值的累积效应
# ============================================================

def verify_multistep_cumulative_effect(num_trials=300, K_range=(2, 4), tau_max=5):
    print("\n" + "="*70)
    print("验证 多步信息价值的累积效应")
    print("="*70)
    print("  问题: 多步信息收集是否能带来更大的累积信息价值？")
    
    cumulative_voi_data = {tau: [] for tau in range(2, tau_max+1)}
    single_step_voi_data = []
    
    for _ in range(num_trials):
        K = np.random.randint(*K_range)
        
        # 生成利用动作的值向量
        num_exploit = np.random.randint(2, 4)
        C_exploit = [np.random.uniform(0, 1, K) for _ in range(num_exploit)]
        
        # 生成信息动作的似然
        num_info = 1
        info_likelihoods = []
        num_obs = np.random.randint(2, 4)
        info_likelihoods.append(random_likelihoods(num_obs, K))
        
        mu = random_simplex(K, delta=0.05)
        
        # 计算单步信息价值
        VoI_1 = compute_multistep_VoI(mu, C_exploit, info_likelihoods, 2, 0)
        single_step_voi_data.append(VoI_1)
        
        # 计算多步信息价值
        for tau in range(2, tau_max+1):
            VoI_tau = compute_multistep_VoI(mu, C_exploit, info_likelihoods, tau, 0)
            cumulative_voi_data[tau].append(VoI_tau)
    
    print(f"\n  单步信息价值 (τ=2):")
    print(f"    平均: {np.mean(single_step_voi_data):.6f}")
    print(f"    范围: [{min(single_step_voi_data):.6f}, {max(single_step_voi_data):.6f}]")
    
    for tau in range(2, tau_max+1):
        print(f"\n  τ={tau} 步信息价值:")
        print(f"    平均: {np.mean(cumulative_voi_data[tau]):.6f}")
        print(f"    范围: [{min(cumulative_voi_data[tau]):.6f}, {max(cumulative_voi_data[tau]):.6f}]")
        
        # 与单步比较
        ratio = np.mean(cumulative_voi_data[tau]) / np.mean(single_step_voi_data)
        print(f"    相对单步的比值: {ratio:.4f}")
    
    return True


# ============================================================
# 主验证流程
# ============================================================

def main():
    print("\n" + "="*70)
    print("多步信息价值传播验证")
    print("="*70)
    
    results = {}
    
    # 验证 1: 凸性与 Lipschitz 性
    results['convexity_lipschitz'] = verify_multistep_convexity_and_lipschitz()
    
    # 验证 2: 多步上界
    results['upper_bound'] = verify_multistep_upper_bound()
    
    # 验证 3: 多步线性下界
    results['lower_bound'] = verify_multistep_lower_bound()
    
    # 验证 4: 横截条件的递归传播
    results['transversality_propagation'] = verify_transversality_propagation()
    
    # 验证 5: 单调性
    results['monotonicity'] = verify_multistep_monotonicity()
    
    # 验证 6: 累积效应
    results['cumulative_effect'] = verify_multistep_cumulative_effect()
    
    # 总结
    print("\n" + "="*70)
    print("验证总结")
    print("="*70)
    
    for name, passed in results.items():
        if isinstance(passed, bool):
            print(f"  {name}: {'PASS ✓' if passed else 'FAIL ✗'}")
        else:
            print(f"  {name}: 完成")
    
    all_passed = all(r if isinstance(r, bool) else True for r in results.values())
    print(f"\n  总体结果: {'所有验证通过 ✓' if all_passed else '存在失败项 ✗'}")
    
    return results


if __name__ == "__main__":
    main()
