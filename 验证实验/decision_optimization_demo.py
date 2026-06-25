"""
值信息对齐理论的实际应用演示
================================
展示如何使用 VoI-EIG 理论来优化下一步决策和预测

应用场景: 传感器网络中的目标捕获
"""

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 核心工具函数
# ============================================================

def normalize_to_simplex(x):
    x = np.maximum(x, 1e-15)
    return x / x.sum()

def d_tv(mu, nu):
    return 0.5 * np.abs(mu - nu).sum()

def d_kl(mu, nu):
    nu = np.maximum(nu, 1e-300)
    mu_safe = np.maximum(mu, 1e-300)
    return np.sum(mu_safe * np.log(mu_safe / nu))

def normalize_likelihoods(likelihoods):
    col_sums = likelihoods.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-300)
    return likelihoods / col_sums

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


# ============================================================
# 决策优化器 - 核心应用类
# ============================================================

class InformationDecisionOptimizer:
    """基于 VoI-EIG 理论的决策优化器
    
    核心功能:
    1. 评估每个候选动作的 VoI 和 EIG
    2. 使用 VoI ≤ √(EIG/2) 进行快速筛选
    3. 使用横截条件进行下界估计
    4. 选择最优信息收集动作
    """
    
    def __init__(self, K, exploit_values, info_likelihoods):
        """
        参数:
            K: 模型数量
            exploit_values: 利用动作的值向量列表,每个向量长度为 K
            info_likelihoods: 信息动作的似然矩阵列表
        """
        self.K = K
        self.C_exploit = exploit_values
        self.info_likelihoods = info_likelihoods
        
    def value_function(self, mu):
        """计算当前信念下的最优利用值"""
        return max(mu @ c for c in self.C_exploit)
    
    def compute_VoI(self, mu, action_idx):
        """计算信息价值 VoI(μ, a)"""
        V_mu = self.value_function(mu)
        likelihoods = self.info_likelihoods[action_idx]
        posts = compute_posterior_distribution(mu, likelihoods)
        E_V = sum(Z * self.value_function(mu_p) for mu_p, Z in posts)
        return E_V - V_mu
    
    def compute_EIG(self, mu, action_idx):
        """计算预期信息增益 EIG(μ, a)"""
        likelihoods = self.info_likelihoods[action_idx]
        posts = compute_posterior_distribution(mu, likelihoods)
        return sum(Z * d_kl(mu_p, mu) for mu_p, Z in posts)
    
    def compute_VoI_upper_bound(self, mu, action_idx):
        """使用定理 3.1: VoI ≤ √(EIG/2)"""
        EIG = self.compute_EIG(mu, action_idx)
        return np.sqrt(EIG / 2)
    
    def check_transversality(self, mu, action_idx, tol=1e-10):
        """检查横截条件是否成立,返回横截常数 m"""
        V_mu = self.value_function(mu)
        likelihoods = self.info_likelihoods[action_idx]
        posts = compute_posterior_distribution(mu, likelihoods)
        
        ratios = []
        for mu_p, Z in posts:
            dtv = d_tv(mu_p, mu)
            if dtv > tol:
                val_diff = self.value_function(mu_p) - V_mu
                if val_diff > 0:
                    ratios.append(val_diff / dtv)
        
        return min(ratios) if ratios else 0.0
    
    def compute_VoI_lower_bound(self, mu, action_idx):
        """使用定理 3.3: VoI ≥ c·EIG (横截条件下)"""
        m = self.check_transversality(mu, action_idx)
        if m <= 0:
            return 0.0
        
        delta = min(mu)
        if delta <= 0:
            return 0.0
        
        EIG = self.compute_EIG(mu, action_idx)
        c = (m * np.sqrt(delta)) / (2 * np.sqrt(np.log(1.0 / delta)))
        return c * EIG
    
    def select_best_action(self, mu, method='exact'):
        """选择最优信息收集动作
        
        方法:
            'exact': 精确计算 VoI 并最大化
            'upper_bound': 使用上界 √(EIG/2) 进行快速筛选
            'lower_bound': 使用下界进行保守估计
        """
        num_actions = len(self.info_likelihoods)
        
        if method == 'exact':
            scores = [self.compute_VoI(mu, a) for a in range(num_actions)]
        elif method == 'upper_bound':
            scores = [self.compute_VoI_upper_bound(mu, a) for a in range(num_actions)]
        elif method == 'lower_bound':
            scores = [self.compute_VoI_lower_bound(mu, a) for a in range(num_actions)]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        best_action = np.argmax(scores)
        return best_action, scores
    
    def evaluate_decision_quality(self, mu, selected_action):
        """评估决策质量: 返回 VoI 和上下界"""
        VoI = self.compute_VoI(mu, selected_action)
        upper = self.compute_VoI_upper_bound(mu, selected_action)
        lower = self.compute_VoI_lower_bound(mu, selected_action)
        EIG = self.compute_EIG(mu, selected_action)
        
        return {
            'VoI': VoI,
            'EIG': EIG,
            'upper_bound': upper,
            'lower_bound': lower,
            'gap_ratio': VoI / upper if upper > 0 else 0,
            'transversality': self.check_transversality(mu, selected_action)
        }


# ============================================================
# 应用示例 1: 传感器选择
# ============================================================

def demo_sensor_selection():
    """传感器选择示例
    
    场景: 一个目标可能位于 K 个区域之一,我们有多个传感器可以选择部署。
    每个传感器提供不同的观测质量(似然),我们需要选择最优的传感器来最大化
    捕获目标的概率。
    """
    print("\n" + "="*70)
    print("应用示例 1: 传感器选择优化")
    print("="*70)
    
    # 设置: 3 个可能的目标位置
    K = 3
    
    # 利用动作: 2 种捕获策略
    # 策略 1: 对区域 1 效果好,对区域 3 效果差
    # 策略 2: 对区域 3 效果好,对区域 1 效果差
    exploit_values = [
        np.array([0.9, 0.5, 0.1]),  # 策略 1
        np.array([0.1, 0.5, 0.9]),  # 策略 2
    ]
    
    # 信息动作: 3 种传感器
    # 传感器 1: 对区域 1 敏感
    # 传感器 2: 对区域 2 敏感
    # 传感器 3: 对区域 3 敏感
    info_likelihoods = [
        normalize_likelihoods(np.array([
            [3.0, 1.0, 0.5],  # 观测 1: 区域 1 概率高
            [0.5, 1.0, 1.5],  # 观测 2: 其他区域
        ])),
        normalize_likelihoods(np.array([
            [1.0, 3.0, 1.0],  # 观测 1: 区域 2 概率高
            [1.0, 0.5, 1.0],  # 观测 2: 其他区域
        ])),
        normalize_likelihoods(np.array([
            [0.5, 1.0, 3.0],  # 观测 1: 区域 3 概率高
            [1.5, 1.0, 0.5],  # 观测 2: 其他区域
        ])),
    ]
    
    # 创建优化器
    optimizer = InformationDecisionOptimizer(K, exploit_values, info_likelihoods)
    
    # 场景 1: 均匀先验
    print("\n--- 场景 1: 均匀先验 μ = [1/3, 1/3, 1/3] ---")
    mu1 = np.array([1/3, 1/3, 1/3])
    
    print(f"当前最优利用值: {optimizer.value_function(mu1):.4f}")
    print("\n各传感器评估:")
    
    for a in range(len(info_likelihoods)):
        metrics = optimizer.evaluate_decision_quality(mu1, a)
        print(f"\n  传感器 {a+1}:")
        print(f"    VoI = {metrics['VoI']:.6f}")
        print(f"    EIG = {metrics['EIG']:.6f}")
        print(f"    上界 √(EIG/2) = {metrics['upper_bound']:.6f}")
        print(f"    下界 c·EIG = {metrics['lower_bound']:.6f}")
        print(f"    实际/上界 = {metrics['gap_ratio']:.4f}")
        print(f"    横截常数 m = {metrics['transversality']:.6f}")
    
    best_action, scores = optimizer.select_best_action(mu1, method='exact')
    print(f"\n  最优选择: 传感器 {best_action+1} (VoI = {scores[best_action]:.6f})")
    
    # 场景 2: 偏向区域 1 的先验
    print("\n--- 场景 2: 偏向区域 1 的先验 μ = [0.6, 0.3, 0.1] ---")
    mu2 = np.array([0.6, 0.3, 0.1])
    
    print(f"当前最优利用值: {optimizer.value_function(mu2):.4f}")
    print("\n各传感器评估:")
    
    for a in range(len(info_likelihoods)):
        metrics = optimizer.evaluate_decision_quality(mu2, a)
        print(f"\n  传感器 {a+1}:")
        print(f"    VoI = {metrics['VoI']:.6f}")
        print(f"    EIG = {metrics['EIG']:.6f}")
    
    best_action, scores = optimizer.select_best_action(mu2, method='exact')
    print(f"\n  最优选择: 传感器 {best_action+1} (VoI = {scores[best_action]:.6f})")
    
    # 场景 3: 决策边界附近
    print("\n--- 场景 3: 决策边界附近 μ = [0.5, 0.0, 0.5] ---")
    mu3 = np.array([0.5, 0.0, 0.5])
    
    print(f"当前最优利用值: {optimizer.value_function(mu3):.4f}")
    print("  (两个利用策略价值相同,位于决策边界)")
    
    print("\n各传感器评估:")
    for a in range(len(info_likelihoods)):
        metrics = optimizer.evaluate_decision_quality(mu3, a)
        print(f"\n  传感器 {a+1}:")
        print(f"    VoI = {metrics['VoI']:.6f}")
        print(f"    EIG = {metrics['EIG']:.6f}")
        print(f"    横截常数 m = {metrics['transversality']:.6f}")
    
    best_action, scores = optimizer.select_best_action(mu3, method='exact')
    print(f"\n  最优选择: 传感器 {best_action+1} (VoI = {scores[best_action]:.6f})")


# ============================================================
# 应用示例 2: 快速筛选算法
# ============================================================

def demo_fast_screening():
    """使用上界进行快速筛选的示例
    
    当有大量候选动作时,精确计算 VoI 可能很昂贵。
    使用上界 √(EIG/2) 可以快速排除低价值的动作。
    """
    print("\n" + "="*70)
    print("应用示例 2: 使用上界进行快速筛选")
    print("="*70)
    
    K = 5
    
    # 利用动作
    exploit_values = [
        np.array([0.9, 0.7, 0.5, 0.3, 0.1]),
        np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
    ]
    
    # 生成大量候选传感器(20个)
    num_candidates = 20
    info_likelihoods = []
    for i in range(num_candidates):
        num_obs = np.random.randint(2, 5)
        L = np.random.uniform(0.1, 5.0, (num_obs, K))
        info_likelihoods.append(normalize_likelihoods(L))
    
    optimizer = InformationDecisionOptimizer(K, exploit_values, info_likelihoods)
    mu = normalize_to_simplex(np.random.uniform(0.1, 1.0, K))
    
    print(f"\n候选动作数量: {num_candidates}")
    print(f"当前信念: {mu}")
    print(f"当前最优利用值: {optimizer.value_function(mu):.4f}")
    
    # 方法 1: 精确计算所有 VoI
    print("\n--- 方法 1: 精确计算 ---")
    import time
    start = time.time()
    best_exact, scores_exact = optimizer.select_best_action(mu, method='exact')
    time_exact = time.time() - start
    
    print(f"  最优动作: {best_exact} (VoI = {scores_exact[best_exact]:.6f})")
    print(f"  计算时间: {time_exact:.4f}s")
    
    # 方法 2: 使用上界快速筛选
    print("\n--- 方法 2: 上界快速筛选 ---")
    start = time.time()
    
    # 第一步: 计算所有上界
    upper_bounds = [optimizer.compute_VoI_upper_bound(mu, a) for a in range(num_candidates)]
    
    # 第二步: 找到上界最高的几个候选
    top_k = 5
    top_candidates = np.argsort(upper_bounds)[-top_k:]
    
    # 第三步: 只对 top_k 精确计算 VoI
    best_filtered = top_candidates[0]
    best_VoI_filtered = optimizer.compute_VoI(mu, best_filtered)
    
    for a in top_candidates[1:]:
        VoI = optimizer.compute_VoI(mu, a)
        if VoI > best_VoI_filtered:
            best_VoI_filtered = VoI
            best_filtered = a
    
    time_filtered = time.time() - start
    
    print(f"  筛选后候选数: {top_k}/{num_candidates}")
    print(f"  最优动作: {best_filtered} (VoI = {best_VoI_filtered:.6f})")
    print(f"  计算时间: {time_filtered:.4f}s")
    print(f"  加速比: {time_exact/time_filtered:.2f}x")
    print(f"  结果一致性: {'✓' if best_exact == best_filtered else '✗'}")


# ============================================================
# 应用示例 3: 预算分配
# ============================================================

def demo_budget_allocation():
    """预算分配示例
    
    使用 VoI/EIG 比率来指导多步决策中的预算分配。
    """
    print("\n" + "="*70)
    print("应用示例 3: 预算分配与多步规划")
    print("="*70)
    
    K = 3
    
    exploit_values = [
        np.array([0.9, 0.5, 0.1]),
        np.array([0.1, 0.5, 0.9]),
    ]
    
    info_likelihoods = [
        normalize_likelihoods(np.array([[3.0, 1.0, 0.5], [0.5, 1.0, 1.5]])),
        normalize_likelihoods(np.array([[0.5, 1.0, 3.0], [1.5, 1.0, 0.5]])),
    ]
    
    optimizer = InformationDecisionOptimizer(K, exploit_values, info_likelihoods)
    
    # 模拟多步决策过程
    print("\n模拟 3 步信息收集过程:")
    
    mu = np.array([1/3, 1/3, 1/3])
    total_VoI = 0
    
    for step in range(3):
        print(f"\n--- 第 {step+1} 步 ---")
        print(f"当前信念: [{mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}]")
        print(f"当前利用值: {optimizer.value_function(mu):.4f}")
        
        # 评估每个动作
        best_action = 0
        best_VoI = -1
        
        for a in range(len(info_likelihoods)):
            VoI = optimizer.compute_VoI(mu, a)
            EIG = optimizer.compute_EIG(mu, a)
            upper = optimizer.compute_VoI_upper_bound(mu, a)
            
            print(f"  动作 {a+1}: VoI={VoI:.6f}, EIG={EIG:.6f}, 上界={upper:.6f}")
            
            if VoI > best_VoI:
                best_VoI = VoI
                best_action = a
        
        print(f"  选择: 动作 {best_action+1}")
        total_VoI += best_VoI
        
        # 模拟观测更新信念
        likelihoods = info_likelihoods[best_action]
        posts = compute_posterior_distribution(mu, likelihoods)
        
        # 选择最可能的后验作为新信念
        max_prob = 0
        new_mu = mu
        for mu_p, Z in posts:
            if Z > max_prob:
                max_prob = Z
                new_mu = mu_p
        
        mu = new_mu
    
    print(f"\n总累积 VoI: {total_VoI:.6f}")
    print(f"最终利用值: {optimizer.value_function(mu):.4f}")


# ============================================================
# 应用示例 4: 预测与置信度估计
# ============================================================

def demo_prediction_confidence():
    """预测与置信度估计
    
    使用 EIG 和 VoI 来估计预测的置信度。
    """
    print("\n" + "="*70)
    print("应用示例 4: 预测置信度估计")
    print("="*70)
    
    K = 3
    
    exploit_values = [
        np.array([0.9, 0.5, 0.1]),
        np.array([0.1, 0.5, 0.9]),
    ]
    
    info_likelihoods = [
        normalize_likelihoods(np.array([[3.0, 1.0, 0.5], [0.5, 1.0, 1.5]])),
        normalize_likelihoods(np.array([[0.5, 1.0, 3.0], [1.5, 1.0, 0.5]])),
    ]
    
    optimizer = InformationDecisionOptimizer(K, exploit_values, info_likelihoods)
    
    print("\n不同信念状态下的预测置信度:")
    
    test_beliefs = [
        np.array([0.9, 0.05, 0.05]),  # 高置信度
        np.array([0.6, 0.3, 0.1]),    # 中等置信度
        np.array([0.4, 0.35, 0.25]),  # 低置信度
        np.array([0.5, 0.0, 0.5]),    # 决策边界
    ]
    
    for i, mu in enumerate(test_beliefs):
        print(f"\n--- 信念 {i+1}: [{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}] ---")
        
        # 当前最优策略
        values = [mu @ c for c in exploit_values]
        best_strategy = np.argmax(values)
        confidence = values[best_strategy] - values[1 - best_strategy]
        
        print(f"  最优策略: 策略 {best_strategy+1}")
        print(f"  策略价值差 (置信度): {confidence:.4f}")
        
        # 信息收集潜力
        max_VoI = max(optimizer.compute_VoI(mu, a) for a in range(len(info_likelihoods)))
        max_EIG = max(optimizer.compute_EIG(mu, a) for a in range(len(info_likelihoods)))
        
        print(f"  最大可获得 VoI: {max_VoI:.6f}")
        print(f"  最大 EIG: {max_EIG:.6f}")
        
        # 信息价值与置信度的关系
        if confidence < 0.1:
            print(f"  状态: 决策边界附近,信息收集价值高")
        elif max_VoI < 0.01:
            print(f"  状态: 已充分知情,无需更多信息")
        else:
            print(f"  状态: 中等不确定性,适度信息收集有益")


# ============================================================
# 主程序
# ============================================================

def main():
    print("\n" + "="*70)
    print("值信息对齐理论的实际应用演示")
    print("="*70)
    
    demo_sensor_selection()
    demo_fast_screening()
    demo_budget_allocation()
    demo_prediction_confidence()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    main()
