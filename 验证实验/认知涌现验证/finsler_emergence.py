"""
Finsler度量涌现与稳定范数计算模块
验证定理4：认知几何涌现定理

核心验证内容：
1. 稳定范数（Mather β-函数）的计算
2. Finsler度量的涌现验证
3. 各向异性与不可逆性验证
4. 基本张量的计算
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FinslerConfig:
    """Finsler度量配置"""
    dimension: int
    num_directions: int
    time_horizon: float
    num_paths: int


class StableNorm:
    """
    稳定范数（Mather β-函数）
    
    β(v) = lim_{t→∞} (1/t) inf_{γ(0)=0, γ(t)=tv} ∫_0^t L(γ, γ̇) ds
    
    这是从微观动力学中涌现出的宏观几何结构
    """
    
    def __init__(self, lagrangian: Callable, config: FinslerConfig):
        self.L = lagrangian
        self.config = config
        self.d = config.dimension
        
    def compute(self, v: np.ndarray, time_scale: float = 10.0) -> float:
        """
        计算稳定范数 β(v)
        
        使用长时间平均近似
        """
        t = time_scale
        target = t * v
        
        num_points = 20
        dt = t / num_points
        
        action = 0.0
        for i in range(num_points):
            alpha = i / num_points
            point = alpha * target
            L_val = self.L(point, v)
            if np.isfinite(L_val):
                action += L_val * dt
            else:
                action = 1e6
                break
        
        return action / t if np.isfinite(action) else 1e6
    
    def compute_on_sphere(self, 
                         num_directions: int = 30,
                         time_scale: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算单位球面上的稳定范数
        
        返回方向向量和对应的稳定范数值
        """
        directions = []
        beta_values = []
        
        for i in range(num_directions):
            theta = 2 * np.pi * i / num_directions
            v = np.array([np.cos(theta), np.sin(theta)])
            
            beta_v = self.compute(v, time_scale)
            
            directions.append(v)
            beta_values.append(beta_v)
        
        return np.array(directions), np.array(beta_values)


class FinslerMetric:
    """
    Finsler度量
    
    F(x, v) = β(v) + α(x, v)
    
    其中β是稳定范数，α是修正项
    """
    
    def __init__(self, 
                 stable_norm: StableNorm,
                 correction: Optional[Callable] = None,
                 config: FinslerConfig = None):
        self.beta = stable_norm
        self.alpha = correction if correction else lambda x, v: 0.0
        self.config = config if config else FinslerConfig(2, 30, 10.0, 50)
        
    def compute(self, x: np.ndarray, v: np.ndarray) -> float:
        """
        计算Finsler度量 F(x, v)
        """
        beta_v = self.beta.compute(v)
        alpha_val = self.alpha(x, v)
        
        return beta_v + alpha_val
    
    def compute_fundamental_tensor(self, 
                                  x: np.ndarray, 
                                  v: np.ndarray,
                                  epsilon: float = 0.01) -> np.ndarray:
        """
        计算基本张量 g_{ij}(x, v)
        
        g_{ij} = (1/2) ∂²F²/∂v^i∂v^j
        """
        d = len(v)
        g = np.zeros((d, d))
        
        F0 = self.compute(x, v)
        F0_sq = F0**2
        
        for i in range(d):
            for j in range(d):
                v_pp = v.copy()
                v_pp[i] += epsilon
                v_pp[j] += epsilon
                
                v_pm = v.copy()
                v_pm[i] += epsilon
                v_pm[j] -= epsilon
                
                v_mp = v.copy()
                v_mp[i] -= epsilon
                v_mp[j] += epsilon
                
                v_mm = v.copy()
                v_mm[i] -= epsilon
                v_mm[j] -= epsilon
                
                F_pp = self.compute(x, v_pp)
                F_pm = self.compute(x, v_pm)
                F_mp = self.compute(x, v_mp)
                F_mm = self.compute(x, v_mm)
                
                g[i, j] = 0.25 * (F_pp**2 - F_pm**2 - F_mp**2 + F_mm**2) / epsilon**2
        
        return 0.5 * g
    
    def check_positive_definite(self, 
                               x: np.ndarray,
                               v: np.ndarray) -> Dict:
        """
        检查基本张量的正定性
        """
        g = self.compute_fundamental_tensor(x, v)
        
        eigenvalues = np.linalg.eigvalsh(g)
        
        is_positive = all(eig > 0 for eig in eigenvalues)
        
        return {
            'tensor': g,
            'eigenvalues': eigenvalues,
            'is_positive_definite': is_positive,
            'min_eigenvalue': min(eigenvalues),
            'condition_number': max(eigenvalues) / max(min(eigenvalues), 1e-10)
        }


class AnisotropyAnalyzer:
    """
    各向异性分析器
    
    分析Finsler度量的各向异性程度
    """
    
    def __init__(self, finsler_metric: FinslerMetric):
        self.F = finsler_metric
        
    def compute_anisotropy_index(self, 
                                x: np.ndarray,
                                num_directions: int = 30) -> Dict:
        """
        计算各向异性指数
        
        度量各向异性程度的无量纲指标
        """
        F_values = []
        
        for i in range(num_directions):
            theta = 2 * np.pi * i / num_directions
            v = np.array([np.cos(theta), np.sin(theta)])
            
            F_val = self.F.compute(x, v)
            F_values.append(F_val)
        
        F_values = np.array(F_values)
        
        F_max = np.max(F_values)
        F_min = np.min(F_values)
        F_mean = np.mean(F_values)
        
        anisotropy_index = (F_max - F_min) / F_mean if F_mean > 0 else 0.0
        
        return {
            'F_values': F_values,
            'F_max': F_max,
            'F_min': F_min,
            'F_mean': F_mean,
            'anisotropy_index': anisotropy_index,
            'is_isotropic': anisotropy_index < 0.1
        }
    
    def compute_irreversibility(self,
                               x: np.ndarray,
                               num_directions: int = 30) -> Dict:
        """
        计算不可逆性
        
        分析 F(x, v) 与 F(x, -v) 的差异
        """
        asymmetries = []
        
        for i in range(num_directions):
            theta = 2 * np.pi * i / num_directions
            v = np.array([np.cos(theta), np.sin(theta)])
            
            F_forward = self.F.compute(x, v)
            F_backward = self.F.compute(x, -v)
            
            if F_forward > 0 and F_backward > 0:
                asymmetry = abs(F_forward - F_backward) / max(F_forward, F_backward)
                asymmetries.append(asymmetry)
        
        if not asymmetries:
            return {
                'mean_asymmetry': 0.0,
                'max_asymmetry': 0.0,
                'is_reversible': True
            }
        
        return {
            'mean_asymmetry': float(np.mean(asymmetries)),
            'max_asymmetry': float(np.max(asymmetries)),
            'is_reversible': np.mean(asymmetries) < 0.1
        }


def verify_theorem4_finsler_emergence():
    """
    验证定理4：认知几何涌现定理
    
    验证内容：
    1. 稳定范数的计算
    2. Finsler度量的涌现
    3. 基本张量的正定性
    4. 各向异性与不可逆性
    """
    print("=" * 60)
    print("验证定理4：认知几何涌现定理")
    print("=" * 60)
    
    def lagrangian(x, v):
        v_norm = np.linalg.norm(v)
        x_norm = np.linalg.norm(x)
        return 0.5 * v_norm**2 + 0.5 * x_norm**2
    
    config = FinslerConfig(
        dimension=2,
        num_directions=30,
        time_horizon=10.0,
        num_paths=50
    )
    
    stable_norm = StableNorm(lagrangian, config)
    
    print("\n计算稳定范数...")
    test_velocities = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2),
        np.array([1.0, -1.0]) / np.sqrt(2)
    ]
    
    print("测试速度向量的稳定范数:")
    for i, v in enumerate(test_velocities):
        beta_v = stable_norm.compute(v, time_scale=10.0)
        print(f"  v_{i+1} = {v}: β(v) = {beta_v:.6f}")
    
    print("\n计算单位球面上的稳定范数...")
    directions, beta_values = stable_norm.compute_on_sphere(
        num_directions=30, 
        time_scale=10.0
    )
    
    print(f"方向数: {len(directions)}")
    print(f"β值范围: [{np.min(beta_values):.6f}, {np.max(beta_values):.6f}]")
    print(f"β值均值: {np.mean(beta_values):.6f}")
    
    def correction(x, v):
        return 0.1 * np.dot(x, v)
    
    finsler = FinslerMetric(stable_norm, correction, config)
    
    print("\n计算Finsler度量...")
    x_test = np.array([0.5, 0.5])
    v_test = np.array([1.0, 0.5])
    
    F_val = finsler.compute(x_test, v_test)
    print(f"F(x={x_test}, v={v_test}) = {F_val:.6f}")
    
    print("\n计算基本张量...")
    g = finsler.compute_fundamental_tensor(x_test, v_test, epsilon=0.01)
    
    print("基本张量 g_{ij}:")
    print(f"  g_11 = {g[0, 0]:.6f}, g_12 = {g[0, 1]:.6f}")
    print(f"  g_21 = {g[1, 0]:.6f}, g_22 = {g[1, 1]:.6f}")
    
    pd_check = finsler.check_positive_definite(x_test, v_test)
    
    print(f"\n正定性检查:")
    print(f"  特征值: {pd_check['eigenvalues']}")
    print(f"  是否正定: {pd_check['is_positive_definite']}")
    print(f"  条件数: {pd_check['condition_number']:.6f}")
    
    aniso = AnisotropyAnalyzer(finsler)
    
    print("\n分析各向异性...")
    aniso_results = aniso.compute_anisotropy_index(x_test, num_directions=30)
    
    print(f"各向异性指数: {aniso_results['anisotropy_index']:.6f}")
    print(f"F_max: {aniso_results['F_max']:.6f}")
    print(f"F_min: {aniso_results['F_min']:.6f}")
    print(f"是否各向同性: {aniso_results['is_isotropic']}")
    
    print("\n分析不可逆性...")
    irrev_results = aniso.compute_irreversibility(x_test, num_directions=30)
    
    print(f"平均不对称性: {irrev_results['mean_asymmetry']:.6f}")
    print(f"最大不对称性: {irrev_results['max_asymmetry']:.6f}")
    print(f"是否可逆: {irrev_results['is_reversible']}")
    
    return {
        'stable_norm_values': beta_values,
        'fundamental_tensor': g,
        'is_positive_definite': pd_check['is_positive_definite'],
        'anisotropy_index': aniso_results['anisotropy_index'],
        'irreversibility': irrev_results['mean_asymmetry'],
        'converged': pd_check['is_positive_definite'] and aniso_results['anisotropy_index'] < 1.0
    }


if __name__ == "__main__":
    results = verify_theorem4_finsler_emergence()
    
    print("\n" + "=" * 60)
    print("定理4验证结果汇总:")
    print("=" * 60)
    print(f"基本张量正定性: {'是' if results['is_positive_definite'] else '否'}")
    print(f"各向异性指数: {results['anisotropy_index']:.6f}")
    print(f"不可逆性指标: {results['irreversibility']:.6f}")
    print(f"\n整体验证: {'通过' if results['converged'] else '未通过'}")
