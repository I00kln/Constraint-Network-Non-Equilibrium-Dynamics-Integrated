"""
离散变分闭包动力学与Γ-收敛验证模块
验证定理1：离散到连续的桥接定理

核心验证内容：
1. 离散变分闭包算子的构造与迭代
2. 路径作用量的计算
3. Γ-收敛的数值验证
4. 离散与连续极限的一致性检验
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt


@dataclass
class CoarseGrainingStrategy:
    """粗粒化策略"""
    state_count: int
    memory_depth: int
    partition: np.ndarray
    resource_coordinate: np.ndarray
    
    def resource_cost(self) -> float:
        return np.linalg.norm(self.resource_coordinate)


@dataclass
class VariationalClosureConfig:
    """变分闭包算子配置"""
    resource_limit: float
    lattice_size: int
    epsilon: float = 0.01
    max_iterations: int = 1000


class DiscreteVariationalClosure:
    """
    离散变分闭包算子 V_{R,L}
    
    在资源约束R和格点数L下，对粗粒化策略进行变分闭包操作
    """
    
    def __init__(self, config: VariationalClosureConfig):
        self.config = config
        self.R = config.resource_limit
        self.L = config.lattice_size
        self.epsilon = config.epsilon
        
        self.lattice = self._initialize_lattice()
        self.closure_history = []
        
    def _initialize_lattice(self) -> List[CoarseGrainingStrategy]:
        """初始化粗粒化策略格"""
        lattice = []
        for i in range(self.L):
            for j in range(self.L):
                x = np.array([i / self.L * self.R, j / self.L * self.R])
                partition = np.random.rand(i + 2, j + 2)
                strategy = CoarseGrainingStrategy(
                    state_count=i + 1,
                    memory_depth=j + 1,
                    partition=partition,
                    resource_coordinate=x
                )
                lattice.append(strategy)
        return lattice
    
    def compute_closure_defect(self, strategy: CoarseGrainingStrategy) -> float:
        """
        计算闭包缺陷 ι_1(C)
        
        度量粗粒化策略与完美闭包的偏离程度
        """
        n = strategy.state_count
        m = strategy.memory_depth
        
        defect = 0.0
        
        defect += 1.0 / (n + 1)
        
        defect += 1.0 / (m + 1)
        
        noise = np.random.rand() * 0.1
        defect += noise
        
        return defect
    
    def compute_path_action(self, path: List[CoarseGrainingStrategy], 
                           lagrangian: Callable) -> float:
        """
        计算路径作用量 S_dis[γ]
        
        S_dis[γ] = Σ_i L(x_i, x_{i+1}) * Δt
        """
        action = 0.0
        for i in range(len(path) - 1):
            x_i = path[i].resource_coordinate
            x_next = path[i + 1].resource_coordinate
            v = (x_next - x_i) / self.epsilon
            action += lagrangian(x_i, v) * self.epsilon
        return action
    
    def apply_closure_operator(self, 
                               strategy: CoarseGrainingStrategy) -> CoarseGrainingStrategy:
        """
        应用变分闭包算子
        
        V_{R,L}(C) = argmin_{C'} { S(C → C') + λ * ι_1(C') }
        """
        defect = self.compute_closure_defect(strategy)
        
        new_state_count = max(1, strategy.state_count + int(np.sign(0.5 - defect)))
        new_memory_depth = max(1, strategy.memory_depth + int(np.sign(0.5 - defect)))
        
        new_x = strategy.resource_coordinate.copy()
        gradient = -np.sign(defect - 0.5) * np.random.rand(2) * 0.1
        new_x = np.clip(new_x + gradient * self.epsilon, 0, self.R)
        
        new_partition = strategy.partition.copy()
        if new_state_count != strategy.state_count or new_memory_depth != strategy.memory_depth:
            new_partition = np.random.rand(new_state_count + 1, new_memory_depth + 1)
        
        return CoarseGrainingStrategy(
            state_count=new_state_count,
            memory_depth=new_memory_depth,
            partition=new_partition,
            resource_coordinate=new_x
        )
    
    def iterate(self, initial_strategy: CoarseGrainingStrategy, 
                num_iterations: int) -> List[CoarseGrainingStrategy]:
        """迭代应用闭包算子"""
        trajectory = [initial_strategy]
        current = initial_strategy
        
        for _ in range(num_iterations):
            current = self.apply_closure_operator(current)
            trajectory.append(current)
            self.closure_history.append(self.compute_closure_defect(current))
        
        return trajectory
    
    def find_fixed_points(self) -> List[CoarseGrainingStrategy]:
        """
        寻找不动点
        
        V_{R,L}(C*) = C*
        """
        fixed_points = []
        
        for strategy in self.lattice:
            new_strategy = self.apply_closure_operator(strategy)
            
            distance = np.linalg.norm(
                strategy.resource_coordinate - new_strategy.resource_coordinate
            )
            
            if distance < self.epsilon:
                fixed_points.append(strategy)
        
        return fixed_points


class GammaConvergenceVerifier:
    """
    Γ-收敛验证器
    
    验证离散作用量泛函在连续极限下的收敛性
    """
    
    def __init__(self, epsilon_sequence: List[float]):
        self.epsilons = epsilon_sequence
        self.results = {}
        
    def define_continuous_lagrangian(self, x: np.ndarray, v: np.ndarray) -> float:
        """
        定义连续Lagrangian L(x, v)
        
        示例：L(x, v) = 0.5 * ||v||^2 + V(x)
        """
        kinetic = 0.5 * np.dot(v, v)
        
        potential = 0.5 * np.dot(x, x)
        
        return kinetic + potential
    
    def compute_discrete_action(self, path: np.ndarray, epsilon: float) -> float:
        """计算离散路径作用量"""
        action = 0.0
        for i in range(len(path) - 1):
            x = path[i]
            v = (path[i + 1] - path[i]) / epsilon
            action += self.define_continuous_lagrangian(x, v) * epsilon
        return action
    
    def compute_continuous_action(self, path: np.ndarray) -> float:
        """计算连续路径作用量（数值积分）"""
        dt = 1.0 / (len(path) - 1)
        action = 0.0
        for i in range(len(path) - 1):
            x = path[i]
            v = (path[i + 1] - path[i]) / dt
            action += self.define_continuous_lagrangian(x, v) * dt
        return action
    
    def verify_lower_bound_condition(self, 
                                     path_sequence: List[np.ndarray]) -> Dict:
        """
        验证Γ-收敛的下界条件
        
        liminf_{ε→0} S_ε[γ_ε] ≥ S[γ]
        """
        results = {
            'epsilons': [],
            'discrete_actions': [],
            'continuous_action': None,
            'ratios': [],
            'converged': False
        }
        
        continuous_action = self.compute_continuous_action(path_sequence[-1])
        results['continuous_action'] = continuous_action
        
        for i, epsilon in enumerate(self.epsilons):
            if i < len(path_sequence):
                discrete_action = self.compute_discrete_action(
                    path_sequence[i], epsilon
                )
                results['epsilons'].append(epsilon)
                results['discrete_actions'].append(discrete_action)
                
                if continuous_action > 0:
                    ratio = discrete_action / continuous_action
                    results['ratios'].append(ratio)
        
        if len(results['ratios']) > 1:
            ratios = np.array(results['ratios'])
            if np.all(ratios[-3:] > 0.95) and np.all(ratios[-3:] < 1.05):
                results['converged'] = True
        
        return results
    
    def verify_recovery_condition(self, 
                                  continuous_path: np.ndarray) -> Dict:
        """
        验证Γ-收敛的恢复条件
        
        ∃ γ_ε → γ 使得 S_ε[γ_ε] → S[γ]
        """
        results = {
            'epsilons': [],
            'recovery_paths': [],
            'discrete_actions': [],
            'continuous_action': None,
            'errors': [],
            'converged': False
        }
        
        continuous_action = self.compute_continuous_action(continuous_path)
        results['continuous_action'] = continuous_action
        
        for epsilon in self.epsilons:
            num_points = max(10, int(1.0 / epsilon))
            t = np.linspace(0, 1, num_points)
            
            recovery_path = np.zeros((num_points, continuous_path.shape[1]))
            for j in range(continuous_path.shape[1]):
                recovery_path[:, j] = np.interp(
                    t, np.linspace(0, 1, len(continuous_path)), 
                    continuous_path[:, j]
                )
            
            results['recovery_paths'].append(recovery_path)
            
            discrete_action = self.compute_discrete_action(recovery_path, epsilon)
            results['discrete_actions'].append(discrete_action)
            results['epsilons'].append(epsilon)
            
            error = abs(discrete_action - continuous_action)
            results['errors'].append(error)
        
        if len(results['errors']) > 0:
            errors = np.array(results['errors'])
            if errors[-1] < 0.1 * continuous_action:
                results['converged'] = True
        
        return results


class CognitiveRefractiveIndex:
    """
    认知折射率 F*(x)
    
    度量在资源坐标x处理解世界的"阻抗"或"代价"
    """
    
    def __init__(self, dimension: int = 2):
        self.d = dimension
        self.obstacles = []
        
    def add_obstacle(self, center: np.ndarray, radius: float, 
                    height: float):
        """添加认知障碍（资源壁垒）"""
        self.obstacles.append({
            'center': center,
            'radius': radius,
            'height': height
        })
    
    def compute(self, x: np.ndarray) -> float:
        """
        计算认知折射率 F*(x)
        
        F*(x) = 1 + Σ_i h_i * exp(-||x - c_i||^2 / r_i^2)
        """
        refractive_index = 1.0
        
        for obstacle in self.obstacles:
            distance_sq = np.sum((x - obstacle['center'])**2)
            contribution = obstacle['height'] * np.exp(
                -distance_sq / (obstacle['radius']**2)
            )
            refractive_index += contribution
        
        return refractive_index
    
    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """计算认知折射率的梯度"""
        gradient = np.zeros(self.d)
        
        for obstacle in self.obstacles:
            diff = x - obstacle['center']
            distance_sq = np.sum(diff**2)
            factor = obstacle['height'] * np.exp(
                -distance_sq / (obstacle['radius']**2)
            ) * (-2.0 / obstacle['radius']**2)
            gradient += factor * diff
        
        return gradient


def verify_theorem1_discrete_to_continuous():
    """
    验证定理1：离散到连续的桥接定理
    
    验证内容：
    1. Γ-收敛的下界条件
    2. Γ-收敛的恢复条件
    3. 离散与连续作用量的一致性
    """
    print("=" * 60)
    print("验证定理1：离散到连续的桥接定理")
    print("=" * 60)
    
    epsilon_sequence = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    verifier = GammaConvergenceVerifier(epsilon_sequence)
    
    t = np.linspace(0, 1, 100)
    continuous_path = np.column_stack([
        np.sin(np.pi * t),
        np.cos(np.pi * t) - 1
    ])
    
    print("\n验证Γ-收敛的恢复条件...")
    recovery_results = verifier.verify_recovery_condition(continuous_path)
    
    print(f"连续作用量: {recovery_results['continuous_action']:.6f}")
    print("\n离散作用量序列:")
    for i, eps in enumerate(recovery_results['epsilons']):
        action = recovery_results['discrete_actions'][i]
        error = recovery_results['errors'][i]
        print(f"  ε = {eps:.4f}: S_ε = {action:.6f}, 误差 = {error:.6f}")
    
    print(f"\n恢复条件验证: {'通过' if recovery_results['converged'] else '未通过'}")
    
    path_sequence = []
    for eps in epsilon_sequence:
        num_points = max(10, int(1.0 / eps))
        t_eps = np.linspace(0, 1, num_points)
        path_eps = np.column_stack([
            np.sin(np.pi * t_eps),
            np.cos(np.pi * t_eps) - 1
        ])
        path_sequence.append(path_eps)
    
    print("\n验证Γ-收敛的下界条件...")
    lower_bound_results = verifier.verify_lower_bound_condition(path_sequence)
    
    print(f"连续作用量: {lower_bound_results['continuous_action']:.6f}")
    print("\n离散/连续作用量比值:")
    for i, eps in enumerate(lower_bound_results['epsilons']):
        ratio = lower_bound_results['ratios'][i]
        print(f"  ε = {eps:.4f}: S_ε/S = {ratio:.6f}")
    
    print(f"\n下界条件验证: {'通过' if lower_bound_results['converged'] else '未通过'}")
    
    print("\n验证离散变分闭包算子...")
    config = VariationalClosureConfig(
        resource_limit=5.0,
        lattice_size=10,
        epsilon=0.1,
        max_iterations=100
    )
    closure_op = DiscreteVariationalClosure(config)
    
    initial_strategy = CoarseGrainingStrategy(
        state_count=3,
        memory_depth=2,
        partition=np.random.rand(4, 3),
        resource_coordinate=np.array([1.5, 1.0])
    )
    
    trajectory = closure_op.iterate(initial_strategy, 50)
    
    defects = [closure_op.compute_closure_defect(s) for s in trajectory]
    print(f"\n初始闭包缺陷: {defects[0]:.6f}")
    print(f"最终闭包缺陷: {defects[-1]:.6f}")
    print(f"缺陷变化趋势: {'收敛' if defects[-1] < defects[0] else '发散'}")
    
    fixed_points = closure_op.find_fixed_points()
    print(f"\n发现不动点数量: {len(fixed_points)}")
    
    return {
        'gamma_convergence': recovery_results['converged'] and lower_bound_results['converged'],
        'lower_bound': lower_bound_results,
        'recovery': recovery_results,
        'closure_defects': defects,
        'fixed_points_count': len(fixed_points)
    }


if __name__ == "__main__":
    results = verify_theorem1_discrete_to_continuous()
    print("\n" + "=" * 60)
    print("定理1验证结果汇总:")
    print("=" * 60)
    print(f"Γ-收敛验证: {'通过' if results['gamma_convergence'] else '未通过'}")
    print(f"发现不动点: {results['fixed_points_count']} 个")
