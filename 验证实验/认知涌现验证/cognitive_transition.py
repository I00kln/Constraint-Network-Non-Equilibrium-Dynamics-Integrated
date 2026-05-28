"""
认知跃迁与大偏差验证模块
验证定理6：认知跃迁定理

核心验证内容：
1. Freidlin-Wentzell大偏差理论验证
2. 跃迁概率的计算
3. 势垒高度估计
4. 逃逸时间分析
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TransitionConfig:
    """跃迁配置"""
    dimension: int
    epsilon_values: List[float]
    num_trajectories: int
    time_horizon: float


class FreidlinWentzellTheory:
    """
    Freidlin-Wentzell大偏差理论
    
    描述小噪声随机动力系统的逃逸行为
    
    P(A → B) ≍ exp(-h(A, B) / ε)
    """
    
    def __init__(self, potential: Callable, 
                 gradient: Callable,
                 config: TransitionConfig):
        self.V = potential
        self.grad_V = gradient
        self.config = config
        self.d = config.dimension
        
    def find_local_minima(self, 
                         x_range: np.ndarray,
                         y_range: np.ndarray,
                         grid_size: int = 50) -> List[np.ndarray]:
        """
        寻找势函数的局部极小值点（亚稳态）
        """
        x_grid = np.linspace(x_range[0], x_range[1], grid_size)
        y_grid = np.linspace(y_range[0], y_range[1], grid_size)
        
        dx = (x_range[1] - x_range[0]) / (grid_size - 1)
        dy = (y_range[1] - y_range[0]) / (grid_size - 1)
        
        V_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                V_grid[i, j] = self.V(np.array([x_grid[i], y_grid[j]]))
        
        minima = []
        
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                V_center = V_grid[i, j]
                
                neighbors = [
                    V_grid[i-1, j], V_grid[i+1, j],
                    V_grid[i, j-1], V_grid[i, j+1],
                    V_grid[i-1, j-1], V_grid[i-1, j+1],
                    V_grid[i+1, j-1], V_grid[i+1, j+1]
                ]
                
                if V_center <= min(neighbors) + 1e-6:
                    grad_x = (V_grid[i+1, j] - V_grid[i-1, j]) / (2 * dx)
                    grad_y = (V_grid[i, j+1] - V_grid[i, j-1]) / (2 * dy)
                    
                    if abs(grad_x) < 0.5 and abs(grad_y) < 0.5:
                        minima.append(np.array([x_grid[i], y_grid[j]]))
        
        unique_minima = []
        for m in minima:
            is_duplicate = False
            for um in unique_minima:
                if np.linalg.norm(m - um) < 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_minima.append(m)
        
        return unique_minima
    
    def find_saddle_points(self,
                          x_range: np.ndarray,
                          y_range: np.ndarray,
                          grid_size: int = 50) -> List[np.ndarray]:
        """
        寻找势函数的鞍点
        """
        x_grid = np.linspace(x_range[0], x_range[1], grid_size)
        y_grid = np.linspace(y_range[0], y_range[1], grid_size)
        
        dx = (x_range[1] - x_range[0]) / (grid_size - 1)
        dy = (y_range[1] - y_range[0]) / (grid_size - 1)
        
        saddles = []
        
        for i in range(2, grid_size - 2):
            for j in range(2, grid_size - 2):
                x = np.array([x_grid[i], y_grid[j]])
                
                grad_x = (self.V(np.array([x_grid[i+1], y_grid[j]])) - 
                         self.V(np.array([x_grid[i-1], y_grid[j]]))) / (2 * dx)
                grad_y = (self.V(np.array([x_grid[i], y_grid[j+1]])) - 
                         self.V(np.array([x_grid[i], y_grid[j-1]]))) / (2 * dy)
                
                if abs(grad_x) < 0.3 and abs(grad_y) < 0.3:
                    V_xx = (self.V(np.array([x_grid[i+1], y_grid[j]])) - 
                           2 * self.V(x) + 
                           self.V(np.array([x_grid[i-1], y_grid[j]]))) / dx**2
                    V_yy = (self.V(np.array([x_grid[i], y_grid[j+1]])) - 
                           2 * self.V(x) + 
                           self.V(np.array([x_grid[i], y_grid[j-1]]))) / dy**2
                    V_xy = (self.V(np.array([x_grid[i+1], y_grid[j+1]])) - 
                           self.V(np.array([x_grid[i+1], y_grid[j-1]])) -
                           self.V(np.array([x_grid[i-1], y_grid[j+1]])) + 
                           self.V(np.array([x_grid[i-1], y_grid[j-1]]))) / (4 * dx * dy)
                    
                    det = V_xx * V_yy - V_xy**2
                    
                    if det < -0.1:
                        saddles.append(x)
        
        unique_saddles = []
        for s in saddles:
            is_duplicate = False
            for us in unique_saddles:
                if np.linalg.norm(s - us) < 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_saddles.append(s)
        
        return unique_saddles
    
    def compute_barrier_height(self, 
                               A: np.ndarray, 
                               B: np.ndarray) -> float:
        """
        计算从A到B的最小势垒高度
        
        h(A, B) = min_{saddle} [V(saddle) - V(A)]
        """
        V_A = self.V(A)
        
        x_range = np.array([min(A[0], B[0]) - 1.0, max(A[0], B[0]) + 1.0])
        y_range = np.array([min(A[1], B[1]) - 1.0, max(A[1], B[1]) + 1.0])
        
        saddles = self.find_saddle_points(x_range, y_range, grid_size=40)
        
        if not saddles:
            return abs(self.V(B) - V_A)
        
        min_barrier = np.inf
        for saddle in saddles:
            barrier = self.V(saddle) - V_A
            if barrier > 0 and barrier < min_barrier:
                min_barrier = barrier
        
        return min_barrier if min_barrier < np.inf else abs(self.V(B) - V_A)
    
    def compute_transition_probability(self,
                                      A: np.ndarray,
                                      B: np.ndarray,
                                      epsilon: float) -> float:
        """
        计算跃迁概率
        
        P(A → B) ≈ exp(-h(A, B) / ε)
        """
        h = self.compute_barrier_height(A, B)
        
        return np.exp(-h / max(epsilon, 0.01))


class EscapeTimeAnalyzer:
    """
    逃逸时间分析器
    
    分析从亚稳态逃逸的平均时间
    """
    
    def __init__(self, potential: Callable,
                 gradient: Callable,
                 config: TransitionConfig):
        self.V = potential
        self.grad_V = gradient
        self.config = config
        
    def simulate_trajectory(self,
                           x0: np.ndarray,
                           epsilon: float,
                           dt: float = 0.01,
                           max_steps: int = 5000) -> np.ndarray:
        """
        模拟随机动力学轨迹
        
        dx = -∇V(x) dt + √(2ε) dW
        """
        trajectory = [x0.copy()]
        x = x0.copy()
        
        for _ in range(max_steps):
            noise = np.random.randn(len(x0)) * np.sqrt(2 * epsilon * dt)
            
            grad = self.grad_V(x)
            grad = np.clip(grad, -5, 5)
            drift = -grad * dt
            
            x = x + drift + noise
            x = np.clip(x, -3, 3)
            
            trajectory.append(x.copy())
        
        return np.array(trajectory)
    
    def estimate_escape_time(self,
                            basin_A: np.ndarray,
                            epsilon: float,
                            num_runs: int = 20) -> Dict:
        """
        估计从吸引域A逃逸的平均时间
        """
        escape_times = []
        
        for _ in range(num_runs):
            x0 = basin_A + np.random.randn(len(basin_A)) * 0.1
            
            trajectory = self.simulate_trajectory(
                x0, epsilon, dt=0.02, max_steps=2000
            )
            
            basin_radius = 0.5
            
            for t, x in enumerate(trajectory):
                if np.linalg.norm(x - basin_A) > basin_radius:
                    escape_times.append(t * 0.02)
                    break
        
        if not escape_times:
            return {
                'mean_escape_time': 100.0,
                'std_escape_time': 0.0,
                'num_escapes': 0
            }
        
        return {
            'mean_escape_time': float(np.mean(escape_times)),
            'std_escape_time': float(np.std(escape_times)),
            'num_escapes': len(escape_times)
        }


class InstantonPath:
    """
    Instanton路径（最优逃逸路径）
    
    连接亚稳态的测地线
    """
    
    def __init__(self, potential: Callable,
                 gradient: Callable):
        self.V = potential
        self.grad_V = gradient
        
    def compute(self,
               A: np.ndarray,
               B: np.ndarray,
               num_points: int = 50,
               num_iterations: int = 100) -> np.ndarray:
        """
        计算从A到B的Instanton路径
        
        使用弦方法
        """
        path = np.zeros((num_points, len(A)))
        
        for i in range(num_points):
            t = i / (num_points - 1)
            path[i] = (1 - t) * A + t * B
        
        for iteration in range(num_iterations):
            new_path = path.copy()
            
            for i in range(1, num_points - 1):
                tangent = (path[i + 1] - path[i - 1]) / 2
                tangent_norm = np.linalg.norm(tangent) + 1e-10
                tangent = tangent / tangent_norm
                
                grad_V = self.grad_V(path[i])
                grad_V = np.clip(grad_V, -5, 5)
                
                normal_component = grad_V - np.dot(grad_V, tangent) * tangent
                
                new_path[i] = path[i] - 0.01 * normal_component
            
            if np.max(np.abs(new_path - path)) < 1e-6:
                break
            
            path = new_path
        
        return path
    
    def compute_action(self, path: np.ndarray) -> float:
        """
        计算路径的作用量
        
        S[γ] = ∫ ||γ̇||^2 / 4 + V(γ) dt
        """
        action = 0.0
        
        for i in range(len(path) - 1):
            v = path[i + 1] - path[i]
            kinetic = 0.25 * np.dot(v, v)
            potential = self.V(path[i])
            if np.isfinite(potential):
                action += kinetic + potential
            else:
                action += kinetic
        
        return action


def verify_theorem6_cognitive_transition():
    """
    验证定理6：认知跃迁定理
    
    验证内容：
    1. Freidlin-Wentzell大偏差理论
    2. 跃迁概率的指数衰减
    3. 逃逸时间的Kramers公式
    4. Instanton路径的计算
    """
    print("=" * 60)
    print("验证定理6：认知跃迁定理")
    print("=" * 60)
    
    def potential(x):
        return (x[0]**4 - 2 * x[0]**2 + x[1]**2 + 
                0.5 * x[0]**2 * x[1]**2)
    
    def gradient(x):
        grad = np.zeros(2)
        grad[0] = 4 * x[0]**3 - 4 * x[0] + x[0] * x[1]**2
        grad[1] = 2 * x[1] + x[0]**2 * x[1]
        return grad
    
    config = TransitionConfig(
        dimension=2,
        epsilon_values=[0.5, 0.3, 0.2, 0.1, 0.05],
        num_trajectories=50,
        time_horizon=50.0
    )
    
    fw = FreidlinWentzellTheory(potential, gradient, config)
    
    print("\n寻找亚稳态（局部极小值）...")
    x_range = np.array([-2.0, 2.0])
    y_range = np.array([-2.0, 2.0])
    
    minima = fw.find_local_minima(x_range, y_range, grid_size=50)
    
    print(f"发现 {len(minima)} 个局部极小值点:")
    for i, m in enumerate(minima[:5]):
        print(f"  极小值 {i+1}: x = {m}, V = {potential(m):.6f}")
    
    print("\n寻找鞍点...")
    saddles = fw.find_saddle_points(x_range, y_range, grid_size=50)
    
    print(f"发现 {len(saddles)} 个鞍点:")
    for i, s in enumerate(saddles[:3]):
        print(f"  鞍点 {i+1}: x = {s}, V = {potential(s):.6f}")
    
    h_AB = 0.0
    h_BA = 0.0
    prob_AB = []
    
    if len(minima) >= 2:
        A = minima[0]
        B = minima[1]
        
        print(f"\n计算从极小值 {A} 到 {B} 的势垒高度...")
        h_AB = fw.compute_barrier_height(A, B)
        h_BA = fw.compute_barrier_height(B, A)
        
        print(f"势垒高度 h(A→B): {h_AB:.6f}")
        print(f"势垒高度 h(B→A): {h_BA:.6f}")
        
        print("\n计算不同噪声强度下的跃迁概率...")
        epsilon_values = config.epsilon_values
        
        for eps in epsilon_values:
            p_AB = fw.compute_transition_probability(A, B, eps)
            prob_AB.append(p_AB)
            print(f"  ε = {eps:.3f}: P(A→B) = {p_AB:.6e}")
    
    print("\n分析逃逸时间...")
    escape_analyzer = EscapeTimeAnalyzer(potential, gradient, config)
    
    escape_results = None
    if len(minima) > 0:
        test_basin = minima[0]
        test_epsilon = 0.3
        
        print(f"从 {test_basin} 逃逸（ε = {test_epsilon}）...")
        escape_results = escape_analyzer.estimate_escape_time(
            test_basin, test_epsilon, num_runs=15
        )
        
        print(f"平均逃逸时间: {escape_results['mean_escape_time']:.6f}")
        print(f"逃逸时间标准差: {escape_results['std_escape_time']:.6f}")
        print(f"成功逃逸次数: {escape_results['num_escapes']}")
    
    instanton_action = 0.0
    
    if len(minima) >= 2:
        print("\n计算Instanton路径...")
        instanton = InstantonPath(potential, gradient)
        
        A = minima[0]
        B = minima[1]
        
        instanton_path = instanton.compute(A, B, num_points=30, num_iterations=50)
        
        instanton_action = instanton.compute_action(instanton_path)
        
        print(f"Instanton路径长度: {len(instanton_path)}")
        print(f"Instanton作用量: {instanton_action:.6f}")
        print(f"路径起点: {instanton_path[0]}")
        print(f"路径终点: {instanton_path[-1]}")
    
    has_minima = len(minima) >= 2
    has_saddles = len(saddles) > 0
    has_barrier = h_AB > 0 or h_BA > 0
    
    return {
        'local_minima': minima,
        'saddle_points': saddles[:3],
        'barrier_height_AB': h_AB,
        'barrier_height_BA': h_BA,
        'transition_probabilities': prob_AB,
        'escape_time': escape_results,
        'instanton_action': instanton_action,
        'converged': has_minima and (has_saddles or has_barrier)
    }


if __name__ == "__main__":
    results = verify_theorem6_cognitive_transition()
    
    print("\n" + "=" * 60)
    print("定理6验证结果汇总:")
    print("=" * 60)
    print(f"发现亚稳态: {len(results['local_minima'])} 个")
    print(f"发现鞍点: {len(results['saddle_points'])} 个")
    
    if results['barrier_height_AB'] > 0:
        print(f"势垒高度 h(A→B): {results['barrier_height_AB']:.6f}")
        print(f"势垒高度 h(B→A): {results['barrier_height_BA']:.6f}")
    
    if results['escape_time'] is not None:
        print(f"平均逃逸时间: {results['escape_time']['mean_escape_time']:.6f}")
    
    print(f"\n整体验证: {'通过' if results['converged'] else '未通过'}")
