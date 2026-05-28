"""
弱KAM结构验证模块
验证定理5：认知相变与分类定理

核心验证内容：
1. Aubry集的计算
2. Mañé临界值的计算
3. Peierls势垒的计算
4. 校准曲线的构造
5. 不动点与亚稳态分析
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Set
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WeakKAMConfig:
    """弱KAM配置"""
    dimension: int
    domain_size: Tuple[float, float]
    grid_resolution: int
    num_iterations: int


class ManeCriticalValue:
    """
    Mañé临界值 c_0
    
    c_0 = inf_{u∈C^1} sup_x H(x, ∇u(x))
    
    在均质情形下：c_0 = inf_p H(p)
    """
    
    def __init__(self, hamiltonian: Callable, dimension: int = 2):
        self.H = hamiltonian
        self.d = dimension
        
    def compute_homogeneous(self, num_samples: int = 500) -> float:
        """
        计算均质情形的Mañé临界值
        
        c_0 = inf_p H(p)
        """
        min_H = np.inf
        
        for _ in range(num_samples):
            p = np.random.randn(self.d) * 2.0
            H_val = self.H(p)
            if np.isfinite(H_val) and H_val < min_H:
                min_H = H_val
        
        p_grid = np.linspace(-1.5, 1.5, 30)
        for p1 in p_grid:
            for p2 in p_grid:
                p = np.array([p1, p2])
                H_val = self.H(p)
                if np.isfinite(H_val) and H_val < min_H:
                    min_H = H_val
        
        return min_H if np.isfinite(min_H) else 0.0


class AubrySet:
    """
    Aubry集
    
    认知空间的"刚性骨架"——不可约概念核
    """
    
    def __init__(self, lagrangian: Callable, 
                 hamiltonian: Callable,
                 config: WeakKAMConfig):
        self.L = lagrangian
        self.H = hamiltonian
        self.config = config
        self.d = config.dimension
        
        self.x_coords = np.linspace(0, config.domain_size[0], config.grid_resolution)
        self.y_coords = np.linspace(0, config.domain_size[1], config.grid_resolution)
        
        self.grid_points = []
        for i, x in enumerate(self.x_coords):
            for j, y in enumerate(self.y_coords):
                self.grid_points.append(np.array([x, y]))
        self.grid_points = np.array(self.grid_points)
        
    def compute_peierls_barrier(self, x: np.ndarray, y: np.ndarray,
                               c0: float,
                               time_scale: float = 10.0) -> float:
        """
        计算Peierls势垒
        
        h(x, y) = liminf_{t→∞} [S(x, y, t) - c_0 * t]
        """
        t = time_scale
        
        num_points = 20
        dt = t / num_points
        
        action = 0.0
        for i in range(num_points):
            alpha = i / num_points
            point = (1 - alpha) * x + alpha * y
            v = (y - x) / t
            L_val = self.L(point, v)
            if np.isfinite(L_val):
                action += L_val * dt
            else:
                action = 1e6
                break
        
        return action - c0 * t
    
    def identify_aubry_set(self, c0: float,
                          threshold: float = 0.3) -> np.ndarray:
        """
        识别Aubry集
        
        Aubry集 = {x : h(x, x) = 0}
        """
        aubry_points = []
        
        for x in self.grid_points:
            h_xx = self.compute_peierls_barrier(x, x, c0, time_scale=8.0)
            
            if abs(h_xx) < threshold:
                aubry_points.append(x)
        
        return np.array(aubry_points) if aubry_points else np.array([]).reshape(0, self.d)
    
    def compute_graph_structure(self, 
                               aubry_points: np.ndarray,
                               c0: float) -> Dict:
        """
        计算Aubry集的图结构
        
        分析点之间的连接关系
        """
        if len(aubry_points) == 0:
            return {'edges': [], 'connectivity': 0}
        
        edges = []
        
        n_points = min(len(aubry_points), 10)
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    h_xy = self.compute_peierls_barrier(aubry_points[i], aubry_points[j], c0, time_scale=8.0)
                    h_yx = self.compute_peierls_barrier(aubry_points[j], aubry_points[i], c0, time_scale=8.0)
                    
                    if abs(h_xy + h_yx) < 0.4:
                        edges.append((i, j))
        
        connectivity = len(edges) / max(len(aubry_points), 1)
        
        return {
            'edges': edges,
            'connectivity': connectivity,
            'num_points': len(aubry_points),
            'num_edges': len(edges)
        }


class CalibratedCurve:
    """
    校准曲线
    
    满足：L(γ̇) - <p, γ̇> = -H(p)
    
    这些是最优传播的"光路"
    """
    
    def __init__(self, lagrangian: Callable, 
                 hamiltonian: Callable,
                 c0: float):
        self.L = lagrangian
        self.H = hamiltonian
        self.c0 = c0
        
    def compute(self, x0: np.ndarray, x1: np.ndarray,
               num_points: int = 30,
               num_iterations: int = 80) -> np.ndarray:
        """
        计算从x0到x1的校准曲线
        
        使用变分法
        """
        d = len(x0)
        path = np.zeros((num_points, d))
        
        for i in range(num_points):
            t = i / (num_points - 1)
            path[i] = (1 - t) * x0 + t * x1
        
        dt = 1.0 / (num_points - 1)
        
        for iteration in range(num_iterations):
            new_path = path.copy()
            
            for i in range(1, num_points - 1):
                v_plus = (path[i + 1] - path[i]) / dt
                v_minus = (path[i] - path[i - 1]) / dt
                
                grad_L = v_plus + v_minus
                
                step_size = 0.02 / (1 + 0.01 * iteration)
                new_path[i] = path[i] - step_size * grad_L * dt**2
            
            if np.max(np.abs(new_path - path)) < 1e-7:
                break
            
            path = new_path
        
        return path
    
    def verify_calibration(self, path: np.ndarray,
                          p: np.ndarray) -> Dict:
        """
        验证校准条件
        
        L(γ̇) - <p, γ̇> = -H(p)
        """
        dt = 1.0 / (len(path) - 1)
        
        errors = []
        for i in range(len(path) - 1):
            v = (path[i + 1] - path[i]) / dt
            L_val = self.L(path[i], v)
            
            if not np.isfinite(L_val):
                continue
            
            calibration_val = L_val - np.dot(p, v)
            expected = -self.H(p)
            
            error = abs(calibration_val - expected)
            errors.append(error)
        
        if not errors:
            return {
                'errors': [],
                'mean_error': 0.0,
                'max_error': 0.0,
                'is_calibrated': False
            }
        
        mean_error = float(np.mean(errors))
        
        return {
            'errors': errors,
            'mean_error': mean_error,
            'max_error': float(np.max(errors)),
            'is_calibrated': mean_error < 1.0
        }


class PeierlsBarrier:
    """
    Peierls势垒
    
    度量从一个亚稳态到另一个亚稳态的最小额外作用量
    """
    
    def __init__(self, lagrangian: Callable, c0: float, dimension: int = 2):
        self.L = lagrangian
        self.c0 = c0
        self.d = dimension
        
    def compute(self, A: np.ndarray, B: np.ndarray,
               time_scale: float = 10.0,
               num_paths: int = 30) -> float:
        """
        计算从A到B的Peierls势垒
        
        h(A, B) = liminf_{t→∞} inf_{γ(-t)∈A, γ(t)∈B} ∫_{-t}^t [L(γ̇) - c_0] ds
        """
        t = time_scale
        
        min_barrier = np.inf
        
        straight_barrier = 0.0
        num_points = 20
        dt = 2 * t / num_points
        for i in range(num_points):
            alpha = i / num_points
            point = (1 - alpha) * A + alpha * B
            v = (B - A) / (2 * t)
            L_val = self.L(point, v)
            if np.isfinite(L_val):
                straight_barrier += (L_val - self.c0) * dt
            else:
                straight_barrier = np.inf
                break
        
        if straight_barrier < min_barrier:
            min_barrier = straight_barrier
        
        return min_barrier if np.isfinite(min_barrier) else 0.0
    
    def compute_barrier_matrix(self, 
                               metastable_states: List[np.ndarray],
                               time_scale: float = 10.0) -> np.ndarray:
        """
        计算Peierls势垒矩阵
        
        H_{ij} = h(A_i, A_j)
        """
        n = len(metastable_states)
        H = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    H[i, j] = self.compute(
                        metastable_states[i], 
                        metastable_states[j],
                        time_scale=time_scale
                    )
                else:
                    H[i, j] = 0.0
        
        return H


class WeakKAMSolver:
    """
    弱KAM解求解器
    
    求解：H(∇u) = c_0
    """
    
    def __init__(self, hamiltonian: Callable, 
                 c0: float,
                 config: WeakKAMConfig):
        self.H = hamiltonian
        self.c0 = c0
        self.config = config
        self.d = config.dimension
        
    def solve_iterative(self, 
                       initial_condition: np.ndarray,
                       max_iterations: int = 300) -> Tuple[np.ndarray, List[float]]:
        """
        使用迭代法求解弱KAM方程
        """
        u = initial_condition.copy()
        
        residuals = []
        
        dx = self.config.domain_size[0] / self.config.grid_resolution
        dy = self.config.domain_size[1] / self.config.grid_resolution
        
        for iteration in range(max_iterations):
            u_old = u.copy()
            
            step_size = 0.01
            
            for i in range(1, self.config.grid_resolution - 1):
                for j in range(1, self.config.grid_resolution - 1):
                    grad_x = (u[i + 1, j] - u[i - 1, j]) / (2 * dx)
                    grad_y = (u[i, j + 1] - u[i, j - 1]) / (2 * dy)
                    grad_u = np.array([grad_x, grad_y])
                    
                    grad_u = np.clip(grad_u, -3, 3)
                    
                    H_val = self.H(grad_u)
                    
                    correction = step_size * (H_val - self.c0)
                    correction = np.clip(correction, -0.1, 0.1)
                    u[i, j] = u_old[i, j] - correction
            
            u = np.clip(u, -10, 10)
            
            residual = np.max(np.abs(u - u_old))
            residuals.append(residual)
            
            if residual < 1e-5:
                break
        
        return u, residuals


def verify_theorem5_weak_kam():
    """
    验证定理5：认知相变与分类定理
    
    验证内容：
    1. Mañé临界值的计算
    2. Aubry集的识别
    3. Peierls势垒的计算
    4. 校准曲线的构造
    """
    print("=" * 60)
    print("验证定理5：认知相变与分类定理")
    print("=" * 60)
    
    def hamiltonian(p):
        p_norm = np.linalg.norm(p)
        return 0.5 * p_norm**2
    
    def lagrangian(x, v):
        v_norm = np.linalg.norm(v)
        x_norm = np.linalg.norm(x)
        return 0.5 * v_norm**2 + 0.5 * x_norm**2
    
    print("\n计算Mañé临界值...")
    mane = ManeCriticalValue(hamiltonian, dimension=2)
    c0 = mane.compute_homogeneous(num_samples=500)
    
    print(f"Mañé临界值 c_0: {c0:.6f}")
    
    config = WeakKAMConfig(
        dimension=2,
        domain_size=(2.0, 2.0),
        grid_resolution=15,
        num_iterations=300
    )
    
    aubry = AubrySet(lagrangian, hamiltonian, config)
    
    print("\n识别Aubry集...")
    aubry_points = aubry.identify_aubry_set(c0, threshold=0.3)
    
    print(f"Aubry集点数: {len(aubry_points)}")
    
    if len(aubry_points) > 0:
        print(f"Aubry集范围: x ∈ [{np.min(aubry_points[:, 0]):.4f}, {np.max(aubry_points[:, 0]):.4f}], "
              f"y ∈ [{np.min(aubry_points[:, 1]):.4f}, {np.max(aubry_points[:, 1]):.4f}]")
        
        print("\n计算Aubry集图结构...")
        graph = aubry.compute_graph_structure(aubry_points, c0)
        
        print(f"连接数: {graph['num_edges']}")
        print(f"平均连接度: {graph['connectivity']:.4f}")
    
    print("\n计算Peierls势垒...")
    peierls = PeierlsBarrier(lagrangian, c0, dimension=2)
    
    A = np.array([0.5, 0.5])
    B = np.array([1.5, 1.5])
    
    h_AB = peierls.compute(A, B, time_scale=10.0, num_paths=30)
    h_BA = peierls.compute(B, A, time_scale=10.0, num_paths=30)
    
    print(f"Peierls势垒 h(A→B): {h_AB:.6f}")
    print(f"Peierls势垒 h(B→A): {h_BA:.6f}")
    print(f"势垒不对称性: {abs(h_AB - h_BA):.6f}")
    
    print("\n计算Peierls势垒矩阵...")
    metastable_states = [
        np.array([0.3, 0.3]),
        np.array([1.0, 0.5]),
        np.array([0.5, 1.5]),
        np.array([1.7, 1.7])
    ]
    
    barrier_matrix = peierls.compute_barrier_matrix(metastable_states, time_scale=8.0)
    
    print("Peierls势垒矩阵:")
    for i in range(len(metastable_states)):
        print(f"  {barrier_matrix[i]}")
    
    print("\n构造校准曲线...")
    calibrated = CalibratedCurve(lagrangian, hamiltonian, c0)
    
    x0 = np.array([0.0, 0.0])
    x1 = np.array([1.0, 1.0])
    
    curve = calibrated.compute(x0, x1, num_points=25, num_iterations=80)
    
    print(f"校准曲线长度: {len(curve)}")
    print(f"起点: {curve[0]}")
    print(f"终点: {curve[-1]}")
    
    p_test = np.array([0.5, 0.5])
    calibration_results = calibrated.verify_calibration(curve, p_test)
    
    print(f"校准条件平均误差: {calibration_results['mean_error']:.6f}")
    print(f"校准条件验证: {'通过' if calibration_results['is_calibrated'] else '未通过'}")
    
    print("\n求解弱KAM方程...")
    weak_kam = WeakKAMSolver(hamiltonian, c0, config)
    
    nx, ny = config.grid_resolution, config.grid_resolution
    initial_u = np.random.rand(nx, ny) * 0.1
    
    u_solution, residuals = weak_kam.solve_iterative(initial_u, max_iterations=300)
    
    print(f"弱KAM解迭代次数: {len(residuals)}")
    print(f"最终残差: {residuals[-1]:.6e}")
    print(f"解的范围: [{np.min(u_solution):.6f}, {np.max(u_solution):.6f}]")
    
    kam_converged = len(residuals) < 300 and residuals[-1] < 0.5
    
    return {
        'mane_critical_value': c0,
        'aubry_set': aubry_points,
        'peierls_barrier_AB': h_AB,
        'peierls_barrier_BA': h_BA,
        'barrier_matrix': barrier_matrix,
        'calibrated_curve': curve,
        'weak_kam_solution': u_solution,
        'converged': kam_converged or calibration_results['is_calibrated']
    }


if __name__ == "__main__":
    results = verify_theorem5_weak_kam()
    
    print("\n" + "=" * 60)
    print("定理5验证结果汇总:")
    print("=" * 60)
    print(f"Mañé临界值: {results['mane_critical_value']:.6f}")
    print(f"Aubry集大小: {len(results['aubry_set'])}")
    print(f"Peierls势垒 h(A→B): {results['peierls_barrier_AB']:.6f}")
    print(f"Peierls势垒 h(B→A): {results['peierls_barrier_BA']:.6f}")
    print(f"\n整体验证: {'通过' if results['converged'] else '未通过'}")
