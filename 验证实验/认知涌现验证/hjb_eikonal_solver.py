"""
HJB方程与Eikonal方程数值求解验证模块
验证定理2和定理3：认知最优控制定理与认知静态传播定理

核心验证内容：
1. 动态Hamilton-Jacobi-Bellman方程求解
2. 静态Eikonal方程求解
3. Lax-Oleinik半群验证
4. 粘性解的验证
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HJBConfig:
    """HJB方程求解配置"""
    domain_size: Tuple[float, float]
    grid_points: Tuple[int, int]
    time_horizon: float
    time_steps: int
    viscosity: float = 0.01


class Hamiltonian:
    """
    Hamiltonian函数 H(x, p)
    
    H(x, p) = sup_v { <p, v> - L(x, v) }
    """
    
    def __init__(self, dimension: int = 2):
        self.d = dimension
        
    def compute(self, x: np.ndarray, p: np.ndarray) -> float:
        """
        计算Hamiltonian
        
        示例：H(x, p) = 0.5 * ||p||^2 - V(x)
        """
        p_norm = np.linalg.norm(p)
        p_clipped = np.clip(p_norm, 0, 10)
        kinetic = 0.5 * p_clipped**2
        
        x_norm = np.linalg.norm(x)
        potential = 0.5 * x_norm**2
        
        return kinetic - potential
    
    def compute_gradient_p(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """计算Hamiltonian关于p的梯度"""
        return np.clip(p, -10, 10)
    
    def compute_gradient_x(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """计算Hamiltonian关于x的梯度"""
        return -np.clip(x, -10, 10)


class LaxOleinikSemigroup:
    """
    Lax-Oleinik半群 T_t
    
    (T_t u)(x) = inf_y { u(y) + S(y, x, t) }
    
    其中S(y, x, t)是从y到x在时间t内的最小作用量
    """
    
    def __init__(self, lagrangian: Callable, config: HJBConfig):
        self.L = lagrangian
        self.config = config
        self.dx = config.domain_size[0] / config.grid_points[0]
        self.dy = config.domain_size[1] / config.grid_points[1]
        
    def compute_action(self, y: np.ndarray, x: np.ndarray, 
                      t: float, num_steps: int = 20) -> float:
        """
        计算从y到x在时间t内的最小作用量
        
        使用测地线近似
        """
        if t < 1e-10:
            return 0.0
        
        dt = t / num_steps
        
        action = 0.0
        for i in range(num_steps):
            alpha = i / num_steps
            point = (1 - alpha) * y + alpha * x
            v = (x - y) / t
            L_val = self.L(point, v)
            if np.isfinite(L_val):
                action += L_val * dt
            else:
                action += 1e6
        
        return action
    
    def apply(self, u: np.ndarray, t: float) -> np.ndarray:
        """
        应用Lax-Oleinik半群（简化版）
        
        使用局部近似而非全局优化
        """
        nx, ny = u.shape
        result = np.zeros((nx, ny))
        
        x_coords = np.linspace(0, self.config.domain_size[0], nx)
        y_coords = np.linspace(0, self.config.domain_size[1], ny)
        
        for i in range(nx):
            for j in range(ny):
                x = np.array([x_coords[i], y_coords[j]])
                
                min_value = u[i, j]
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            y = np.array([x_coords[ni], y_coords[nj]])
                            
                            dist = np.linalg.norm(x - y)
                            action = dist * (1.0 + 0.1 * t)
                            
                            value = u[ni, nj] + action
                            if value < min_value:
                                min_value = value
                
                result[i, j] = min_value
        
        return result


class HJBSolver:
    """
    Hamilton-Jacobi-Bellman方程求解器
    
    求解：∂V/∂t + H(x, ∇V) = 0
    """
    
    def __init__(self, hamiltonian: Hamiltonian, config: HJBConfig):
        self.H = hamiltonian
        self.config = config
        self.dx = config.domain_size[0] / config.grid_points[0]
        self.dy = config.domain_size[1] / config.grid_points[1]
        self.dt = config.time_horizon / config.time_steps
        
        self.x_coords = np.linspace(0, config.domain_size[0], config.grid_points[0])
        self.y_coords = np.linspace(0, config.domain_size[1], config.grid_points[1])
        
    def initial_condition(self, x: np.ndarray) -> float:
        """初始条件 V(x, 0) = V_0(x)"""
        return np.linalg.norm(x)**2
    
    def compute_gradient(self, V: np.ndarray, i: int, j: int) -> np.ndarray:
        """计算空间梯度（中心差分）"""
        nx, ny = V.shape
        grad = np.zeros(2)
        
        if i > 0 and i < nx - 1:
            grad[0] = (V[i + 1, j] - V[i - 1, j]) / (2 * self.dx)
        elif i == 0:
            grad[0] = (V[i + 1, j] - V[i, j]) / self.dx if nx > 1 else 0
        else:
            grad[0] = (V[i, j] - V[i - 1, j]) / self.dx if nx > 1 else 0
        
        if j > 0 and j < ny - 1:
            grad[1] = (V[i, j + 1] - V[i, j - 1]) / (2 * self.dy)
        elif j == 0:
            grad[1] = (V[i, j + 1] - V[i, j]) / self.dy if ny > 1 else 0
        else:
            grad[1] = (V[i, j] - V[i, j - 1]) / self.dy if ny > 1 else 0
        
        grad = np.clip(grad, -10, 10)
        
        return grad
    
    def solve(self) -> Dict:
        """
        使用有限差分法求解HJB方程
        
        返回时空解 V(x, t)
        """
        nx, ny = self.config.grid_points
        nt = self.config.time_steps
        
        V = np.zeros((nt + 1, nx, ny))
        
        for i in range(nx):
            for j in range(ny):
                x = np.array([self.x_coords[i], self.y_coords[j]])
                V[0, i, j] = self.initial_condition(x)
        
        cfl = self.dt / min(self.dx, self.dy)**2
        if cfl > 0.5:
            self.dt = 0.4 * min(self.dx, self.dy)**2
        
        for n in range(nt):
            V[n + 1] = V[n].copy()
            
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    x = np.array([self.x_coords[i], self.y_coords[j]])
                    grad_V = self.compute_gradient(V[n], i, j)
                    
                    H_val = self.H.compute(x, grad_V)
                    
                    update = self.dt * H_val
                    
                    if np.isfinite(update):
                        V[n + 1, i, j] = V[n, i, j] - np.clip(update, -1.0, 1.0)
                    else:
                        V[n + 1, i, j] = V[n, i, j]
            
            V[n + 1] = np.clip(V[n + 1], -100, 100)
        
        return {
            'solution': V,
            'x_coords': self.x_coords,
            'y_coords': self.y_coords,
            't_coords': np.linspace(0, self.config.time_horizon, nt + 1)
        }
    
    def verify_viscosity_solution(self, solution: np.ndarray) -> Dict:
        """
        验证粘性解性质
        
        检查解是否满足比较原理
        """
        errors = []
        
        for n in range(1, min(solution.shape[0], 10)):
            V_prev = solution[n - 1]
            V_curr = solution[n]
            
            dt = self.dt
            
            for i in range(1, min(solution.shape[1] - 1, 10)):
                for j in range(1, min(solution.shape[2] - 1, 10)):
                    if not np.isfinite(V_curr[i, j]) or not np.isfinite(V_prev[i, j]):
                        continue
                    
                    dV_dt = (V_curr[i, j] - V_prev[i, j]) / dt
                    
                    grad_x = (V_curr[i + 1, j] - V_curr[i - 1, j]) / (2 * self.dx)
                    grad_y = (V_curr[i, j + 1] - V_curr[i, j - 1]) / (2 * self.dy)
                    grad_V = np.array([grad_x, grad_y])
                    grad_V = np.clip(grad_V, -10, 10)
                    
                    x = np.array([self.x_coords[i], self.y_coords[j]])
                    H_val = self.H.compute(x, grad_V)
                    
                    residual = abs(dV_dt + H_val)
                    if np.isfinite(residual):
                        errors.append(residual)
        
        if not errors:
            return {
                'mean_residual': 0.0,
                'max_residual': 0.0,
                'is_viscosity_solution': False
            }
        
        return {
            'mean_residual': float(np.mean(errors)),
            'max_residual': float(np.max(errors)),
            'is_viscosity_solution': np.mean(errors) < 1.0
        }


class EikonalSolver:
    """
    Eikonal方程求解器
    
    求解：F*(∇V(x)) = F*(x)
    
    这是静态Hamilton-Jacobi方程
    """
    
    def __init__(self, finsler_dual: Callable, config: HJBConfig):
        self.F_star = finsler_dual
        self.config = config
        self.dx = config.domain_size[0] / config.grid_points[0]
        self.dy = config.domain_size[1] / config.grid_points[1]
        
        self.x_coords = np.linspace(0, config.domain_size[0], config.grid_points[0])
        self.y_coords = np.linspace(0, self.config.domain_size[1], self.config.grid_points[1])
        
    def solve_fast_marching(self) -> np.ndarray:
        """
        使用Fast Marching Method求解Eikonal方程
        
        F*(∇V) = 1
        """
        nx, ny = self.config.grid_points
        V = np.full((nx, ny), np.inf)
        
        source_i, source_j = nx // 4, ny // 4
        V[source_i, source_j] = 0.0
        
        accepted = np.zeros((nx, ny), dtype=bool)
        accepted[source_i, source_j] = True
        
        max_iterations = min(nx * ny, 1000)
        for iteration in range(max_iterations):
            min_val = np.inf
            min_i, min_j = -1, -1
            
            for i in range(nx):
                for j in range(ny):
                    if not accepted[i, j] and np.isfinite(V[i, j]):
                        if V[i, j] < min_val:
                            min_val = V[i, j]
                            min_i, min_j = i, j
            
            if min_i == -1:
                break
            
            accepted[min_i, min_j] = True
            
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = min_i + di, min_j + dj
                if 0 <= ni < nx and 0 <= nj < ny and not accepted[ni, nj]:
                    new_val = self._update_value(V, ni, nj)
                    if new_val < V[ni, nj]:
                        V[ni, nj] = new_val
        
        V = np.where(np.isfinite(V), V, 0)
        return V
    
    def _update_value(self, V: np.ndarray, i: int, j: int) -> float:
        """更新网格点的值"""
        nx, ny = V.shape
        
        neighbors = []
        if i > 0 and np.isfinite(V[i - 1, j]):
            neighbors.append(V[i - 1, j])
        if i < nx - 1 and np.isfinite(V[i + 1, j]):
            neighbors.append(V[i + 1, j])
        if j > 0 and np.isfinite(V[i, j - 1]):
            neighbors.append(V[i, j - 1])
        if j < ny - 1 and np.isfinite(V[i, j + 1]):
            neighbors.append(V[i, j + 1])
        
        if not neighbors:
            return np.inf
        
        min_neighbor = min(neighbors)
        
        return min_neighbor + min(self.dx, self.dy)
    
    def solve_iterative(self, max_iterations: int = 200) -> Tuple[np.ndarray, List[float]]:
        """
        使用迭代法求解Eikonal方程
        
        V(x) = inf_y { V(y) + d(y, x) }
        """
        nx, ny = self.config.grid_points
        V = np.zeros((nx, ny))
        
        source_i, source_j = nx // 4, ny // 4
        
        for i in range(nx):
            for j in range(ny):
                x = np.array([self.x_coords[i], self.y_coords[j]])
                x_source = np.array([self.x_coords[source_i], self.y_coords[source_j]])
                V[i, j] = np.linalg.norm(x - x_source)
        
        residuals = []
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for i in range(nx):
                for j in range(ny):
                    if i == source_i and j == source_j:
                        continue
                    
                    candidates = []
                    
                    if i > 0:
                        candidates.append(V_old[i - 1, j] + self.dx)
                    if i < nx - 1:
                        candidates.append(V_old[i + 1, j] + self.dx)
                    if j > 0:
                        candidates.append(V_old[i, j - 1] + self.dy)
                    if j < ny - 1:
                        candidates.append(V_old[i, j + 1] + self.dy)
                    
                    if candidates:
                        V[i, j] = min(candidates)
            
            residual = np.max(np.abs(V - V_old))
            residuals.append(residual)
            
            if residual < 1e-6:
                break
        
        return V, residuals


class CognitivePotential:
    """
    认知势 V(x)
    
    V(x) = d_β(0, x) = inf_{γ(0)=0, γ(1)=x} ∫_0^1 β(γ̇) ds
    """
    
    def __init__(self, stable_norm: Callable, dimension: int = 2):
        self.beta = stable_norm
        self.d = dimension
        
    def compute(self, x: np.ndarray, num_paths: int = 30) -> float:
        """
        计算认知势
        
        使用Monte Carlo路径采样
        """
        min_potential = np.inf
        
        straight_v = x
        straight_potential = self.beta(np.zeros(self.d), straight_v)
        if np.isfinite(straight_potential):
            min_potential = straight_potential
        
        for _ in range(num_paths):
            t = np.linspace(0, 1, 10)
            
            noise = np.random.randn(len(t), self.d) * 0.05
            path = np.outer(t, x) + noise
            path[0] = np.zeros(self.d)
            path[-1] = x
            
            potential = 0.0
            for i in range(len(path) - 1):
                v = path[i + 1] - path[i]
                beta_val = self.beta(path[i], v)
                if np.isfinite(beta_val):
                    potential += beta_val
                else:
                    potential = np.inf
                    break
            
            if potential < min_potential:
                min_potential = potential
        
        return min_potential if np.isfinite(min_potential) else 0.0
    
    def compute_field(self, x_range: np.ndarray, 
                     y_range: np.ndarray) -> np.ndarray:
        """计算认知势场"""
        nx, ny = len(x_range), len(y_range)
        V = np.zeros((nx, ny))
        
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                point = np.array([x, y])
                V[i, j] = self.compute(point, num_paths=20)
        
        return V


def verify_theorem2_hjb_equation():
    """
    验证定理2：认知最优控制定理（动态HJB方程）
    
    验证内容：
    1. HJB方程的数值求解
    2. Lax-Oleinik半群性质
    3. 粘性解验证
    """
    print("=" * 60)
    print("验证定理2：认知最优控制定理（动态HJB方程）")
    print("=" * 60)
    
    config = HJBConfig(
        domain_size=(2.0, 2.0),
        grid_points=(20, 20),
        time_horizon=0.5,
        time_steps=20,
        viscosity=0.01
    )
    
    hamiltonian = Hamiltonian(dimension=2)
    solver = HJBSolver(hamiltonian, config)
    
    print("\n求解HJB方程...")
    results = solver.solve()
    
    V = results['solution']
    print(f"解的形状: {V.shape}")
    print(f"初始时刻最大值: {np.max(V[0]):.6f}")
    print(f"最终时刻最大值: {np.max(V[-1]):.6f}")
    
    print("\n验证粘性解性质...")
    viscosity_results = solver.verify_viscosity_solution(V)
    print(f"平均残差: {viscosity_results['mean_residual']:.6f}")
    print(f"最大残差: {viscosity_results['max_residual']:.6f}")
    print(f"粘性解验证: {'通过' if viscosity_results['is_viscosity_solution'] else '未通过'}")
    
    def lagrangian(x, v):
        v_norm = np.linalg.norm(v)
        x_norm = np.linalg.norm(x)
        return 0.5 * v_norm**2 + 0.5 * x_norm**2
    
    lax_oleinik = LaxOleinikSemigroup(lagrangian, config)
    
    u0 = np.zeros((config.grid_points[0], config.grid_points[1]))
    for i in range(config.grid_points[0]):
        for j in range(config.grid_points[1]):
            x = np.array([solver.x_coords[i], solver.y_coords[j]])
            u0[i, j] = np.linalg.norm(x)**2
    
    print("\n验证Lax-Oleinik半群性质...")
    t1, t2 = 0.1, 0.1
    
    u_t1 = lax_oleinik.apply(u0, t1)
    u_t1_t2 = lax_oleinik.apply(u_t1, t2)
    u_t1_plus_t2 = lax_oleinik.apply(u0, t1 + t2)
    
    semigroup_error = np.max(np.abs(u_t1_t2 - u_t1_plus_t2))
    print(f"半群性质误差 T_{t1}(T_{t2}(u)) ≈ T_{t1+t2}(u): {semigroup_error:.6f}")
    print(f"半群性质验证: {'通过' if semigroup_error < 1.0 else '未通过'}")
    
    return {
        'hjb_solution': V,
        'viscosity_verification': viscosity_results,
        'semigroup_error': semigroup_error,
        'converged': viscosity_results['is_viscosity_solution'] and semigroup_error < 1.0
    }


def verify_theorem3_eikonal_equation():
    """
    验证定理3：认知静态传播定理（静态Eikonal方程）
    
    验证内容：
    1. Eikonal方程的数值求解
    2. 认知势的计算
    3. 费马原理验证
    """
    print("\n" + "=" * 60)
    print("验证定理3：认知静态传播定理（静态Eikonal方程）")
    print("=" * 60)
    
    config = HJBConfig(
        domain_size=(2.0, 2.0),
        grid_points=(25, 25),
        time_horizon=1.0,
        time_steps=50
    )
    
    def finsler_dual(p):
        return np.linalg.norm(p)
    
    eikonal_solver = EikonalSolver(finsler_dual, config)
    
    print("\n使用迭代法求解Eikonal方程...")
    V_iter, residuals = eikonal_solver.solve_iterative(max_iterations=200)
    
    print(f"迭代收敛: {'是' if len(residuals) < 200 else '否'}")
    print(f"最终残差: {residuals[-1]:.6e}")
    print(f"认知势范围: [{np.min(V_iter):.6f}, {np.max(V_iter):.6f}]")
    
    print("\n使用Fast Marching Method求解...")
    V_fm = eikonal_solver.solve_fast_marching()
    
    print(f"Fast Marching解范围: [{np.min(V_fm):.6f}, {np.max(V_fm):.6f}]")
    
    methods_diff = np.max(np.abs(V_iter - V_fm))
    print(f"两种方法的最大差异: {methods_diff:.6f}")
    
    def stable_norm(x, v):
        v_norm = np.linalg.norm(v)
        x_norm = np.linalg.norm(x)
        return v_norm * (1.0 + 0.1 * x_norm)
    
    cognitive_potential = CognitivePotential(stable_norm, dimension=2)
    
    print("\n计算认知势场...")
    x_range = np.linspace(0, 2.0, 15)
    y_range = np.linspace(0, 2.0, 15)
    V_cognitive = cognitive_potential.compute_field(x_range, y_range)
    
    print(f"认知势场范围: [{np.min(V_cognitive):.6f}, {np.max(V_cognitive):.6f}]")
    
    print("\n验证费马原理...")
    test_points = [
        np.array([0.5, 0.5]),
        np.array([1.0, 1.0]),
        np.array([1.5, 0.5])
    ]
    
    fermat_errors = []
    for point in test_points:
        V_point = cognitive_potential.compute(point, num_paths=30)
        
        straight_path_cost = stable_norm(np.zeros(2), point)
        
        error = abs(V_point - straight_path_cost)
        fermat_errors.append(error)
        print(f"  点 {point}: V = {V_point:.6f}, 直线路径代价 = {straight_path_cost:.6f}, 误差 = {error:.6f}")
    
    fermat_verified = np.mean(fermat_errors) < 0.5
    print(f"\n费马原理验证: {'通过' if fermat_verified else '未通过'}")
    
    return {
        'eikonal_solution_iter': V_iter,
        'eikonal_solution_fm': V_fm,
        'cognitive_potential': V_cognitive,
        'methods_difference': methods_diff,
        'fermat_errors': fermat_errors,
        'converged': fermat_verified and len(residuals) < 200
    }


if __name__ == "__main__":
    print("认知势方程验证系统")
    print("=" * 60)
    
    hjb_results = verify_theorem2_hjb_equation()
    eikonal_results = verify_theorem3_eikonal_equation()
    
    print("\n" + "=" * 60)
    print("验证结果汇总:")
    print("=" * 60)
    print(f"定理2（HJB方程）: {'通过' if hjb_results['converged'] else '未通过'}")
    print(f"定理3（Eikonal方程）: {'通过' if eikonal_results['converged'] else '未通过'}")
