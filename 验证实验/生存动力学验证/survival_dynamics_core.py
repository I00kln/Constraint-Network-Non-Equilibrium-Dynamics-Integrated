"""
生存动力学验证系统 - 核心引擎
基于文档《生存动力学：基于 Clifford 环面的弹性正规双曲吸引结构》
实现耗散-陀螺动力学系统并验证各定理
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SystemParameters:
    r1: float = 1.0
    r2: float = 1.0
    omega: float = 1.0
    gamma: float = 0.5
    
    def __post_init__(self):
        self.R = np.sqrt(self.r1**2 + self.r2**2)

class SurvivalDynamicsSystem:
    def __init__(self, params: SystemParameters):
        self.params = params
        self.p = params
    
    def elastic_potential(self, z1: complex, z2: complex) -> float:
        return 0.5 * ((abs(z1)**2 - self.p.r1**2)**2 + 
                      (abs(z2)**2 - self.p.r2**2)**2)
    
    def gradient_V(self, z1: complex, z2: complex) -> Tuple[complex, complex]:
        grad_z1 = 2 * (abs(z1)**2 - self.p.r1**2) * z1
        grad_z2 = 2 * (abs(z2)**2 - self.p.r2**2) * z2
        return grad_z1, grad_z2
    
    def mechanical_energy(self, z1: complex, z2: complex, 
                          v1: complex, v2: complex) -> float:
        kinetic = 0.5 * (abs(v1)**2 + abs(v2)**2)
        potential = self.elastic_potential(z1, z2)
        return kinetic + potential
    
    def energy_dissipation_rate(self, v1: complex, v2: complex) -> float:
        return -self.p.gamma * (abs(v1)**2 + abs(v2)**2)
    
    def dynamics_ode(self, t: float, state: np.ndarray) -> np.ndarray:
        z1 = state[0] + 1j * state[1]
        z2 = state[2] + 1j * state[3]
        v1 = state[4] + 1j * state[5]
        v2 = state[6] + 1j * state[7]
        
        grad_z1, grad_z2 = self.gradient_V(z1, z2)
        
        dz1 = 1j * self.p.omega * z1 + v1
        dz2 = 1j * self.p.omega * z2 + v2
        dv1 = -self.p.gamma * v1 - grad_z1
        dv2 = -self.p.gamma * v2 - grad_z2
        
        return np.array([
            dz1.real, dz1.imag, dz2.real, dz2.imag,
            dv1.real, dv1.imag, dv2.real, dv2.imag
        ])
    
    def second_order_form(self, t: float, state: np.ndarray) -> np.ndarray:
        z1 = state[0] + 1j * state[1]
        z2 = state[2] + 1j * state[3]
        dz1 = state[4] + 1j * state[5]
        dz2 = state[6] + 1j * state[7]
        
        term1 = 2 * (abs(z1)**2 - self.p.r1**2) * z1
        term2 = 2 * (abs(z2)**2 - self.p.r2**2) * z2
        
        ddz1 = -(self.p.gamma - 1j * self.p.omega) * dz1 + \
               1j * self.p.omega * self.p.gamma * z1 - term1
        ddz2 = -(self.p.gamma - 1j * self.p.omega) * dz2 + \
               1j * self.p.omega * self.p.gamma * z2 - term2
        
        return np.array([
            dz1.real, dz1.imag, dz2.real, dz2.imag,
            ddz1.real, ddz1.imag, ddz2.real, ddz2.imag
        ])
    
    def simulate(self, initial_state: np.ndarray, 
                 t_span: Tuple[float, float],
                 n_points: int = 1000) -> Dict:
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(
            self.dynamics_ode, t_span, initial_state,
            t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12
        )
        
        trajectory = {
            't': sol.t,
            'z1': sol.y[0] + 1j * sol.y[1],
            'z2': sol.y[2] + 1j * sol.y[3],
            'v1': sol.y[4] + 1j * sol.y[5],
            'v2': sol.y[6] + 1j * sol.y[7]
        }
        
        trajectory['energy'] = np.array([
            self.mechanical_energy(trajectory['z1'][i], trajectory['z2'][i],
                                   trajectory['v1'][i], trajectory['v2'][i])
            for i in range(len(trajectory['t']))
        ])
        
        trajectory['dissipation'] = np.array([
            self.energy_dissipation_rate(trajectory['v1'][i], trajectory['v2'][i])
            for i in range(len(trajectory['t']))
        ])
        
        trajectory['dist_to_torus'] = np.sqrt(
            (np.abs(trajectory['z1']) - self.p.r1)**2 + 
            (np.abs(trajectory['z2']) - self.p.r2)**2
        )
        
        return trajectory
    
    def linearization_at_origin(self) -> np.ndarray:
        gamma = self.p.gamma
        omega = self.p.omega
        r1 = self.p.r1
        r2 = self.p.r2
        
        def block_matrix(r_k):
            A = np.zeros((4, 4))
            A[0, 1] = omega
            A[0, 2] = 1
            A[1, 0] = -omega
            A[1, 3] = 1
            A[2, 0] = 2 * r_k**2
            A[2, 2] = -gamma
            A[2, 3] = omega
            A[3, 1] = 2 * r_k**2
            A[3, 2] = -omega
            A[3, 3] = -gamma
            return A
        
        A1 = block_matrix(r1)
        A2 = block_matrix(r2)
        
        A = np.zeros((8, 8))
        A[:4, :4] = A1
        A[4:, 4:] = A2
        
        return A
    
    def linearization_on_torus(self) -> Tuple[np.ndarray, np.ndarray]:
        gamma = self.p.gamma
        omega = self.p.omega
        r1 = self.p.r1
        r2 = self.p.r2
        
        def block_Ak(r_k):
            A = np.array([
                [0, 1, 0, 0],
                [-4*r_k**2, -gamma, omega, 0],
                [0, -omega, -gamma, 0],
                [0, 0, 1, 0]
            ])
            return A
        
        A1 = block_Ak(r1)
        A2 = block_Ak(r2)
        
        A = np.zeros((8, 8))
        A[:4, :4] = A1
        A[4:, 4:] = A2
        
        eigenvalues, eigenvectors = eig(A)
        
        return A, (eigenvalues, eigenvectors)
    
    def compute_characteristic_polynomial_coeffs(self, r_k: float) -> Tuple[float, ...]:
        gamma = self.p.gamma
        omega = self.p.omega
        
        a3 = 1.0
        a2 = 2 * gamma
        a1 = gamma**2 + omega**2 + 4 * r_k**2
        a0 = 4 * r_k**2 * gamma
        
        return (a3, a2, a1, a0)
    
    def routh_hurwitz_check(self, r_k: float) -> Dict:
        a3, a2, a1, a0 = self.compute_characteristic_polynomial_coeffs(r_k)
        
        cond1 = a2 > 0
        cond2 = a1 > 0
        cond3 = a0 > 0
        cond4 = a2 * a1 - a3 * a0 > 0
        
        return {
            'a3': a3, 'a2': a2, 'a1': a1, 'a0': a0,
            'a2_positive': cond1,
            'a1_positive': cond2,
            'a0_positive': cond3,
            'Routh_Hurwitz': cond4,
            'all_stable': cond1 and cond2 and cond3 and cond4
        }

class TheoremVerifier:
    def __init__(self, system: SurvivalDynamicsSystem):
        self.sys = system
    
    def verify_theorem1_energy_decay(self, trajectory: Dict) -> Dict:
        E = trajectory['energy']
        t = trajectory['t']
        
        is_monotonic = np.all(np.diff(E) <= 1e-10)
        
        dE_dt_numerical = np.gradient(E, t)
        dE_dt_theoretical = trajectory['dissipation']
        
        error = np.max(np.abs(dE_dt_numerical - dE_dt_theoretical))
        
        E_decrease = E[0] - E[-1]
        total_dissipation = -np.trapezoid(trajectory['dissipation'], t)
        
        return {
            'is_monotonic_decreasing': is_monotonic,
            'max_gradient_error': error,
            'energy_decrease': E_decrease,
            'total_dissipation_integral': total_dissipation,
            'energy_balance_error': abs(E_decrease - total_dissipation),
            'initial_energy': E[0],
            'final_energy': E[-1],
            'theorem_holds': is_monotonic and error < 1e-6
        }
    
    def verify_theorem2_hyperbolic_saddle(self) -> Dict:
        A = self.sys.linearization_at_origin()
        eigenvalues, _ = eig(A)
        
        real_parts = np.real(eigenvalues)
        n_positive = np.sum(real_parts > 1e-10)
        n_negative = np.sum(real_parts < -1e-10)
        n_zero = np.sum(np.abs(real_parts) <= 1e-10)
        
        is_hyperbolic = n_zero == 0
        is_saddle = n_positive > 0 and n_negative > 0
        
        return {
            'eigenvalues': eigenvalues,
            'real_parts': real_parts,
            'n_positive_real': n_positive,
            'n_negative_real': n_negative,
            'n_zero_real': n_zero,
            'is_hyperbolic': is_hyperbolic,
            'is_saddle_point': is_saddle,
            'stable_dim': n_negative,
            'unstable_dim': n_positive,
            'theorem_holds': is_hyperbolic and is_saddle and \
                            n_positive == 4 and n_negative == 4
        }
    
    def verify_theorem3_global_attraction(self, trajectory: Dict, 
                                          tolerance: float = 1e-2) -> Dict:
        dist = trajectory['dist_to_torus']
        
        converges_to_zero = dist[-1] < tolerance
        
        if converges_to_zero:
            convergence_idx = np.where(dist < tolerance)[0]
            if len(convergence_idx) > 0:
                convergence_time = trajectory['t'][convergence_idx[0]]
            else:
                convergence_time = float('inf')
        else:
            convergence_time = float('inf')
        
        final_dist = dist[-1]
        min_dist = np.min(dist)
        
        v_norm = np.sqrt(np.abs(trajectory['v1'])**2 + np.abs(trajectory['v2'])**2)
        final_v_norm = v_norm[-1]
        
        return {
            'converges_to_torus': converges_to_zero,
            'convergence_time': convergence_time,
            'final_distance_to_torus': final_dist,
            'min_distance_to_torus': min_dist,
            'final_velocity_norm': final_v_norm,
            'distance_history': dist,
            'theorem_holds': converges_to_zero and final_v_norm < tolerance
        }
    
    def verify_theorem5_normal_hyperbolicity(self) -> Dict:
        A, (eigenvalues, eigenvectors) = self.sys.linearization_on_torus()
        
        real_parts = np.real(eigenvalues)
        
        n_zero = np.sum(np.abs(real_parts) < 1e-10)
        n_negative = np.sum(real_parts < -1e-10)
        n_positive = np.sum(real_parts > 1e-10)
        
        zero_eigenvalues = eigenvalues[np.abs(real_parts) < 1e-10]
        stable_eigenvalues = eigenvalues[real_parts < -1e-10]
        
        rh_check_r1 = self.sys.routh_hurwitz_check(self.sys.p.r1)
        rh_check_r2 = self.sys.routh_hurwitz_check(self.sys.p.r2)
        
        max_stable_real = np.max(real_parts[real_parts < -1e-10]) if n_negative > 0 else None
        
        return {
            'eigenvalues': eigenvalues,
            'real_parts': real_parts,
            'n_zero_eigenvalues': n_zero,
            'n_negative_eigenvalues': n_negative,
            'n_positive_eigenvalues': n_positive,
            'zero_eigenvalues': zero_eigenvalues,
            'stable_eigenvalues': stable_eigenvalues,
            'max_stable_real_part': max_stable_real,
            'Routh_Hurwitz_r1': rh_check_r1,
            'Routh_Hurwitz_r2': rh_check_r2,
            'is_attracting_NHIM': n_zero == 2 and n_negative == 6 and n_positive == 0,
            'theorem_holds': n_zero == 2 and n_negative == 6 and n_positive == 0
        }

def generate_random_initial_condition(params: SystemParameters,
                                      seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    z1_mag = np.random.uniform(0.1, 2.0) * params.r1
    z1_phase = np.random.uniform(0, 2*np.pi)
    z1 = z1_mag * np.exp(1j * z1_phase)
    
    z2_mag = np.random.uniform(0.1, 2.0) * params.r2
    z2_phase = np.random.uniform(0, 2*np.pi)
    z2 = z2_mag * np.exp(1j * z2_phase)
    
    v1 = (np.random.randn() + 1j * np.random.randn()) * 0.5
    v2 = (np.random.randn() + 1j * np.random.randn()) * 0.5
    
    return np.array([
        z1.real, z1.imag, z2.real, z2.imag,
        v1.real, v1.imag, v2.real, v2.imag
    ])

def generate_initial_on_torus(params: SystemParameters,
                               phase1: float = 0, phase2: float = 0) -> np.ndarray:
    z1 = params.r1 * np.exp(1j * phase1)
    z2 = params.r2 * np.exp(1j * phase2)
    
    return np.array([
        z1.real, z1.imag, z2.real, z2.imag,
        0, 0, 0, 0
    ])

def generate_initial_near_origin(params: SystemParameters,
                                  epsilon: float = 0.1) -> np.ndarray:
    z1 = epsilon * (np.random.randn() + 1j * np.random.randn())
    z2 = epsilon * (np.random.randn() + 1j * np.random.randn())
    v1 = epsilon * (np.random.randn() + 1j * np.random.randn())
    v2 = epsilon * (np.random.randn() + 1j * np.random.randn())
    
    return np.array([
        z1.real, z1.imag, z2.real, z2.imag,
        v1.real, v1.imag, v2.real, v2.imag
    ])

if __name__ == "__main__":
    print("=" * 70)
    print("生存动力学核心系统 - 基础测试")
    print("=" * 70)
    
    params = SystemParameters(r1=1.0, r2=1.0, omega=1.0, gamma=0.5)
    system = SurvivalDynamicsSystem(params)
    verifier = TheoremVerifier(system)
    
    print(f"\n系统参数: r1={params.r1}, r2={params.r2}, ω={params.omega}, γ={params.gamma}")
    
    print("\n" + "=" * 50)
    print("定理2验证: 原点的双曲鞍点结构")
    print("=" * 50)
    result2 = verifier.verify_theorem2_hyperbolic_saddle()
    print(f"特征值实部: {np.sort(result2['real_parts'])}")
    print(f"正实部个数: {result2['n_positive_real']}, 负实部个数: {result2['n_negative_real']}")
    print(f"是否为双曲鞍点: {result2['is_saddle_point']}")
    print(f"定理2成立: {result2['theorem_holds']}")
    
    print("\n" + "=" * 50)
    print("定理5验证: 正规双曲吸引性")
    print("=" * 50)
    result5 = verifier.verify_theorem5_normal_hyperbolicity()
    print(f"零特征值个数: {result5['n_zero_eigenvalues']}")
    print(f"负实部特征值个数: {result5['n_negative_eigenvalues']}")
    print(f"Routh-Hurwitz条件(r1): {result5['Routh_Hurwitz_r1']['all_stable']}")
    print(f"Routh-Hurwitz条件(r2): {result5['Routh_Hurwitz_r2']['all_stable']}")
    print(f"定理5成立: {result5['theorem_holds']}")
    
    print("\n" + "=" * 50)
    print("动力学模拟测试")
    print("=" * 50)
    
    initial_state = generate_random_initial_condition(params, seed=42)
    trajectory = system.simulate(initial_state, t_span=(0, 50), n_points=2000)
    
    print(f"初始能量: {trajectory['energy'][0]:.6f}")
    print(f"最终能量: {trajectory['energy'][-1]:.6f}")
    print(f"最终到环面距离: {trajectory['dist_to_torus'][-1]:.6e}")
    
    print("\n" + "=" * 50)
    print("定理1验证: 能量单调衰减")
    print("=" * 50)
    result1 = verifier.verify_theorem1_energy_decay(trajectory)
    print(f"能量单调递减: {result1['is_monotonic_decreasing']}")
    print(f"最大梯度误差: {result1['max_gradient_error']:.2e}")
    print(f"定理1成立: {result1['theorem_holds']}")
    
    print("\n" + "=" * 50)
    print("定理3验证: 几乎全局吸引性")
    print("=" * 50)
    result3 = verifier.verify_theorem3_global_attraction(trajectory)
    print(f"收敛到环面: {result3['converges_to_torus']}")
    print(f"最终距离: {result3['final_distance_to_torus']:.2e}")
    print(f"定理3成立: {result3['theorem_holds']}")
    
    print("\n" + "=" * 70)
    print("基础测试完成")
    print("=" * 70)
