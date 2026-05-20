"""
核心数据结构：有限域运算、观测矩阵构造、几何核计算
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CARule:
    """元胞自动机规则 P(z) = sum a_j z^j"""
    r: int
    coeffs: Dict[int, int]
    p: int
    
    def __post_init__(self):
        self.d_minus = self._compute_d_minus()
        self.d_plus = self._compute_d_plus()
    
    def _compute_d_minus(self) -> int:
        max_val = 0
        for j in self.coeffs.keys():
            if j < 0 and self.coeffs[j] % self.p != 0:
                max_val = max(max_val, abs(j))
        return max_val
    
    def _compute_d_plus(self) -> int:
        max_val = 0
        for j in self.coeffs.keys():
            if j > 0 and self.coeffs[j] % self.p != 0:
                max_val = max(max_val, j)
        return max_val
    
    def get_coeff(self, j: int) -> int:
        return self.coeffs.get(j, 0) % self.p
    
    def __repr__(self):
        terms = []
        for j in sorted(self.coeffs.keys()):
            c = self.coeffs[j] % self.p
            if c != 0:
                if j == 0:
                    terms.append(f"{c}")
                elif j == 1:
                    terms.append(f"{c}z")
                elif j == -1:
                    terms.append(f"{c}z^-1")
                else:
                    terms.append(f"{c}z^{j}")
        return f"P(z) = {' + '.join(terms) if terms else '0'} (mod {self.p})"


def create_rule_90(p: int = 2) -> CARule:
    """规则90: P(z) = z^-1 + z"""
    return CARule(r=1, coeffs={-1: 1, 1: 1}, p=p)


def create_rule_150(p: int = 2) -> CARule:
    """规则150: P(z) = z^-1 + 1 + z"""
    return CARule(r=1, coeffs={-1: 1, 0: 1, 1: 1}, p=p)


def create_frobenius_resonance_model(p: int = 2) -> CARule:
    """Frobenius共振模型: P(z) = z^-2 + 1 + z^2"""
    return CARule(r=2, coeffs={-2: 1, 0: 1, 2: 1}, p=p)


def create_general_rule(r: int, coeffs: Dict[int, int], p: int = 2) -> CARule:
    """创建一般规则"""
    return CARule(r=r, coeffs=coeffs, p=p)


class FiniteFieldMatrix:
    """有限域上的矩阵运算"""
    
    @staticmethod
    def mod(a: np.ndarray, p: int) -> np.ndarray:
        return a % p
    
    @staticmethod
    def rank(matrix: np.ndarray, p: int) -> int:
        m = matrix.copy() % p
        rows, cols = m.shape
        rank = 0
        row = 0
        
        for col in range(cols):
            pivot_row = -1
            for r in range(row, rows):
                if m[r, col] % p != 0:
                    pivot_row = r
                    break
            
            if pivot_row == -1:
                continue
            
            if pivot_row != row:
                m[[row, pivot_row]] = m[[pivot_row, row]]
            
            inv = pow(int(m[row, col]), -1, p)
            m[row] = (m[row] * inv) % p
            
            for r in range(rows):
                if r != row and m[r, col] % p != 0:
                    m[r] = (m[r] - m[r, col] * m[row]) % p
            
            rank += 1
            row += 1
            if row >= rows:
                break
        
        return rank
    
    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray, p: int) -> Optional[np.ndarray]:
        n = A.shape[1]
        augmented = np.column_stack([A, b])
        m = augmented.copy() % p
        rows, cols = m.shape
        
        pivot_cols = []
        row = 0
        for col in range(n):
            pivot_row = -1
            for r in range(row, rows):
                if m[r, col] % p != 0:
                    pivot_row = r
                    break
            
            if pivot_row == -1:
                continue
            
            if pivot_row != row:
                m[[row, pivot_row]] = m[[pivot_row, row]]
            
            inv = pow(int(m[row, col]), -1, p)
            m[row] = (m[row] * inv) % p
            
            for r in range(rows):
                if r != row and m[r, col] % p != 0:
                    m[r] = (m[r] - m[r, col] * m[row]) % p
            
            pivot_cols.append(col)
            row += 1
        
        for r in range(row, rows):
            if m[r, -1] % p != 0:
                return None
        
        solution = np.zeros(n, dtype=int)
        for i, col in enumerate(pivot_cols):
            solution[col] = m[i, -1] % p
        
        return solution


class ObservationMatrix:
    """观测矩阵 Θ_T 的构造与操作"""
    
    def __init__(self, rule: CARule, T: int, c: int):
        self.rule = rule
        self.T = T
        self.c = c
        self.p = rule.p
        self.n_T = 2 * (c + rule.r) * T + 1
        self.A_T = list(range(-(c + rule.r) * T, (c + rule.r) * T + 1))
        self.matrix = None
        self._rank = None
    
    def evolve_step(self, config: np.ndarray) -> np.ndarray:
        new_config = np.zeros_like(config)
        n = len(config)
        
        for i in range(n):
            for j, coeff in self.rule.coeffs.items():
                source_idx = i + j
                if 0 <= source_idx < n:
                    new_config[i] = (new_config[i] + coeff * config[source_idx]) % self.p
        
        return new_config
    
    def construct_matrix(self) -> np.ndarray:
        window_size = 2 * self.c * self.T + 1
        num_observations = (self.T + 1) * window_size
        num_initial = self.n_T
        
        self.matrix = np.zeros((num_observations, num_initial), dtype=int)
        
        for col_idx, i in enumerate(self.A_T):
            config = np.zeros(self.n_T, dtype=int)
            center_idx = self.n_T // 2
            actual_idx = center_idx + i
            if 0 <= actual_idx < self.n_T:
                config[actual_idx] = 1
            
            current_config = config.copy()
            
            for t in range(self.T + 1):
                window_start = center_idx - self.c * self.T
                window_end = center_idx + self.c * self.T + 1
                
                for j in range(window_size):
                    actual_pos = window_start + j
                    if 0 <= actual_pos < self.n_T:
                        obs_value = current_config[actual_pos]
                        row_idx = t * window_size + j
                        self.matrix[row_idx, col_idx] = obs_value
                
                if t < self.T:
                    current_config = self.evolve_step(current_config)
        
        return self.matrix
    
    def compute_rank(self) -> int:
        if self.matrix is None:
            self.construct_matrix()
        
        if self._rank is None:
            self._rank = FiniteFieldMatrix.rank(self.matrix, self.p)
        
        return self._rank
    
    def get_defect(self) -> int:
        rank = self.compute_rank()
        return self.n_T - rank


class GeometricKernel:
    """几何核的计算"""
    
    @staticmethod
    def compute_dimension(rule: CARule, T: int, c: int) -> int:
        d_minus = rule.d_minus
        d_plus = rule.d_plus
        r = rule.r
        
        dim = T * (r - d_minus) + T * (r - d_plus)
        return dim
    
    @staticmethod
    def get_infinity_indices(rule: CARule, T: int, c: int) -> List[int]:
        d_minus = rule.d_minus
        d_plus = rule.d_plus
        r = rule.r
        
        A_T_min = -(c + r) * T
        A_T_max = (c + r) * T
        
        I_infinity = []
        
        if d_plus > 0:
            left_threshold = -c * T - d_plus * T
            for i in range(A_T_min, left_threshold):
                I_infinity.append(i)
        
        if d_minus > 0:
            right_threshold = c * T + d_minus * T
            for i in range(right_threshold + 1, A_T_max + 1):
                I_infinity.append(i)
        
        return sorted(I_infinity)
    
    @staticmethod
    def compute_tau(i: int, rule: CARule, T: int, c: int) -> int:
        d_minus = rule.d_minus
        d_plus = rule.d_plus
        
        if abs(i) <= c * T:
            return 0
        
        if i < -c * T and d_plus > 0:
            tau = int(np.ceil((-c * T - i) / d_plus))
            return tau if tau <= T else float('inf')
        
        if i > c * T and d_minus > 0:
            tau = int(np.ceil((i - c * T) / d_minus))
            return tau if tau <= T else float('inf')
        
        return float('inf')


def compute_leading_term_info(rule: CARule, T: int, c: int, i: int) -> Tuple[int, int, int]:
    tau = GeometricKernel.compute_tau(i, rule, T, c)
    
    if tau == float('inf') or tau > T:
        return (float('inf'), None, 0)
    
    if abs(i) <= c * T:
        j_star = i
        c_star = 1
        return (0, j_star, c_star)
    
    if i < -c * T:
        j_star = -c * T
        c_star = pow(rule.get_coeff(rule.d_plus), tau, rule.p)
        return (tau, j_star, c_star)
    
    if i > c * T:
        j_star = c * T
        c_star = pow(rule.get_coeff(-rule.d_minus), tau, rule.p)
        return (tau, j_star, c_star)
    
    return (tau, None, 0)


if __name__ == "__main__":
    print("=" * 60)
    print("核心数据结构测试")
    print("=" * 60)
    
    rule = create_frobenius_resonance_model(p=2)
    print(f"\n规则: {rule}")
    print(f"d_- = {rule.d_minus}, d_+ = {rule.d_plus}")
    
    T, c = 3, 2
    obs = ObservationMatrix(rule, T, c)
    matrix = obs.construct_matrix()
    rank = obs.compute_rank()
    defect = obs.get_defect()
    
    print(f"\nT={T}, c={c}")
    print(f"观测矩阵大小: {matrix.shape}")
    print(f"秩: {rank}")
    print(f"缺陷: {defect}")
    
    geo_dim = GeometricKernel.compute_dimension(rule, T, c)
    print(f"几何核维数: {geo_dim}")
    
    print(f"\n定理验证: 缺陷({defect}) == 几何核维数({geo_dim})? {defect == geo_dim}")
