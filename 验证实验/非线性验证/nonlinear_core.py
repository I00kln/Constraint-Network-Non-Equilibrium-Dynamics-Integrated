"""
非线性元胞自动机验证实验 - 核心引擎
支持一般非线性规则、有效传播速度计算、观测块码构造、等价类细化
"""
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from itertools import product
import math


@dataclass
class NonlinearRule:
    state_set: List[int]
    s0: int
    r: int
    truth_table: Dict[Tuple, int]
    name: str = ""
    
    def __post_init__(self):
        self.S = self.state_set
        self.n_states = len(self.S)
        self.v_minus, self.v_plus = self._compute_propagation_speeds()
    
    def _compute_propagation_speeds(self) -> Tuple[int, int]:
        v_plus = 0
        v_minus = 0
        for j in range(-self.r, self.r + 1):
            if self._is_effective_dependency(j):
                if j > 0:
                    v_plus = max(v_plus, j)
                elif j < 0:
                    v_minus = max(v_minus, abs(j))
        return v_minus, v_plus
    
    def _is_effective_dependency(self, j: int) -> bool:
        neighborhood_size = 2 * self.r + 1
        for combo in product(self.S, repeat=neighborhood_size):
            a = list(combo)
            for alt_val in self.S:
                if alt_val == a[j + self.r]:
                    continue
                b = a.copy()
                b[j + self.r] = alt_val
                key_a = tuple(a)
                key_b = tuple(b)
                if key_a in self.truth_table and key_b in self.truth_table:
                    if self.truth_table[key_a] != self.truth_table[key_b]:
                        return True
        return False
    
    def apply(self, neighborhood: Tuple) -> int:
        return self.truth_table.get(neighborhood, self.s0)
    
    def __repr__(self):
        return f"NonlinearRule({self.name}, r={self.r}, v-={self.v_minus}, v+={self.v_plus})"


def create_rule_90() -> NonlinearRule:
    S = [0, 1]
    s0 = 0
    r = 1
    tt = {}
    for combo in product(S, repeat=3):
        tt[combo] = (combo[0] + combo[2]) % 2
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=tt, name="Rule90")


def create_rule_150() -> NonlinearRule:
    S = [0, 1]
    s0 = 0
    r = 1
    tt = {}
    for combo in product(S, repeat=3):
        tt[combo] = (combo[0] + combo[1] + combo[2]) % 2
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=tt, name="Rule150")


def create_rule_110() -> NonlinearRule:
    S = [0, 1]
    s0 = 0
    r = 1
    rule_map = {
        (1,1,1): 0, (1,1,0): 1, (1,0,1): 1, (1,0,0): 0,
        (0,1,1): 1, (0,1,0): 1, (0,0,1): 1, (0,0,0): 0
    }
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=rule_map, name="Rule110")


def create_rule_30() -> NonlinearRule:
    S = [0, 1]
    s0 = 0
    r = 1
    rule_map = {
        (1,1,1): 0, (1,1,0): 0, (1,0,1): 0, (1,0,0): 1,
        (0,1,1): 1, (0,1,0): 1, (0,0,1): 1, (0,0,0): 0
    }
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=rule_map, name="Rule30")


def create_rule_184() -> NonlinearRule:
    S = [0, 1]
    s0 = 0
    r = 1
    rule_map = {
        (1,1,1): 1, (1,1,0): 0, (1,0,1): 1, (1,0,0): 1,
        (0,1,1): 1, (0,1,0): 0, (0,0,1): 0, (0,0,0): 0
    }
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=rule_map, name="Rule184")


def create_asymmetric_rule() -> NonlinearRule:
    S = [0, 1]
    s0 = 0
    r = 2
    tt = {}
    for combo in product(S, repeat=5):
        tt[combo] = combo[0]
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=tt, name="Asymmetric_v-2_v+0")


def create_shift_right_rule() -> NonlinearRule:
    S = [0, 1]
    s0 = 0
    r = 1
    tt = {}
    for combo in product(S, repeat=3):
        tt[combo] = combo[0]
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=tt, name="ShiftRight_v-1_v+0")


def create_3state_rule() -> NonlinearRule:
    S = [0, 1, 2]
    s0 = 0
    r = 1
    tt = {}
    for combo in product(S, repeat=3):
        total = sum(combo)
        if total == 0:
            tt[combo] = 0
        elif total <= 2:
            tt[combo] = 1
        else:
            tt[combo] = 2
    return NonlinearRule(state_set=S, s0=s0, r=r, truth_table=tt, name="3StateThreshold")


class CAEvolver:
    def __init__(self, rule: NonlinearRule):
        self.rule = rule
    
    def evolve(self, config: np.ndarray) -> np.ndarray:
        n = len(config)
        new_config = np.full(n, self.rule.s0, dtype=int)
        for i in range(self.rule.r, n - self.rule.r):
            neighborhood = tuple(config[i - self.rule.r: i + self.rule.r + 1])
            new_config[i] = self.rule.apply(neighborhood)
        return new_config
    
    def evolve_steps(self, config: np.ndarray, steps: int) -> List[np.ndarray]:
        trajectory = [config.copy()]
        current = config.copy()
        for _ in range(steps):
            current = self.evolve(current)
            trajectory.append(current.copy())
        return trajectory


class ObservationBlockCode:
    def __init__(self, rule: NonlinearRule, T: int, c: int):
        self.rule = rule
        self.T = T
        self.c = c
        self.v = max(rule.v_minus, rule.v_plus)
        self.A_T_half = (c + self.v) * T
        self.A_T = list(range(-self.A_T_half, self.A_T_half + 1))
        self.n_T = len(self.A_T)
        self.E_T_left = -c * T
        self.E_T_right = c * T
        self.E_T = list(range(self.E_T_left, self.E_T_right + 1))
        self.I_infty = self._compute_hidden_cells()
        self.kappa = 2 * self.v - rule.v_minus - rule.v_plus
    
    def _compute_hidden_cells(self) -> List[int]:
        hidden = []
        for i in self.A_T:
            if self._is_hidden(i):
                hidden.append(i)
        return hidden
    
    def _is_hidden(self, i: int) -> bool:
        for t in range(self.T + 1):
            influence_left = i - self.rule.v_plus * t
            influence_right = i + self.rule.v_minus * t
            obs_left = -self.c * self.T
            obs_right = self.c * self.T
            if influence_right >= obs_left and influence_left <= obs_right:
                return False
        return True
    
    def compute_tau_geo(self, i: int) -> int:
        if abs(i) <= self.c * self.T:
            return 0
        if i < -self.c * self.T and self.rule.v_plus > 0:
            tau = math.ceil((-self.c * self.T - i) / self.rule.v_plus)
            return tau if tau <= self.T else float('inf')
        if i > self.c * self.T and self.rule.v_minus > 0:
            tau = math.ceil((i - self.c * self.T) / self.rule.v_minus)
            return tau if tau <= self.T else float('inf')
        return float('inf')
    
    def config_to_array(self, config_dict: Dict[int, int]) -> np.ndarray:
        arr = np.full(self.n_T, self.rule.s0, dtype=int)
        center = self.A_T_half
        for i, val in config_dict.items():
            idx = center + i
            if 0 <= idx < self.n_T:
                arr[idx] = val
        return arr
    
    def array_to_dict(self, arr: np.ndarray) -> Dict[int, int]:
        center = self.A_T_half
        result = {}
        for idx, val in enumerate(arr):
            i = idx - center
            if val != self.rule.s0:
                result[i] = val
        return result
    
    def observe(self, config: np.ndarray, t: int) -> np.ndarray:
        center = self.A_T_half
        left = center - self.c * self.T
        right = center + self.c * self.T + 1
        return config[left:right].copy()
    
    def compute_observation_block(self, config: np.ndarray) -> Tuple:
        evolver = CAEvolver(self.rule)
        trajectory = evolver.evolve_steps(config, self.T)
        observations = []
        for t in range(self.T + 1):
            obs = self.observe(trajectory[t], t)
            observations.append(tuple(obs))
        return tuple(observations)
    
    def compute_causal_past(self, t: int, p: int) -> List[int]:
        left = p - self.rule.v_plus * t
        right = p + self.rule.v_minus * t
        return list(range(left, right + 1))
    
    def compute_causal_past_set(self, t: int, J: List[int]) -> List[int]:
        result = set()
        for p in J:
            result.update(self.compute_causal_past(t, p))
        return sorted(result)


class EquivalenceRefiner:
    def __init__(self, rule: NonlinearRule, T: int, c: int):
        self.rule = rule
        self.T = T
        self.c = c
        self.obs = ObservationBlockCode(rule, T, c)
    
    def enumerate_configs(self, max_configs: int = 10000) -> List[np.ndarray]:
        non_hidden = [i for i in self.obs.A_T if i not in self.obs.I_infty]
        n_free = len(non_hidden)
        total = self.rule.n_states ** n_free
        
        if total > max_configs:
            configs = []
            for _ in range(max_configs):
                arr = np.full(self.obs.n_T, self.rule.s0, dtype=int)
                center = self.obs.A_T_half
                for i in non_hidden:
                    idx = center + i
                    if 0 <= idx < self.obs.n_T:
                        arr[idx] = np.random.choice(self.rule.S)
                configs.append(arr)
            return configs
        
        configs = []
        for assignment in product(self.rule.S, repeat=n_free):
            arr = np.full(self.obs.n_T, self.rule.s0, dtype=int)
            center = self.obs.A_T_half
            for i, val in zip(non_hidden, assignment):
                idx = center + i
                if 0 <= idx < self.obs.n_T:
                    arr[idx] = val
            configs.append(arr)
        return configs
    
    def compute_partition(self, configs: List[np.ndarray]) -> Dict[Tuple, List[int]]:
        blocks = {}
        for idx, config in enumerate(configs):
            obs_block = self.obs.compute_observation_block(config)
            if obs_block not in blocks:
                blocks[obs_block] = []
            blocks[obs_block].append(idx)
        return blocks
    
    def compute_delta_max(self, configs: List[np.ndarray]) -> float:
        blocks = self.compute_partition(configs)
        if not blocks:
            return 0.0
        max_block_size = max(len(indices) for indices in blocks.values())
        if max_block_size <= 1:
            return 0.0
        return math.log(max_block_size) / math.log(self.rule.n_states)
    
    def compute_delta_max_full(self) -> float:
        non_hidden = [i for i in self.obs.A_T if i not in self.obs.I_infty]
        n_free = len(non_hidden)
        total = self.rule.n_states ** n_free
        
        if total > 50000:
            configs = self.enumerate_configs(max_configs=5000)
            return self.compute_delta_max(configs)
        
        configs = self.enumerate_configs(max_configs=total + 1)
        return self.compute_delta_max(configs)
    
    def is_injective(self, configs: List[np.ndarray]) -> bool:
        blocks = self.compute_partition(configs)
        return all(len(indices) == 1 for indices in blocks.values())
    
    def compute_layered_partition(self, config: np.ndarray, max_t: int = None) -> List[Dict]:
        if max_t is None:
            max_t = self.T
        evolver = CAEvolver(self.rule)
        trajectory = evolver.evolve_steps(config, max_t)
        
        partitions = []
        for t in range(max_t + 1):
            obs_at_t = self.obs.observe(trajectory[t], t)
            partitions.append(tuple(obs_at_t))
        return partitions


class ExcitedStateAnalyzer:
    def __init__(self, rule: NonlinearRule, T: int, c: int):
        self.rule = rule
        self.T = T
        self.c = c
        self.obs = ObservationBlockCode(rule, T, c)
    
    def is_visible(self, A: List[int], sigma: Dict[int, int]) -> bool:
        config = np.full(self.obs.n_T, self.rule.s0, dtype=int)
        center = self.obs.A_T_half
        for i in A:
            idx = center + i
            if 0 <= idx < self.obs.n_T:
                config[idx] = sigma.get(i, self.rule.s0)
        
        zero_config = np.full(self.obs.n_T, self.rule.s0, dtype=int)
        
        obs_e = self.obs.compute_observation_block(config)
        obs_0 = self.obs.compute_observation_block(zero_config)
        
        return obs_e != obs_0
    
    def find_minimal_trigger_clusters(self, max_search: int = 1000) -> List[Tuple]:
        clusters = []
        non_hidden = [i for i in self.obs.A_T if i not in self.obs.I_infty]
        
        for size in range(1, min(4, len(non_hidden) + 1)):
            from itertools import combinations
            for A_tuple in combinations(non_hidden, size):
                A = list(A_tuple)
                for sigma_assignment in product(self.rule.S, repeat=size):
                    if all(v == self.rule.s0 for v in sigma_assignment):
                        continue
                    sigma = dict(zip(A, sigma_assignment))
                    if self.is_visible(A, sigma):
                        is_minimal = True
                        for sub_size in range(1, size):
                            for sub_A_tuple in combinations(A, sub_size):
                                sub_A = list(sub_A_tuple)
                                for sub_sigma_assignment in product(self.rule.S, repeat=sub_size):
                                    if all(v == self.rule.s0 for v in sub_sigma_assignment):
                                        continue
                                    sub_sigma = dict(zip(sub_A, sub_sigma_assignment))
                                    if self.is_visible(sub_A, sub_sigma):
                                        is_minimal = False
                                        break
                                if not is_minimal:
                                    break
                            if not is_minimal:
                                break
                        if is_minimal:
                            clusters.append((A, sigma))
                            if len(clusters) >= max_search:
                                return clusters
        return clusters
    
    def compute_first_visible_info(self, A: List[int], sigma: Dict[int, int]) -> Optional[Tuple]:
        config = np.full(self.obs.n_T, self.rule.s0, dtype=int)
        center = self.obs.A_T_half
        for i in A:
            idx = center + i
            if 0 <= idx < self.obs.n_T:
                config[idx] = sigma.get(i, self.rule.s0)
        
        zero_config = np.full(self.obs.n_T, self.rule.s0, dtype=int)
        
        evolver = CAEvolver(self.rule)
        traj_e = evolver.evolve_steps(config, self.T)
        traj_0 = evolver.evolve_steps(zero_config, self.T)
        
        for t in range(self.T + 1):
            obs_e = self.obs.observe(traj_e[t], t)
            obs_0 = self.obs.observe(traj_0[t], t)
            if not np.array_equal(obs_e, obs_0):
                J = []
                phi = {}
                for j_idx in range(len(obs_e)):
                    if obs_e[j_idx] != obs_0[j_idx]:
                        actual_pos = j_idx - self.c * self.T
                        J.append(actual_pos)
                        phi[actual_pos] = obs_e[j_idx]
                return (t, J, phi)
        return None
    
    def is_decodable_support(self, A: List[int]) -> bool:
        visible_sigmas = []
        first_visible_infos = []
        
        for sigma_assignment in product(self.rule.S, repeat=len(A)):
            if all(v == self.rule.s0 for v in sigma_assignment):
                continue
            sigma = dict(zip(A, sigma_assignment))
            if self.is_visible(A, sigma):
                info = self.compute_first_visible_info(A, sigma)
                if info is not None:
                    visible_sigmas.append(sigma)
                    first_visible_infos.append(info)
        
        if len(first_visible_infos) <= 1:
            return True
        
        for i in range(len(first_visible_infos)):
            for j in range(i + 1, len(first_visible_infos)):
                if first_visible_infos[i] == first_visible_infos[j]:
                    return False
        return True
    
    def check_causal_collision(self, e1: Tuple, e2: Tuple) -> bool:
        A1, sigma1 = e1
        A2, sigma2 = e2
        
        info1 = self.compute_first_visible_info(A1, sigma1)
        if info1 is None:
            return False
        
        tau1, J1, _ = info1
        causal_past = self.obs.compute_causal_past_set(tau1, J1)
        
        for i in A2:
            if i in causal_past:
                return True
        return False


class HiddenDOFModel:
    def __init__(self, T: int):
        self.T = T
        self.S_size = 4
        self.E_T_size = 2 * T + 1
    
    def compute_h_top(self) -> float:
        return math.log(2)
    
    def compute_h_fib(self) -> float:
        return math.log(2)
    
    def compute_h_obs(self) -> float:
        return 2 * math.log(2)
    
    def compute_Delta_max(self) -> float:
        return (2 * self.T + 1) * math.log(2)
    
    def compute_rho(self) -> float:
        density = self.compute_Delta_max() / self.E_T_size
        h_top_over_logS = self.compute_h_top() / math.log(self.S_size)
        return density - h_top_over_logS
    
    def verify_conjecture_2_2(self) -> Dict:
        h_fib = self.compute_h_fib()
        h_obs = self.compute_h_obs()
        kappa = 0
        
        is_tight = abs(h_fib - (h_obs + kappa * math.log(self.S_size))) < 1e-10
        rho = self.compute_rho()
        is_rho_positive = rho > 0
        
        return {
            'h_fib': h_fib,
            'h_obs': h_obs,
            'kappa': kappa,
            'is_tight': is_tight,
            'rho': rho,
            'rho_positive': is_rho_positive,
            'conjecture_2_2_holds': (not is_tight) and is_rho_positive
        }


if __name__ == "__main__":
    print("=" * 70)
    print("非线性验证核心引擎 - 基础测试")
    print("=" * 70)
    
    for rule_fn in [create_rule_90, create_rule_110, create_rule_30, create_rule_184]:
        rule = rule_fn()
        print(f"\n{rule}")
    
    rule110 = create_rule_110()
    obs = ObservationBlockCode(rule110, T=2, c=1)
    print(f"\nRule 110, T=2, c=1:")
    print(f"  A_T = [{obs.A_T[0]},...,{obs.A_T[-1]}], |A_T|={obs.n_T}")
    print(f"  E_T = [{obs.E_T_left},...,{obs.E_T_right}]")
    print(f"  I_∞ = {obs.I_infty}")
    print(f"  κ = {obs.kappa}")
    
    refiner = EquivalenceRefiner(rule110, T=2, c=1)
    delta = refiner.compute_delta_max_full()
    print(f"  δ_max = {delta:.4f}")
    
    model = HiddenDOFModel(T=10)
    result = model.verify_conjecture_2_2()
    print(f"\n隐藏自由度模型 (T=10):")
    for k, v in result.items():
        print(f"  {k}: {v}")
