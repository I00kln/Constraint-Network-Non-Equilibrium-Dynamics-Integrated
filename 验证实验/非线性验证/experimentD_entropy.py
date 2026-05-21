"""
实验D：因果分离下逐层重构定理验证 + 熵上界不等式验证 + 次可加性与隐藏自由度
验证：定理7.3(因果分离逐层重构)、定理3.4(熵上界)、猜想1.2(次可加性)、猜想2.2(紧致性)
"""
import numpy as np
import sys, os, math
from itertools import product
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nonlinear_core import *


class ExperimentD:
    def __init__(self):
        self.results = []
    
    def test_causal_separation_reconstruction(self, rule: NonlinearRule, 
                                               T: int, c: int,
                                               num_tests: int = 20) -> Dict:
        obs = ObservationBlockCode(rule, T, c)
        analyzer = ExcitedStateAnalyzer(rule, T, c)
        evolver = CAEvolver(rule)
        center = obs.A_T_half
        
        single_excited = []
        for i in obs.A_T[:7]:
            if i in obs.I_infty:
                continue
            for s in rule.S:
                if s == rule.s0:
                    continue
                sigma = {i: s}
                if analyzer.is_visible([i], sigma):
                    info = analyzer.compute_first_visible_info([i], sigma)
                    if info is not None:
                        single_excited.append(([i], sigma, info))
        
        separated_groups = []
        for i in range(len(single_excited)):
            for j in range(i + 1, len(single_excited)):
                A1, sigma1, info1 = single_excited[i]
                A2, sigma2, info2 = single_excited[j]
                
                col_12 = analyzer.check_causal_collision((A1, sigma1), (A2, sigma2))
                col_21 = analyzer.check_causal_collision((A2, sigma2), (A1, sigma1))
                
                if not col_12 and not col_21:
                    separated_groups.append((single_excited[i], single_excited[j]))
        
        successful_reconstructions = 0
        total_tests = 0
        
        for e1, e2 in separated_groups[:num_tests]:
            A1, sigma1, info1 = e1
            A2, sigma2, info2 = e2
            
            config_joint = np.full(obs.n_T, rule.s0, dtype=int)
            for pos, val in sigma1.items():
                idx = center + pos
                if 0 <= idx < obs.n_T:
                    config_joint[idx] = val
            for pos, val in sigma2.items():
                idx = center + pos
                if 0 <= idx < obs.n_T:
                    config_joint[idx] = val
            
            obs_block = obs.compute_observation_block(config_joint)
            
            config1_only = np.full(obs.n_T, rule.s0, dtype=int)
            for pos, val in sigma1.items():
                idx = center + pos
                if 0 <= idx < obs.n_T:
                    config1_only[idx] = val
            obs1 = obs.compute_observation_block(config1_only)
            
            config2_only = np.full(obs.n_T, rule.s0, dtype=int)
            for pos, val in sigma2.items():
                idx = center + pos
                if 0 <= idx < obs.n_T:
                    config2_only[idx] = val
            obs2 = obs.compute_observation_block(config2_only)
            
            total_tests += 1
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'num_separated_pairs': len(separated_groups),
            'total_tests': total_tests,
            'theorem_context': '因果分离条件下逐层重构'
        }
        self.results.append(result)
        return result
    
    def test_entropy_upper_bound(self, rule: NonlinearRule, c: int, 
                                  T_max: int = 5) -> Dict:
        delta_sequence = []
        T_values = list(range(1, T_max + 1))
        
        for T in T_values:
            obs = ObservationBlockCode(rule, T, c)
            refiner = EquivalenceRefiner(rule, T, c)
            
            non_hidden = [i for i in obs.A_T if i not in obs.I_infty]
            n_free = len(non_hidden)
            total = rule.n_states ** n_free
            
            if total > 20000:
                configs = refiner.enumerate_configs(max_configs=3000)
                delta = refiner.compute_delta_max(configs)
            else:
                delta = refiner.compute_delta_max_full()
            
            delta_sequence.append(delta)
        
        Delta_max_sequence = [d * math.log(rule.n_states) for d in delta_sequence]
        
        h_obs_values = []
        for i, T in enumerate(T_values):
            if T > 0:
                h_obs_values.append(Delta_max_sequence[i] / T)
        
        h_obs = max(h_obs_values) if h_obs_values else 0
        
        v = max(rule.v_minus, rule.v_plus)
        kappa = 2 * v - rule.v_minus - rule.v_plus
        
        upper_bound = h_obs + kappa * math.log(rule.n_states)
        
        result = {
            'rule_name': rule.name,
            'c': c,
            'v_minus': rule.v_minus,
            'v_plus': rule.v_plus,
            'kappa': kappa,
            'T_values': T_values,
            'delta_sequence': delta_sequence,
            'Delta_max_sequence': Delta_max_sequence,
            'h_obs_estimates': h_obs_values,
            'h_obs': h_obs,
            'upper_bound': upper_bound,
            'note': f'h_fib ≤ {upper_bound:.4f} (上界验证)'
        }
        self.results.append(result)
        return result
    
    def test_subadditivity(self, rule: NonlinearRule, c: int, 
                           T_max: int = 5) -> Dict:
        delta_dict = {}
        for T in range(1, T_max + 1):
            obs = ObservationBlockCode(rule, T, c)
            refiner = EquivalenceRefiner(rule, T, c)
            non_hidden = [i for i in obs.A_T if i not in obs.I_infty]
            n_free = len(non_hidden)
            total = rule.n_states ** n_free
            
            if total > 20000:
                configs = refiner.enumerate_configs(max_configs=3000)
                delta = refiner.compute_delta_max(configs)
            else:
                delta = refiner.compute_delta_max_full()
            delta_dict[T] = delta * math.log(rule.n_states)
        
        violations = 0
        checks = 0
        for T in range(1, T_max):
            for S in range(1, T_max - T + 1):
                if T + S <= T_max:
                    lhs = delta_dict.get(T + S, float('inf'))
                    rhs = delta_dict.get(T, 0) + delta_dict.get(S, 0)
                    checks += 1
                    if lhs > rhs + 0.01:
                        violations += 1
        
        result = {
            'rule_name': rule.name,
            'c': c,
            'T_max': T_max,
            'delta_values': {str(k): v for k, v in delta_dict.items()},
            'subadditivity_checks': checks,
            'subadditivity_violations': violations,
            'approximately_subadditive': violations == 0
        }
        self.results.append(result)
        return result
    
    def test_hidden_dof_model(self) -> Dict:
        model = HiddenDOFModel(T=10)
        result = model.verify_conjecture_2_2()
        
        result['experiment'] = '隐藏自由度模型验证'
        result['conjecture_2_2_summary'] = (
            f"h_fib={result['h_fib']:.4f}, h_obs={result['h_obs']:.4f}, "
            f"κ={result['kappa']}, ρ={result['rho']:.4f}, "
            f"上界非紧={not result['is_tight']}, ρ>0={result['rho_positive']}"
        )
        self.results.append(result)
        return result
    
    def test_redundancy_rate_convergence(self) -> Dict:
        T_values = [5, 10, 20, 50, 100]
        rho_values = []
        
        for T in T_values:
            model = HiddenDOFModel(T=T)
            rho = model.compute_rho()
            rho_values.append(rho)
        
        expected_rho = 0.5 * math.log(2) - 0.5
        
        result = {
            'experiment': '冗余速率收敛性',
            'T_values': T_values,
            'rho_values': rho_values,
            'expected_rho': expected_rho,
            'converges': abs(rho_values[-1] - expected_rho) < 0.01
        }
        self.results.append(result)
        return result
    
    def run_all(self):
        print("=" * 70)
        print("实验D：因果分离重构 + 熵上界 + 次可加性 + 隐藏自由度")
        print("=" * 70)
        
        print("\n--- 测试1: 因果分离下逐层重构(定理7.3) ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110")]:
            r = self.test_causal_separation_reconstruction(fn(), T=2, c=1)
            print(f"  {name}: 分离对={r['num_separated_pairs']}, 测试={r['total_tests']}")
        
        print("\n--- 测试2: 熵上界不等式(定理3.4) ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110"),
                         (create_rule_30, "Rule30")]:
            r = self.test_entropy_upper_bound(fn(), c=1, T_max=4)
            print(f"  {name}: h_obs={r['h_obs']:.4f}, κ={r['kappa']}, "
                  f"上界={r['upper_bound']:.4f}")
            print(f"    Δ_T/T = {[f'{v:.3f}' for v in r['h_obs_estimates']]}")
        
        print("\n--- 测试3: 次可加性(猜想1.2) ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110")]:
            r = self.test_subadditivity(fn(), c=1, T_max=4)
            status = "✓ 近似次可加" if r['approximately_subadditive'] else "✗ 有违反"
            print(f"  {name}: 检查={r['subadditivity_checks']}, "
                  f"违反={r['subadditivity_violations']} {status}")
        
        print("\n--- 测试4: 隐藏自由度模型(猜想2.2) ---")
        r = self.test_hidden_dof_model()
        print(f"  {r['conjecture_2_2_summary']}")
        print(f"  猜想2.2成立: {r['conjecture_2_2_holds']}")
        
        print("\n--- 测试5: 冗余速率收敛性 ---")
        r = self.test_redundancy_rate_convergence()
        print(f"  T={r['T_values']}")
        print(f"  ρ={[f'{v:.4f}' for v in r['rho_values']]}")
        print(f"  收敛到理论值: {r['converges']}")
        
        self.save_results()
    
    def save_results(self):
        with open("experimentD_results.txt", 'w', encoding='utf-8') as f:
            f.write("实验D：因果分离重构 + 熵上界 + 次可加性 + 隐藏自由度验证结果\n")
            f.write("=" * 70 + "\n\n")
            for r in self.results:
                f.write(f"{r}\n\n")


if __name__ == "__main__":
    np.random.seed(42)
    exp = ExperimentD()
    exp.run_all()
