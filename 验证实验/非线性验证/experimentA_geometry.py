"""
实验A：有效传播速度计算与几何障碍验证
验证文档v12中的：引理2.1(有限传播)、定义2(有效依赖)、命题3.1(几何隐蔽下界)
验证文档v15中的：定义1.1(最小依赖偏移)、引理1.4(溢出集无干扰)、命题1.5(溢出集大小)
"""
import numpy as np
import sys, os, math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nonlinear_core import *


class ExperimentA:
    def __init__(self):
        self.results = []
    
    def test_effective_dependencies(self, rule: NonlinearRule, 
                                    expected_v_minus: int, expected_v_plus: int,
                                    case_name: str) -> Dict:
        actual_v_minus = rule.v_minus
        actual_v_plus = rule.v_plus
        
        result = {
            'case_name': case_name,
            'rule_name': rule.name,
            'r': rule.r,
            'expected_v_minus': expected_v_minus,
            'expected_v_plus': expected_v_plus,
            'actual_v_minus': actual_v_minus,
            'actual_v_plus': actual_v_plus,
            'v_correct': (actual_v_minus == expected_v_minus and actual_v_plus == expected_v_plus)
        }
        self.results.append(result)
        return result
    
    def test_finite_propagation(self, rule: NonlinearRule, T: int, c: int,
                               num_tests: int = 50) -> Dict:
        obs = ObservationBlockCode(rule, T, c)
        evolver = CAEvolver(rule)
        
        violations = 0
        for _ in range(num_tests):
            config = np.full(obs.n_T, rule.s0, dtype=int)
            center = obs.A_T_half
            num_nonzero = np.random.randint(1, min(5, len(obs.A_T)) + 1)
            positions = np.random.choice(obs.A_T, size=num_nonzero, replace=False)
            for pos in positions:
                idx = center + pos
                if 0 <= idx < obs.n_T:
                    config[idx] = np.random.choice(rule.S)
            
            support = set()
            for idx in range(obs.n_T):
                if config[idx] != rule.s0:
                    support.add(idx - center)
            
            trajectory = evolver.evolve_steps(config, T)
            
            for t in range(T + 1):
                actual_support = set()
                for idx in range(obs.n_T):
                    if trajectory[t][idx] != rule.s0:
                        actual_support.add(idx - center)
                
                for i in actual_support:
                    in_cone = False
                    for k in support:
                        if k - rule.v_minus * t <= i <= k + rule.v_plus * t:
                            in_cone = True
                            break
                    if not in_cone:
                        violations += 1
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'num_tests': num_tests,
            'violations': violations,
            'lemma_holds': violations == 0
        }
        self.results.append(result)
        return result
    
    def test_hidden_cells_proposition_3_1(self, rule: NonlinearRule, T: int, c: int,
                                          num_tests: int = 100) -> Dict:
        obs = ObservationBlockCode(rule, T, c)
        evolver = CAEvolver(rule)
        
        if not obs.I_infty:
            return {
                'rule_name': rule.name, 'T': T, 'c': c,
                'I_infty_size': 0, 'proposition_holds': True,
                'note': 'No hidden cells'
            }
        
        violations = 0
        for i in obs.I_infty:
            for _ in range(num_tests // len(obs.I_infty)):
                config = np.full(obs.n_T, rule.s0, dtype=int)
                center = obs.A_T_half
                for pos in obs.A_T:
                    if pos != i:
                        idx = center + pos
                        if 0 <= idx < obs.n_T:
                            config[idx] = np.random.choice(rule.S)
                
                obs_block_orig = obs.compute_observation_block(config)
                
                for alt_state in rule.S:
                    if alt_state == rule.s0:
                        continue
                    config_alt = config.copy()
                    idx = center + i
                    config_alt[idx] = alt_state
                    obs_block_alt = obs.compute_observation_block(config_alt)
                    
                    if obs_block_orig != obs_block_alt:
                        violations += 1
        
        result = {
            'rule_name': rule.name, 'T': T, 'c': c,
            'I_infty_size': len(obs.I_infty),
            'violations': violations,
            'proposition_holds': violations == 0
        }
        self.results.append(result)
        return result
    
    def test_overflow_set_size(self, rule: NonlinearRule, T_values: List[int], c: int) -> Dict:
        v = max(rule.v_minus, rule.v_plus)
        expected_kappa = 2 * v - rule.v_minus - rule.v_plus
        
        computed_sizes = []
        expected_sizes = []
        
        for T in T_values:
            obs = ObservationBlockCode(rule, T, c)
            actual_size = len(obs.I_infty)
            expected_size = T * expected_kappa
            computed_sizes.append(actual_size)
            expected_sizes.append(expected_size)
        
        kappa_estimates = []
        for i, T in enumerate(T_values):
            if T > 0:
                kappa_estimates.append(computed_sizes[i] / T)
        
        avg_kappa = np.mean(kappa_estimates) if kappa_estimates else 0
        
        result = {
            'rule_name': rule.name,
            'v_minus': rule.v_minus,
            'v_plus': rule.v_plus,
            'v': v,
            'expected_kappa': expected_kappa,
            'computed_kappa': avg_kappa,
            'kappa_correct': abs(avg_kappa - expected_kappa) < 0.5,
            'T_values': T_values,
            'computed_sizes': computed_sizes,
            'expected_sizes': expected_sizes
        }
        self.results.append(result)
        return result
    
    def test_no_interference_lemma(self, rule: NonlinearRule, T: int, c: int,
                                   num_tests: int = 50) -> Dict:
        obs = ObservationBlockCode(rule, T, c)
        evolver = CAEvolver(rule)
        
        if not obs.I_infty:
            return {
                'rule_name': rule.name, 'T': T, 'c': c,
                'I_infty_size': 0, 'violations': 0, 'lemma_holds': True,
                'note': 'No overflow cells'
            }
        
        violations = 0
        center = obs.A_T_half
        
        for _ in range(num_tests):
            config = np.full(obs.n_T, rule.s0, dtype=int)
            for pos in obs.A_T:
                idx = center + pos
                if 0 <= idx < obs.n_T:
                    config[idx] = np.random.choice(rule.S)
            
            obs_orig = obs.compute_observation_block(config)
            
            for i in obs.I_infty[:3]:
                for alt_state in rule.S:
                    if alt_state == config[center + i]:
                        continue
                    config_alt = config.copy()
                    config_alt[center + i] = alt_state
                    obs_alt = obs.compute_observation_block(config_alt)
                    if obs_orig != obs_alt:
                        violations += 1
        
        result = {
            'rule_name': rule.name, 'T': T, 'c': c,
            'I_infty_size': len(obs.I_infty),
            'violations': violations,
            'lemma_holds': violations == 0
        }
        self.results.append(result)
        return result
    
    def run_all(self):
        print("=" * 70)
        print("实验A：有效传播速度与几何障碍验证")
        print("=" * 70)
        
        print("\n--- 测试1: 有效依赖与传播速度 ---")
        test_cases = [
            (create_rule_90, 1, 1, "Rule 90"),
            (create_rule_150, 1, 1, "Rule 150"),
            (create_rule_110, 1, 1, "Rule 110"),
            (create_rule_30, 1, 1, "Rule 30"),
            (create_rule_184, 1, 1, "Rule 184"),
        ]
        for fn, evm, evp, name in test_cases:
            r = self.test_effective_dependencies(fn(), evm, evp, name)
            status = "✓" if r['v_correct'] else "✗"
            print(f"  {name}: v-={r['actual_v_minus']}, v+={r['actual_v_plus']} {status}")
        
        print("\n--- 测试2: 有限传播引理 ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110"), 
                         (create_rule_30, "Rule30")]:
            r = self.test_finite_propagation(fn(), T=3, c=1, num_tests=30)
            status = "✓" if r['lemma_holds'] else "✗"
            print(f"  {name}: violations={r['violations']} {status}")
        
        print("\n--- 测试3: 几何隐蔽下界(命题3.1) ---")
        shift_rule = create_shift_right_rule()
        asym_rule = create_asymmetric_rule()
        for rule, name in [(shift_rule, "ShiftRight"), (asym_rule, "Asymmetric_r2")]:
            for T, c in [(2, 1), (3, 1)]:
                r = self.test_hidden_cells_proposition_3_1(rule, T, c, num_tests=30)
                status = "✓" if r['proposition_holds'] else "✗"
                print(f"  {name} T={T} c={c}: |I_∞|={r['I_infty_size']} {status}")
        
        print("\n--- 测试4: 溢出集大小公式(命题1.5) ---")
        for rule, name in [(shift_rule, "ShiftRight_v-1_v+0"), (asym_rule, "Asymmetric_v-2_v+0")]:
            r = self.test_overflow_set_size(rule, T_values=[3,5,7,10], c=1)
            status = "✓" if r['kappa_correct'] else "✗"
            print(f"  {name}: κ_expected={r['expected_kappa']}, κ_computed={r['computed_kappa']:.2f} {status}")
        
        print("\n--- 测试5: 溢出集无干扰引理(引理1.4) ---")
        for rule, name in [(shift_rule, "ShiftRight"), (asym_rule, "Asymmetric_r2")]:
            for T, c in [(3, 1), (5, 1)]:
                r = self.test_no_interference_lemma(rule, T, c, num_tests=30)
                status = "✓" if r['lemma_holds'] else "✗"
                print(f"  {name} T={T} c={c}: |I_∞|={r['I_infty_size']}, violations={r['violations']} {status}")
        
        self.save_results()
    
    def save_results(self):
        with open("experimentA_results.txt", 'w', encoding='utf-8') as f:
            f.write("实验A：有效传播速度与几何障碍验证结果\n")
            f.write("=" * 70 + "\n\n")
            for r in self.results:
                f.write(f"{r}\n\n")


if __name__ == "__main__":
    np.random.seed(42)
    exp = ExperimentA()
    exp.run_all()
