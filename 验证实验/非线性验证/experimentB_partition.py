"""
实验B：观测等价类逐层细化与信息亏损计算
验证：命题4.1(可观测性划分刻画)、定义4.2(信息亏损)、定理8.1(结构化判定)
"""
import numpy as np
import sys, os, math
from itertools import product
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nonlinear_core import *


class ExperimentB:
    def __init__(self):
        self.results = []
    
    def compute_partition_and_defect(self, rule: NonlinearRule, T: int, c: int) -> Dict:
        obs = ObservationBlockCode(rule, T, c)
        refiner = EquivalenceRefiner(rule, T, c)
        
        non_hidden = [i for i in obs.A_T if i not in obs.I_infty]
        n_free = len(non_hidden)
        total_configs = rule.n_states ** n_free
        
        if total_configs > 20000:
            configs = refiner.enumerate_configs(max_configs=5000)
            blocks = refiner.compute_partition(configs)
            delta_max = refiner.compute_delta_max(configs)
            is_injective = refiner.is_injective(configs)
            sampled = True
        else:
            configs = refiner.enumerate_configs(max_configs=total_configs + 1)
            blocks = refiner.compute_partition(configs)
            delta_max = refiner.compute_delta_max(configs)
            is_injective = refiner.is_injective(configs)
            sampled = False
        
        max_fiber_size = max(len(v) for v in blocks.values()) if blocks else 0
        num_blocks = len(blocks)
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'n_free': n_free,
            'total_configs': total_configs,
            'sampled': sampled,
            'num_equivalence_classes': num_blocks,
            'max_fiber_size': max_fiber_size,
            'delta_max': delta_max,
            'I_infty_size': len(obs.I_infty),
            'is_injective': is_injective,
            'complete_obs': len(obs.I_infty) == 0 and delta_max == 0
        }
        self.results.append(result)
        return result
    
    def verify_proposition_4_1(self, rule: NonlinearRule, T: int, c: int) -> Dict:
        obs = ObservationBlockCode(rule, T, c)
        refiner = EquivalenceRefiner(rule, T, c)
        
        non_hidden = [i for i in obs.A_T if i not in obs.I_infty]
        n_free = len(non_hidden)
        total_configs = rule.n_states ** n_free
        
        if total_configs > 20000:
            return {
                'rule_name': rule.name, 'T': T, 'c': c,
                'status': 'skipped_too_large',
                'total_configs': total_configs
            }
        
        configs = refiner.enumerate_configs(max_configs=total_configs + 1)
        blocks = refiner.compute_partition(configs)
        
        all_singletons = all(len(v) == 1 for v in blocks.values())
        no_hidden = len(obs.I_infty) == 0
        
        is_injective = no_hidden and all_singletons
        
        cross_check_injective = True
        for indices in blocks.values():
            if len(indices) > 1:
                cross_check_injective = False
                break
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'no_hidden': no_hidden,
            'all_singletons': all_singletons,
            'is_injective_by_criterion': is_injective,
            'is_injective_by_blocks': cross_check_injective,
            'proposition_holds': is_injective == cross_check_injective
        }
        self.results.append(result)
        return result
    
    def compute_delta_T_sequence(self, rule: NonlinearRule, c: int, 
                                  T_max: int = 6) -> Dict:
        delta_sequence = []
        for T in range(1, T_max + 1):
            obs = ObservationBlockCode(rule, T, c)
            refiner = EquivalenceRefiner(rule, T, c)
            non_hidden = [i for i in obs.A_T if i not in obs.I_infty]
            n_free = len(non_hidden)
            total = rule.n_states ** n_free
            
            if total > 30000:
                configs = refiner.enumerate_configs(max_configs=5000)
                delta = refiner.compute_delta_max(configs)
            else:
                delta = refiner.compute_delta_max_full()
            
            delta_sequence.append(delta)
        
        result = {
            'rule_name': rule.name,
            'c': c,
            'T_values': list(range(1, T_max + 1)),
            'delta_sequence': delta_sequence,
            'delta_per_T': [d / T for d, T in zip(delta_sequence, range(1, T_max + 1))]
        }
        self.results.append(result)
        return result
    
    def verify_linear_vs_nonlinear(self, T: int, c: int) -> Dict:
        rule90 = create_rule_90()
        rule110 = create_rule_110()
        
        obs90 = ObservationBlockCode(rule90, T, c)
        obs110 = ObservationBlockCode(rule110, T, c)
        
        refiner90 = EquivalenceRefiner(rule90, T, c)
        refiner110 = EquivalenceRefiner(rule110, T, c)
        
        delta90 = refiner90.compute_delta_max_full()
        delta110 = refiner110.compute_delta_max_full()
        
        result = {
            'T': T, 'c': c,
            'rule90_delta_max': delta90,
            'rule110_delta_max': delta110,
            'rule90_I_infty': len(obs90.I_infty),
            'rule110_I_infty': len(obs110.I_infty),
            'rule90_complete_obs': len(obs90.I_infty) == 0 and delta90 == 0,
            'rule110_complete_obs': len(obs110.I_infty) == 0 and delta110 == 0,
            'nonlinear_has_extra_defect': delta110 > delta90
        }
        self.results.append(result)
        return result
    
    def run_all(self):
        print("=" * 70)
        print("实验B：观测等价类逐层细化与信息亏损计算")
        print("=" * 70)
        
        print("\n--- 测试1: 信息亏损δ_max计算 ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110"),
                         (create_rule_30, "Rule30"), (create_rule_184, "Rule184")]:
            for T, c in [(1, 1), (2, 1), (3, 1)]:
                r = self.compute_partition_and_defect(fn(), T, c)
                obs_status = "完全可观测" if r['complete_obs'] else f"缺陷δ={r['delta_max']:.2f}"
                print(f"  {name} T={T} c={c}: {obs_status}, |I_∞|={r['I_infty_size']}")
        
        print("\n--- 测试2: 命题4.1验证(划分刻画) ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110")]:
            for T, c in [(1, 1), (2, 1)]:
                r = self.verify_proposition_4_1(fn(), T, c)
                if 'status' in r and r['status'] == 'skipped_too_large':
                    print(f"  {name} T={T}: 跳过(状态空间过大)")
                else:
                    status = "✓" if r['proposition_holds'] else "✗"
                    print(f"  {name} T={T}: 单射={r['is_injective_by_criterion']} {status}")
        
        print("\n--- 测试3: 亏损序列δ_T(T) ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110"),
                         (create_rule_30, "Rule30")]:
            r = self.compute_delta_T_sequence(fn(), c=1, T_max=4)
            print(f"  {name}: δ_T = {r['delta_sequence']}")
            print(f"  {name}: δ_T/T = {[f'{x:.2f}' for x in r['delta_per_T']]}")
        
        print("\n--- 测试4: 线性vs非线性对比 ---")
        for T, c in [(1, 1), (2, 1), (3, 1)]:
            r = self.verify_linear_vs_nonlinear(T, c)
            print(f"  T={T}: Rule90 δ={r['rule90_delta_max']:.2f}, "
                  f"Rule110 δ={r['rule110_delta_max']:.2f}, "
                  f"非线性额外缺陷={r['nonlinear_has_extra_defect']}")
        
        self.save_results()
    
    def save_results(self):
        with open("experimentB_results.txt", 'w', encoding='utf-8') as f:
            f.write("实验B：观测等价类逐层细化与信息亏损计算结果\n")
            f.write("=" * 70 + "\n\n")
            for r in self.results:
                f.write(f"{r}\n\n")


if __name__ == "__main__":
    np.random.seed(42)
    exp = ExperimentB()
    exp.run_all()
