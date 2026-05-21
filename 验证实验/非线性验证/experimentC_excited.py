"""
实验C：激发态、最小触发簇与因果碰撞验证
验证：定义5.1(可见激发态/最小触发簇)、定义5.2(首次可见信息)、
      定义6.3(因果碰撞)、命题6.4(无碰撞独立演化)
"""
import numpy as np
import sys, os, math
from itertools import product
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nonlinear_core import *


class ExperimentC:
    def __init__(self):
        self.results = []
    
    def test_single_cell_excited_states(self, rule: NonlinearRule, T: int, c: int) -> Dict:
        obs = ObservationBlockCode(rule, T, c)
        analyzer = ExcitedStateAnalyzer(rule, T, c)
        center = obs.A_T_half
        
        visible_cells = []
        invisible_cells = []
        first_visible_info = {}
        
        for i in obs.A_T:
            if i in obs.I_infty:
                invisible_cells.append(i)
                continue
            
            for s in rule.S:
                if s == rule.s0:
                    continue
                sigma = {i: s}
                if analyzer.is_visible([i], sigma):
                    if i not in [v[0] for v in visible_cells]:
                        visible_cells.append((i, s))
                        info = analyzer.compute_first_visible_info([i], sigma)
                        if info is not None:
                            first_visible_info[(i, s)] = info
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'visible_count': len(visible_cells),
            'invisible_count': len(invisible_cells),
            'first_visible_info': {str(k): v for k, v in first_visible_info.items()},
            'I_infty_size': len(obs.I_infty)
        }
        self.results.append(result)
        return result
    
    def test_minimal_trigger_clusters(self, rule: NonlinearRule, T: int, c: int) -> Dict:
        analyzer = ExcitedStateAnalyzer(rule, T, c)
        
        clusters = analyzer.find_minimal_trigger_clusters(max_search=50)
        
        cluster_info = []
        for A, sigma in clusters:
            info = analyzer.compute_first_visible_info(A, sigma)
            cluster_info.append({
                'support': A,
                'sigma': sigma,
                'first_visible': info
            })
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'num_clusters': len(clusters),
            'clusters': cluster_info
        }
        self.results.append(result)
        return result
    
    def test_causal_collision(self, rule: NonlinearRule, T: int, c: int) -> Dict:
        analyzer = ExcitedStateAnalyzer(rule, T, c)
        obs = ObservationBlockCode(rule, T, c)
        center = obs.A_T_half
        
        single_excited = []
        for i in obs.A_T[:5]:
            if i in obs.I_infty:
                continue
            for s in rule.S:
                if s == rule.s0:
                    continue
                sigma = {i: s}
                if analyzer.is_visible([i], sigma):
                    single_excited.append(([i], sigma))
        
        collisions_found = 0
        no_collision_pairs = 0
        collision_details = []
        
        for i in range(len(single_excited)):
            for j in range(i + 1, len(single_excited)):
                e1 = single_excited[i]
                e2 = single_excited[j]
                
                has_collision = analyzer.check_causal_collision(e1, e2)
                if has_collision:
                    collisions_found += 1
                    collision_details.append((e1, e2))
                else:
                    no_collision_pairs += 1
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'num_excited_states': len(single_excited),
            'collision_pairs': collisions_found,
            'no_collision_pairs': no_collision_pairs,
            'collision_details': [(str(e1), str(e2)) for e1, e2 in collision_details[:5]]
        }
        self.results.append(result)
        return result
    
    def test_independent_evolution_proposition_6_4(self, rule: NonlinearRule, 
                                                    T: int, c: int,
                                                    num_tests: int = 20) -> Dict:
        analyzer = ExcitedStateAnalyzer(rule, T, c)
        obs = ObservationBlockCode(rule, T, c)
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
        
        violations = 0
        tests_run = 0
        
        for i in range(len(single_excited)):
            for j in range(i + 1, len(single_excited)):
                A1, sigma1, info1 = single_excited[i]
                A2, sigma2, info2 = single_excited[j]
                
                has_collision = analyzer.check_causal_collision(
                    (A1, sigma1), (A2, sigma2))
                has_collision_rev = analyzer.check_causal_collision(
                    (A2, sigma2), (A1, sigma1))
                
                if has_collision or has_collision_rev:
                    continue
                
                config_joint = np.full(obs.n_T, rule.s0, dtype=int)
                for pos, val in sigma1.items():
                    idx = center + pos
                    if 0 <= idx < obs.n_T:
                        config_joint[idx] = val
                for pos, val in sigma2.items():
                    idx = center + pos
                    if 0 <= idx < obs.n_T:
                        config_joint[idx] = val
                
                obs_joint = obs.compute_observation_block(config_joint)
                
                tau1, J1, phi1 = info1
                center_obs = obs.A_T_half
                left = center_obs - c * T
                right = center_obs + c * T + 1
                
                traj_joint = evolver.evolve_steps(config_joint, T)
                if tau1 <= T:
                    window_at_tau = traj_joint[tau1][left:right]
                    for j_pos in J1:
                        j_idx = j_pos + c * T
                        if 0 <= j_idx < len(window_at_tau):
                            if j_pos in phi1:
                                if window_at_tau[j_idx] != phi1[j_pos]:
                                    violations += 1
                tests_run += 1
                
                if tests_run >= num_tests:
                    break
            if tests_run >= num_tests:
                break
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'tests_run': tests_run,
            'violations': violations,
            'proposition_holds': violations == 0
        }
        self.results.append(result)
        return result
    
    def test_decodable_support(self, rule: NonlinearRule, T: int, c: int) -> Dict:
        analyzer = ExcitedStateAnalyzer(rule, T, c)
        obs = ObservationBlockCode(rule, T, c)
        
        decodable = []
        non_decodable = []
        
        for i in obs.A_T[:5]:
            if i in obs.I_infty:
                continue
            is_dec = analyzer.is_decodable_support([i])
            if is_dec:
                decodable.append(i)
            else:
                non_decodable.append(i)
        
        result = {
            'rule_name': rule.name,
            'T': T, 'c': c,
            'decodable_cells': decodable,
            'non_decodable_cells': non_decodable
        }
        self.results.append(result)
        return result
    
    def run_all(self):
        print("=" * 70)
        print("实验C：激发态、最小触发簇与因果碰撞验证")
        print("=" * 70)
        
        print("\n--- 测试1: 单细胞激发态可见性 ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110"),
                         (create_rule_30, "Rule30")]:
            r = self.test_single_cell_excited_states(fn(), T=2, c=1)
            print(f"  {name}: 可见={r['visible_count']}, 不可见={r['invisible_count']}")
        
        print("\n--- 测试2: 最小触发簇 ---")
        for fn, name in [(create_rule_110, "Rule110"), (create_rule_30, "Rule30")]:
            r = self.test_minimal_trigger_clusters(fn(), T=2, c=1)
            print(f"  {name}: 最小触发簇数={r['num_clusters']}")
            for ci in r['clusters'][:3]:
                print(f"    支撑={ci['support']}, 首次可见={ci['first_visible']}")
        
        print("\n--- 测试3: 因果碰撞检测 ---")
        for fn, name in [(create_rule_110, "Rule110"), (create_rule_30, "Rule30")]:
            r = self.test_causal_collision(fn(), T=2, c=1)
            print(f"  {name}: 碰撞对={r['collision_pairs']}, "
                  f"无碰撞对={r['no_collision_pairs']}")
        
        print("\n--- 测试4: 无碰撞独立演化(命题6.4) ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110")]:
            r = self.test_independent_evolution_proposition_6_4(fn(), T=2, c=1)
            status = "✓" if r['proposition_holds'] else "✗"
            print(f"  {name}: 测试={r['tests_run']}, 违反={r['violations']} {status}")
        
        print("\n--- 测试5: 可解码支撑 ---")
        for fn, name in [(create_rule_90, "Rule90"), (create_rule_110, "Rule110")]:
            r = self.test_decodable_support(fn(), T=2, c=1)
            print(f"  {name}: 可解码={r['decodable_cells']}, "
                  f"不可解码={r['non_decodable_cells']}")
        
        self.save_results()
    
    def save_results(self):
        with open("experimentC_results.txt", 'w', encoding='utf-8') as f:
            f.write("实验C：激发态、最小触发簇与因果碰撞验证结果\n")
            f.write("=" * 70 + "\n\n")
            for r in self.results:
                f.write(f"{r}\n\n")


if __name__ == "__main__":
    np.random.seed(42)
    exp = ExperimentC()
    exp.run_all()
