"""
实验二：构造性核元素注入
验证几何核的存在性和完备性
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_structures import (
    CARule, ObservationMatrix, GeometricKernel,
    create_rule_90, create_rule_150, create_frobenius_resonance_model,
    create_general_rule
)


class Experiment2:
    """构造性核元素注入实验"""
    
    def __init__(self, output_file: str = "experiment2_results.txt"):
        self.output_file = output_file
        self.results = []
    
    def evolve_config(self, rule: CARule, config: np.ndarray, 
                      steps: int, window_center: int, window_size: int) -> List[np.ndarray]:
        observations = []
        current = config.copy()
        n = len(config)
        
        for t in range(steps + 1):
            window_start = window_center - window_size // 2
            window_end = window_start + window_size
            obs = current[window_start:window_end].copy()
            observations.append(obs)
            
            if t < steps:
                new_config = np.zeros_like(current)
                for i in range(n):
                    for j, coeff in rule.coeffs.items():
                        source_idx = i + j
                        if 0 <= source_idx < n:
                            new_config[i] = (new_config[i] + coeff * current[source_idx]) % rule.p
                current = new_config
        
        return observations
    
    def test_geometric_kernel_element(self, rule: CARule, T: int, c: int,
                                     i: int) -> Dict:
        n_T = 2 * (c + rule.r) * T + 1
        config = np.zeros(n_T, dtype=int)
        center_idx = n_T // 2
        actual_idx = center_idx + i
        
        if 0 <= actual_idx < n_T:
            config[actual_idx] = 1
        
        window_size = 2 * c * T + 1
        observations = self.evolve_config(rule, config, T, center_idx, window_size)
        
        all_zero = all(np.all(obs == 0) for obs in observations)
        
        return {
            'i': i,
            'all_zero': all_zero,
            'observations': observations
        }
    
    def test_geometric_kernel_completeness(self, rule: CARule, T: int, c: int,
                                          case_name: str) -> Dict:
        I_infinity = GeometricKernel.get_infinity_indices(rule, T, c)
        
        geo_elements_results = []
        for i in I_infinity:
            result = self.test_geometric_kernel_element(rule, T, c, i)
            geo_elements_results.append(result)
        
        all_geo_in_kernel = all(r['all_zero'] for r in geo_elements_results)
        
        num_random_tests = 100
        hidden_kernel_found = False
        hidden_kernel_examples = []
        
        A_T_min = -(c + rule.r) * T
        A_T_max = (c + rule.r) * T
        A_T = list(range(A_T_min, A_T_max + 1))
        
        reachable_indices = [i for i in A_T if i not in I_infinity]
        
        if len(reachable_indices) > 0:
            for _ in range(num_random_tests):
                config = np.zeros(2 * (c + rule.r) * T + 1, dtype=int)
                center_idx = len(config) // 2
                
                num_nonzero = np.random.randint(1, min(5, len(reachable_indices)) + 1)
                chosen_indices = np.random.choice(reachable_indices, 
                                                 size=num_nonzero, 
                                                 replace=False)
                
                for idx in chosen_indices:
                    actual_idx = center_idx + idx
                    if 0 <= actual_idx < len(config):
                        config[actual_idx] = np.random.randint(1, rule.p)
                
                window_size = 2 * c * T + 1
                observations = self.evolve_config(rule, config, T, center_idx, window_size)
                
                all_zero = all(np.all(obs == 0) for obs in observations)
                
                if all_zero:
                    hidden_kernel_found = True
                    hidden_kernel_examples.append({
                        'config': config.copy(),
                        'nonzero_indices': chosen_indices.tolist()
                    })
                    if len(hidden_kernel_examples) >= 3:
                        break
        
        result = {
            'case_name': case_name,
            'rule': str(rule),
            'T': T,
            'c': c,
            'p': rule.p,
            'r': rule.r,
            'd_minus': rule.d_minus,
            'd_plus': rule.d_plus,
            'geo_dim': len(I_infinity),
            'I_infinity_size': len(I_infinity),
            'all_geo_in_kernel': all_geo_in_kernel,
            'reachable_indices_count': len(reachable_indices),
            'num_random_tests': num_random_tests,
            'hidden_kernel_found': hidden_kernel_found,
            'hidden_kernel_count': len(hidden_kernel_examples),
            'theorem_holds': all_geo_in_kernel and not hidden_kernel_found
        }
        
        self.results.append(result)
        return result
    
    def run_comprehensive_tests(self):
        print("=" * 80)
        print("实验二：构造性核元素注入 - 验证几何核完备性")
        print("=" * 80)
        print("\n目标：")
        print("1. 验证几何核元素确实在核中")
        print("2. 验证不存在隐藏的代数核")
        print()
        
        test_cases = []
        
        print("测试组1: 几何核非零的情况（d_- < r 或 d_+ < r）")
        for T in range(1, 5):
            rule = create_general_rule(r=2, coeffs={-1: 1, 1: 1}, p=2)
            test_cases.append((rule, T, 2, f"d±=1_T={T}"))
        
        print("测试组2: 几何核为零但r>=2的情况")
        for T in range(1, 5):
            test_cases.append((create_frobenius_resonance_model(p=2), T, 2,
                             f"Frobenius_T={T}"))
        
        print("测试组3: r=1完全可观测情况")
        for T in range(1, 5):
            test_cases.append((create_rule_90(p=2), T, 1, f"规则90_T={T}"))
        
        print("\n开始计算...\n")
        
        for i, (rule, T, c, name) in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] 测试: {name}")
            result = self.test_geometric_kernel_completeness(rule, T, c, name)
            
            status1 = "✓" if result['all_geo_in_kernel'] else "✗"
            status2 = "✓ 未发现" if not result['hidden_kernel_found'] else "✗ 发现"
            
            print(f"  几何核元素在核中: {status1}")
            print(f"  隐藏代数核: {status2}")
            print(f"  几何核维数: {result['geo_dim']}")
        
        self.analyze_results()
        self.save_results()
    
    def analyze_results(self):
        print("\n" + "=" * 80)
        print("结果分析")
        print("=" * 80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['theorem_holds'])
        
        print(f"\n总测试数: {total}")
        print(f"定理成立: {passed}/{total} ({100*passed/total:.1f}%)")
        
        geo_in_kernel_all = sum(1 for r in self.results if r['all_geo_in_kernel'])
        print(f"\n几何核元素全在核中: {geo_in_kernel_all}/{total}")
        
        hidden_found_count = sum(1 for r in self.results if r['hidden_kernel_found'])
        print(f"发现隐藏代数核: {hidden_found_count}/{total}")
        
        if hidden_found_count > 0:
            print("\n发现隐藏代数核的案例:")
            for r in self.results:
                if r['hidden_kernel_found']:
                    print(f"  - {r['case_name']}: 发现{r['hidden_kernel_count']}个隐藏核元素")
        
        geo_nonzero_cases = [r for r in self.results if r['geo_dim'] > 0]
        if geo_nonzero_cases:
            print(f"\n几何核非零案例详情:")
            for r in geo_nonzero_cases:
                print(f"  - {r['case_name']}: dim(G_T)={r['geo_dim']}, "
                      f"几何核⊆核: {r['all_geo_in_kernel']}")
    
    def save_results(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("实验二：构造性核元素注入结果\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("测试配置:\n")
            f.write("- 目标1: 验证几何核元素确实在核中\n")
            f.write("- 目标2: 验证不存在隐藏的代数核（随机测试）\n")
            f.write("- 方法: 构造几何核元素并演化，随机生成可达向量测试\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 80 + "\n")
            
            for r in self.results:
                f.write(f"\n案例: {r['case_name']}\n")
                f.write(f"  规则: {r['rule']}\n")
                f.write(f"  参数: T={r['T']}, c={r['c']}, p={r['p']}, r={r['r']}\n")
                f.write(f"  极值步幅: d_-={r['d_minus']}, d_+={r['d_plus']}\n")
                f.write(f"  几何核维数: {r['geo_dim']}\n")
                f.write(f"  几何核元素在核中: {r['all_geo_in_kernel']}\n")
                f.write(f"  可达索引数: {r['reachable_indices_count']}\n")
                f.write(f"  随机测试数: {r['num_random_tests']}\n")
                f.write(f"  发现隐藏代数核: {r['hidden_kernel_found']}\n")
                f.write(f"  定理验证: {'通过' if r['theorem_holds'] else '失败'}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("总结:\n")
            total = len(self.results)
            passed = sum(1 for r in self.results if r['theorem_holds'])
            f.write(f"  总测试: {total}\n")
            f.write(f"  通过: {passed} ({100*passed/total:.1f}%)\n")
        
        print(f"\n结果已保存到: {self.output_file}")


if __name__ == "__main__":
    np.random.seed(42)
    exp2 = Experiment2()
    exp2.run_comprehensive_tests()
