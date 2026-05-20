"""
实验一：总体秩检验
验证主定理：核 = 几何核，即缺陷 = dim(G_T)
"""
import numpy as np
from typing import List, Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_structures import (
    CARule, ObservationMatrix, GeometricKernel,
    create_rule_90, create_rule_150, create_frobenius_resonance_model,
    create_general_rule
)


class Experiment1:
    """总体秩检验实验"""
    
    def __init__(self, output_file: str = "experiment1_results.txt"):
        self.output_file = output_file
        self.results = []
    
    def test_single_case(self, rule: CARule, T: int, c: int, 
                        case_name: str) -> Dict:
        obs = ObservationMatrix(rule, T, c)
        matrix = obs.construct_matrix()
        rank = obs.compute_rank()
        defect = obs.get_defect()
        
        geo_dim = GeometricKernel.compute_dimension(rule, T, c)
        
        theorem_holds = (defect == geo_dim)
        
        result = {
            'case_name': case_name,
            'rule': str(rule),
            'T': T,
            'c': c,
            'p': rule.p,
            'r': rule.r,
            'd_minus': rule.d_minus,
            'd_plus': rule.d_plus,
            'matrix_shape': matrix.shape,
            'rank': rank,
            'defect': defect,
            'geo_dim': geo_dim,
            'theorem_holds': theorem_holds,
            'n_T': obs.n_T
        }
        
        self.results.append(result)
        return result
    
    def run_comprehensive_tests(self):
        print("=" * 80)
        print("实验一：总体秩检验 - 验证主定理")
        print("=" * 80)
        print("\n目标：验证缺陷 = 几何核维数")
        print()
        
        test_cases = []
        
        print("测试组1: r=1 规则（完全可观测）")
        for T in range(1, 7):
            for c in [1, 2]:
                test_cases.append((create_rule_90(p=2), T, c, f"规则90_T={T}_c={c}"))
                test_cases.append((create_rule_150(p=2), T, c, f"规则150_T={T}_c={c}"))
        
        print("测试组2: r=2 Frobenius共振模型")
        for T in range(1, 6):
            for c in [2, 3]:
                test_cases.append((create_frobenius_resonance_model(p=2), T, c, 
                                 f"Frobenius_T={T}_c={c}"))
        
        print("测试组3: 不同特征域")
        for p in [2, 3, 5, 7]:
            rule = create_rule_90(p=p)
            test_cases.append((rule, 3, 2, f"规则90_p={p}"))
        
        print("测试组4: 几何核非零的情况")
        for T in range(1, 5):
            rule = create_general_rule(r=2, coeffs={-1: 1, 1: 1}, p=2)
            test_cases.append((rule, T, 2, f"r=2_d±=1_T={T}"))
        
        print("\n开始计算...\n")
        
        for i, (rule, T, c, name) in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] 测试: {name}")
            result = self.test_single_case(rule, T, c, name)
            
            status = "✓ 通过" if result['theorem_holds'] else "✗ 失败"
            print(f"  缺陷={result['defect']}, 几何核={result['geo_dim']}, {status}")
        
        self.analyze_results()
        self.save_results()
    
    def analyze_results(self):
        print("\n" + "=" * 80)
        print("结果分析")
        print("=" * 80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['theorem_holds'])
        failed = total - passed
        
        print(f"\n总测试数: {total}")
        print(f"通过: {passed} ({100*passed/total:.1f}%)")
        print(f"失败: {failed}")
        
        if failed > 0:
            print("\n失败案例详情:")
            for r in self.results:
                if not r['theorem_holds']:
                    print(f"  - {r['case_name']}: 缺陷={r['defect']}, 几何核={r['geo_dim']}")
        
        r1_cases = [r for r in self.results if r['r'] == 1 and r['d_minus'] == 1 and r['d_plus'] == 1]
        if r1_cases:
            print(f"\nr=1且端部非零的案例: {len(r1_cases)}个")
            r1_passed = sum(1 for r in r1_cases if r['defect'] == 0)
            print(f"  完全可观测(缺陷=0): {r1_passed}/{len(r1_cases)}")
        
        geo_nonzero = [r for r in self.results if r['geo_dim'] > 0]
        if geo_nonzero:
            print(f"\n几何核非零的案例: {len(geo_nonzero)}个")
            for r in geo_nonzero[:5]:
                print(f"  - {r['case_name']}: 几何核={r['geo_dim']}, 缺陷={r['defect']}")
    
    def save_results(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("实验一：总体秩检验结果\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("测试配置:\n")
            f.write("- 目标: 验证缺陷 = 几何核维数\n")
            f.write("- 方法: 直接计算观测矩阵秩并与理论值比较\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 80 + "\n")
            
            for r in self.results:
                f.write(f"\n案例: {r['case_name']}\n")
                f.write(f"  规则: {r['rule']}\n")
                f.write(f"  参数: T={r['T']}, c={r['c']}, p={r['p']}, r={r['r']}\n")
                f.write(f"  极值步幅: d_-={r['d_minus']}, d_+={r['d_plus']}\n")
                f.write(f"  矩阵大小: {r['matrix_shape']}\n")
                f.write(f"  秩: {r['rank']}/{r['n_T']}\n")
                f.write(f"  缺陷: {r['defect']}\n")
                f.write(f"  几何核维数: {r['geo_dim']}\n")
                f.write(f"  定理验证: {'通过' if r['theorem_holds'] else '失败'}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("总结:\n")
            total = len(self.results)
            passed = sum(1 for r in self.results if r['theorem_holds'])
            f.write(f"  总测试: {total}\n")
            f.write(f"  通过: {passed} ({100*passed/total:.1f}%)\n")
            f.write(f"  失败: {total - passed}\n")
        
        print(f"\n结果已保存到: {self.output_file}")


if __name__ == "__main__":
    exp1 = Experiment1()
    exp1.run_comprehensive_tests()
