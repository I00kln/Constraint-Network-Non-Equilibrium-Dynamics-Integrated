"""
实验四：Frobenius稀疏化的压力测试
验证Frobenius自同态是否会制造信息黑洞
"""
import numpy as np
from typing import List, Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_structures import (
    CARule, ObservationMatrix, GeometricKernel,
    create_rule_90, create_rule_150, create_general_rule
)


class Experiment4:
    """Frobenius稀疏化压力测试"""
    
    def __init__(self, output_file: str = "experiment4_results.txt"):
        self.output_file = output_file
        self.results = []
    
    def test_frobenius_case(self, rule: CARule, T: int, c: int, p: int,
                           case_name: str) -> Dict:
        obs = ObservationMatrix(rule, T, c)
        matrix = obs.construct_matrix()
        rank = obs.compute_rank()
        defect = obs.get_defect()
        
        geo_dim = GeometricKernel.compute_dimension(rule, T, c)
        
        is_full_rank = (defect == geo_dim)
        
        has_frobenius_risk = False
        if rule.r >= 2:
            for t in range(1, T + 1):
                if t >= p:
                    has_frobenius_risk = True
                    break
        
        result = {
            'case_name': case_name,
            'rule': str(rule),
            'T': T,
            'c': c,
            'p': p,
            'r': rule.r,
            'd_minus': rule.d_minus,
            'd_plus': rule.d_plus,
            'matrix_shape': matrix.shape,
            'rank': rank,
            'defect': defect,
            'geo_dim': geo_dim,
            'is_full_rank': is_full_rank,
            'has_frobenius_risk': has_frobenius_risk,
            'theorem_holds': is_full_rank
        }
        
        self.results.append(result)
        return result
    
    def run_comprehensive_tests(self):
        print("=" * 80)
        print("实验四：Frobenius稀疏化压力测试")
        print("=" * 80)
        print("\n目标：验证Frobenius自同态是否会制造信息黑洞")
        print("测试在不同特征p和时间T下的秩行为")
        print()
        
        test_cases = []
        
        print("测试组1: 特征p扫描（规则: P(z) = z^-1 + 1 + z）")
        for p in [2, 3, 5, 7, 11]:
            rule = create_rule_150(p=p)
            for T in [p, p+1, 2*p]:
                if T <= 10:
                    test_cases.append((rule, T, 1, p, f"规则150_p={p}_T={T}"))
        
        print("测试组2: 特征时间点测试")
        for p in [2, 3, 5]:
            rule = create_general_rule(r=2, coeffs={-2: 1, 0: 1, 2: 1}, p=p)
            for T in range(1, min(3*p, 10) + 1):
                test_cases.append((rule, T, 2, p, f"Frobenius_p={p}_T={T}"))
        
        print("测试组3: 大特征域测试")
        for p in [13, 17, 19]:
            rule = create_rule_90(p=p)
            test_cases.append((rule, 5, 1, p, f"规则90_p={p}_T=5"))
        
        print("测试组4: 多项式幂次稀疏化测试")
        for p in [2, 3]:
            rule = create_general_rule(r=2, coeffs={-1: 1, 1: 1}, p=p)
            for T in range(1, 8):
                test_cases.append((rule, T, 2, p, f"r=2_d±=1_p={p}_T={T}"))
        
        print("\n开始计算...\n")
        
        for i, (rule, T, c, p, name) in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] 测试: {name}")
            result = self.test_frobenius_case(rule, T, c, p, name)
            
            status = "✓ 满秩" if result['is_full_rank'] else "✗ 秩亏"
            risk = "有风险" if result['has_frobenius_risk'] else "无风险"
            
            print(f"  缺陷={result['defect']}, 几何核={result['geo_dim']}, {status}")
            print(f"  Frobenius风险: {risk}")
        
        self.analyze_results()
        self.save_results()
    
    def analyze_results(self):
        print("\n" + "=" * 80)
        print("结果分析")
        print("=" * 80)
        
        total = len(self.results)
        full_rank_count = sum(1 for r in self.results if r['is_full_rank'])
        
        print(f"\n总测试数: {total}")
        print(f"满秩（缺陷=几何核）: {full_rank_count}/{total} ({100*full_rank_count/total:.1f}%)")
        
        frobenius_risk_cases = [r for r in self.results if r['has_frobenius_risk']]
        if frobenius_risk_cases:
            print(f"\n存在Frobenius风险的案例: {len(frobenius_risk_cases)}个")
            full_rank_with_risk = sum(1 for r in frobenius_risk_cases if r['is_full_rank'])
            print(f"  其中满秩: {full_rank_with_risk}/{len(frobenius_risk_cases)}")
            
            if full_rank_with_risk == len(frobenius_risk_cases):
                print("  结论: Frobenius风险未导致实际秩亏！")
        
        print("\n按特征p分组统计:")
        p_groups = {}
        for r in self.results:
            p = r['p']
            if p not in p_groups:
                p_groups[p] = {'total': 0, 'full_rank': 0}
            p_groups[p]['total'] += 1
            if r['is_full_rank']:
                p_groups[p]['full_rank'] += 1
        
        for p in sorted(p_groups.keys()):
            stats = p_groups[p]
            print(f"  p={p}: {stats['full_rank']}/{stats['total']} 满秩")
        
        print("\n猜想验证:")
        print("  猜想：端部系数非零时，Frobenius不制造额外缺陷")
        all_passed = all(r['is_full_rank'] for r in self.results 
                        if r['d_minus'] == r['r'] and r['d_plus'] == r['r'])
        print(f"  验证结果: {'✓ 支持' if all_passed else '✗ 反例存在'}")
    
    def save_results(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("实验四：Frobenius稀疏化压力测试结果\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("测试配置:\n")
            f.write("- 目标: 验证Frobenius自同态是否制造信息黑洞\n")
            f.write("- 方法: 扫描不同特征p和时间T，检验秩行为\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 80 + "\n")
            
            for r in self.results:
                f.write(f"\n案例: {r['case_name']}\n")
                f.write(f"  规则: {r['rule']}\n")
                f.write(f"  参数: T={r['T']}, c={r['c']}, p={r['p']}, r={r['r']}\n")
                f.write(f"  极值步幅: d_-={r['d_minus']}, d_+={r['d_plus']}\n")
                f.write(f"  矩阵大小: {r['matrix_shape']}\n")
                f.write(f"  秩: {r['rank']}\n")
                f.write(f"  缺陷: {r['defect']}\n")
                f.write(f"  几何核维数: {r['geo_dim']}\n")
                f.write(f"  满秩: {r['is_full_rank']}\n")
                f.write(f"  Frobenius风险: {r['has_frobenius_risk']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("总结:\n")
            total = len(self.results)
            full_rank = sum(1 for r in self.results if r['is_full_rank'])
            f.write(f"  总测试: {total}\n")
            f.write(f"  满秩: {full_rank} ({100*full_rank/total:.1f}%)\n")
            
            frobenius_risk = sum(1 for r in self.results if r['has_frobenius_risk'])
            f.write(f"  Frobenius风险案例: {frobenius_risk}\n")
        
        print(f"\n结果已保存到: {self.output_file}")


if __name__ == "__main__":
    exp4 = Experiment4()
    exp4.run_comprehensive_tests()
