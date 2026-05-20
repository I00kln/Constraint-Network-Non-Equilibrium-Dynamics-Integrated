"""
实验三：极值满秩子矩阵的恢复能力测试
验证局部pivot证书的实用性
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_structures import (
    CARule, ObservationMatrix, GeometricKernel, FiniteFieldMatrix,
    create_rule_90, create_rule_150, create_frobenius_resonance_model,
    create_general_rule, compute_leading_term_info
)


class Experiment3:
    """极值满秩子矩阵恢复能力测试"""
    
    def __init__(self, output_file: str = "experiment3_results.txt"):
        self.output_file = output_file
        self.results = []
    
    def find_extreme_columns(self, rule: CARule, T: int, c: int) -> Tuple[List[int], List[Tuple[int, int]]]:
        A_T_min = -(c + rule.r) * T
        A_T_max = (c + rule.r) * T
        A_T = list(range(A_T_min, A_T_max + 1))
        
        extreme_cols = []
        leading_positions = []
        
        for i in A_T:
            tau, j_star, c_star = compute_leading_term_info(rule, T, c, i)
            
            if tau != float('inf') and tau <= T and c_star % rule.p != 0:
                extreme_cols.append(i)
                leading_positions.append((tau, j_star))
        
        unique_positions = list(set(leading_positions))
        
        selected_cols = []
        selected_positions = []
        
        for pos in sorted(unique_positions):
            candidates = [col for col, p in zip(extreme_cols, leading_positions) if p == pos]
            if candidates:
                selected_cols.append(candidates[0])
                selected_positions.append(pos)
        
        return selected_cols, selected_positions
    
    def extract_submatrix(self, rule: CARule, T: int, c: int,
                         col_indices: List[int], 
                         row_positions: List[Tuple[int, int]]) -> np.ndarray:
        obs = ObservationMatrix(rule, T, c)
        full_matrix = obs.construct_matrix()
        
        window_size = 2 * c * T + 1
        
        col_indices_global = [obs.A_T.index(i) for i in col_indices]
        
        row_indices_global = []
        for t, j in row_positions:
            row_idx = t * window_size + (j + c * T)
            row_indices_global.append(row_idx)
        
        submatrix = full_matrix[np.ix_(row_indices_global, col_indices_global)]
        
        return submatrix
    
    def test_recovery(self, rule: CARule, T: int, c: int,
                     case_name: str) -> Dict:
        extreme_cols, leading_positions = self.find_extreme_columns(rule, T, c)
        
        if len(extreme_cols) == 0:
            return {
                'case_name': case_name,
                'rule': str(rule),
                'T': T,
                'c': c,
                'submatrix_size': (0, 0),
                'submatrix_rank': 0,
                'is_invertible': False,
                'recovery_tests': 0,
                'recovery_success': 0,
                'theorem_holds': True
            }
        
        submatrix = self.extract_submatrix(rule, T, c, extreme_cols, leading_positions)
        submatrix_rank = FiniteFieldMatrix.rank(submatrix, rule.p)
        is_invertible = (submatrix.shape[0] == submatrix.shape[1] and 
                        submatrix_rank == submatrix.shape[0])
        
        num_recovery_tests = 20
        recovery_success = 0
        
        if is_invertible:
            obs = ObservationMatrix(rule, T, c)
            full_matrix = obs.construct_matrix()
            window_size = 2 * c * T + 1
            
            col_indices_global = [obs.A_T.index(i) for i in extreme_cols]
            row_indices_global = []
            for t, j in leading_positions:
                row_idx = t * window_size + (j + c * T)
                row_indices_global.append(row_idx)
            
            for _ in range(num_recovery_tests):
                x_sub = np.random.randint(0, rule.p, size=len(extreme_cols))
                
                x_full = np.zeros(obs.n_T, dtype=int)
                for idx, col_idx in enumerate(col_indices_global):
                    x_full[col_idx] = x_sub[idx]
                
                y_full = (full_matrix @ x_full) % rule.p
                
                y_sub = y_full[row_indices_global]
                
                x_recovered = FiniteFieldMatrix.solve(submatrix, y_sub, rule.p)
                
                if x_recovered is not None and np.array_equal(x_recovered % rule.p, x_sub):
                    recovery_success += 1
        
        result = {
            'case_name': case_name,
            'rule': str(rule),
            'T': T,
            'c': c,
            'p': rule.p,
            'r': rule.r,
            'submatrix_size': submatrix.shape,
            'submatrix_rank': submatrix_rank,
            'is_invertible': is_invertible,
            'num_extreme_cols': len(extreme_cols),
            'recovery_tests': num_recovery_tests,
            'recovery_success': recovery_success,
            'theorem_holds': is_invertible and recovery_success == num_recovery_tests
        }
        
        self.results.append(result)
        return result
    
    def run_comprehensive_tests(self):
        print("=" * 80)
        print("实验三：极值满秩子矩阵恢复能力测试")
        print("=" * 80)
        print("\n目标：验证局部pivot证书的实用性")
        print("1. 构造极值满秩子矩阵")
        print("2. 验证子矩阵可逆")
        print("3. 测试状态恢复能力")
        print()
        
        test_cases = []
        
        print("测试组1: r=1规则")
        for T in range(1, 5):
            test_cases.append((create_rule_90(p=2), T, 1, f"规则90_T={T}"))
            test_cases.append((create_rule_150(p=2), T, 1, f"规则150_T={T}"))
        
        print("测试组2: r=2规则")
        for T in range(1, 4):
            test_cases.append((create_frobenius_resonance_model(p=2), T, 2,
                             f"Frobenius_T={T}"))
        
        print("测试组3: 不同特征域")
        for p in [2, 3, 5]:
            rule = create_rule_90(p=p)
            test_cases.append((rule, 3, 1, f"规则90_p={p}"))
        
        print("\n开始计算...\n")
        
        for i, (rule, T, c, name) in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] 测试: {name}")
            result = self.test_recovery(rule, T, c, name)
            
            status = "✓ 可逆" if result['is_invertible'] else "✗ 不可逆"
            print(f"  子矩阵大小: {result['submatrix_size']}")
            print(f"  子矩阵秩: {result['submatrix_rank']}")
            print(f"  可逆性: {status}")
            if result['recovery_tests'] > 0:
                print(f"  恢复成功率: {result['recovery_success']}/{result['recovery_tests']}")
        
        self.analyze_results()
        self.save_results()
    
    def analyze_results(self):
        print("\n" + "=" * 80)
        print("结果分析")
        print("=" * 80)
        
        total = len(self.results)
        invertible_count = sum(1 for r in self.results if r['is_invertible'])
        
        print(f"\n总测试数: {total}")
        print(f"子矩阵可逆: {invertible_count}/{total} ({100*invertible_count/total:.1f}%)")
        
        recovery_tests = [r for r in self.results if r['recovery_tests'] > 0]
        if recovery_tests:
            perfect_recovery = sum(1 for r in recovery_tests 
                                  if r['recovery_success'] == r['recovery_tests'])
            print(f"完美恢复: {perfect_recovery}/{len(recovery_tests)}")
        
        print("\n子矩阵大小统计:")
        sizes = {}
        for r in self.results:
            size = r['submatrix_size']
            sizes[size] = sizes.get(size, 0) + 1
        for size, count in sorted(sizes.items()):
            print(f"  {size}: {count}个")
    
    def save_results(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("实验三：极值满秩子矩阵恢复能力测试结果\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("测试配置:\n")
            f.write("- 目标: 验证局部pivot证书的实用性\n")
            f.write("- 方法: 构造极值满秩子矩阵并测试状态恢复\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 80 + "\n")
            
            for r in self.results:
                f.write(f"\n案例: {r['case_name']}\n")
                f.write(f"  规则: {r['rule']}\n")
                f.write(f"  参数: T={r['T']}, c={r['c']}, p={r['p']}, r={r['r']}\n")
                f.write(f"  子矩阵大小: {r['submatrix_size']}\n")
                f.write(f"  子矩阵秩: {r['submatrix_rank']}\n")
                f.write(f"  可逆: {r['is_invertible']}\n")
                f.write(f"  极值列数: {r['num_extreme_cols']}\n")
                if r['recovery_tests'] > 0:
                    f.write(f"  恢复测试: {r['recovery_success']}/{r['recovery_tests']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("总结:\n")
            total = len(self.results)
            invertible = sum(1 for r in self.results if r['is_invertible'])
            f.write(f"  总测试: {total}\n")
            f.write(f"  可逆子矩阵: {invertible} ({100*invertible/total:.1f}%)\n")
        
        print(f"\n结果已保存到: {self.output_file}")


if __name__ == "__main__":
    np.random.seed(42)
    exp3 = Experiment3()
    exp3.run_comprehensive_tests()
