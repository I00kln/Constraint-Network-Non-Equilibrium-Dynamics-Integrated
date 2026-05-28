"""
认知与规律涌现框架 - 主验证脚本
整合所有验证模块，运行完整的验证流程
"""
import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discrete_closure_dynamics import verify_theorem1_discrete_to_continuous
from hjb_eikonal_solver import verify_theorem2_hjb_equation, verify_theorem3_eikonal_equation
from finsler_emergence import verify_theorem4_finsler_emergence
from weak_kam_structure import verify_theorem5_weak_kam
from cognitive_transition import verify_theorem6_cognitive_transition
from visualization import CognitiveVisualization, create_summary_report


class CognitiveEmergenceVerification:
    """认知涌现框架验证系统"""
    
    def __init__(self, output_dir: str = "./verification_results"):
        self.output_dir = output_dir
        self.results = {}
        self.timing = {}
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.viz = CognitiveVisualization()
    
    def run_all_verifications(self, 
                             enable_visualization: bool = True,
                             save_results: bool = True) -> Dict:
        """运行所有验证"""
        print("\n" + "=" * 80)
        print("认知与规律涌现框架 - 完整验证流程")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("验证定理1：离散到连续的桥接定理（Γ-收敛）")
        print("=" * 80)
        t1_start = time.time()
        try:
            theorem1_results = verify_theorem1_discrete_to_continuous()
            self.results['theorem1'] = theorem1_results.get('gamma_convergence', False)
            self.results['theorem1_details'] = theorem1_results
            print("✓ 定理1验证完成")
        except Exception as e:
            print(f"✗ 定理1验证失败: {e}")
            self.results['theorem1'] = False
            self.results['theorem1_details'] = {'error': str(e)}
        self.timing['theorem1'] = time.time() - t1_start
        
        print("\n" + "=" * 80)
        print("验证定理2：认知最优控制定理（动态HJB方程）")
        print("=" * 80)
        t2_start = time.time()
        try:
            theorem2_results = verify_theorem2_hjb_equation()
            self.results['theorem2'] = theorem2_results.get('converged', False)
            self.results['theorem2_details'] = theorem2_results
            print("✓ 定理2验证完成")
        except Exception as e:
            print(f"✗ 定理2验证失败: {e}")
            self.results['theorem2'] = False
            self.results['theorem2_details'] = {'error': str(e)}
        self.timing['theorem2'] = time.time() - t2_start
        
        print("\n" + "=" * 80)
        print("验证定理3：认知静态传播定理（静态Eikonal方程）")
        print("=" * 80)
        t3_start = time.time()
        try:
            theorem3_results = verify_theorem3_eikonal_equation()
            self.results['theorem3'] = theorem3_results.get('converged', False)
            self.results['theorem3_details'] = theorem3_results
            print("✓ 定理3验证完成")
        except Exception as e:
            print(f"✗ 定理3验证失败: {e}")
            self.results['theorem3'] = False
            self.results['theorem3_details'] = {'error': str(e)}
        self.timing['theorem3'] = time.time() - t3_start
        
        print("\n" + "=" * 80)
        print("验证定理4：认知几何涌现定理（Finsler度量）")
        print("=" * 80)
        t4_start = time.time()
        try:
            theorem4_results = verify_theorem4_finsler_emergence()
            self.results['theorem4'] = theorem4_results.get('converged', False)
            self.results['theorem4_details'] = theorem4_results
            print("✓ 定理4验证完成")
        except Exception as e:
            print(f"✗ 定理4验证失败: {e}")
            self.results['theorem4'] = False
            self.results['theorem4_details'] = {'error': str(e)}
        self.timing['theorem4'] = time.time() - t4_start
        
        print("\n" + "=" * 80)
        print("验证定理5：认知相变与分类定理（弱KAM结构）")
        print("=" * 80)
        t5_start = time.time()
        try:
            theorem5_results = verify_theorem5_weak_kam()
            self.results['theorem5'] = theorem5_results.get('converged', False)
            self.results['theorem5_details'] = theorem5_results
            print("✓ 定理5验证完成")
        except Exception as e:
            print(f"✗ 定理5验证失败: {e}")
            self.results['theorem5'] = False
            self.results['theorem5_details'] = {'error': str(e)}
        self.timing['theorem5'] = time.time() - t5_start
        
        print("\n" + "=" * 80)
        print("验证定理6：认知跃迁定理（大偏差理论）")
        print("=" * 80)
        t6_start = time.time()
        try:
            theorem6_results = verify_theorem6_cognitive_transition()
            self.results['theorem6'] = theorem6_results.get('converged', False)
            self.results['theorem6_details'] = theorem6_results
            print("✓ 定理6验证完成")
        except Exception as e:
            print(f"✗ 定理6验证失败: {e}")
            self.results['theorem6'] = False
            self.results['theorem6_details'] = {'error': str(e)}
        self.timing['theorem6'] = time.time() - t6_start
        
        total_time = time.time() - start_time
        
        if enable_visualization:
            print("\n" + "=" * 80)
            print("生成可视化结果...")
            print("=" * 80)
            self._generate_visualizations()
        
        if save_results:
            self._save_results()
        
        print("\n" + "=" * 80)
        print("验证结果汇总")
        print("=" * 80)
        self._print_summary()
        
        print(f"\n总验证时间: {total_time:.2f} 秒")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results
    
    def _generate_visualizations(self):
        """生成可视化结果"""
        try:
            if 'theorem6_details' in self.results:
                details = self.results['theorem6_details']
                if 'local_minima' in details and len(details['local_minima']) > 0:
                    def potential(x):
                        return (x[0]**4 - 2 * x[0]**2 + x[1]**2 + 
                                0.5 * x[0]**2 * x[1]**2)
                    
                    fig = self.viz.plot_potential_landscape(
                        potential,
                        title="认知势景观",
                        minima=details['local_minima'],
                        saddles=details.get('saddle_points', None)
                    )
                    fig.savefig(os.path.join(self.output_dir, 'potential_landscape.png'),
                               dpi=150, bbox_inches='tight')
                    print("  ✓ 势景观可视化已保存")
            
            if 'theorem4_details' in self.results:
                details = self.results['theorem4_details']
                if 'unit_ball' in details and len(details['unit_ball']) > 0:
                    fig = self.viz.plot_finsler_unit_ball(
                        details['unit_ball'],
                        title="Finsler度量单位球"
                    )
                    fig.savefig(os.path.join(self.output_dir, 'finsler_unit_ball.png'),
                               dpi=150, bbox_inches='tight')
                    print("  ✓ Finsler单位球可视化已保存")
            
            if 'theorem6_details' in self.results:
                details = self.results['theorem6_details']
                if 'transition_probabilities' in details and len(details['transition_probabilities']) > 0:
                    epsilon_values = [0.5, 0.3, 0.2, 0.1, 0.05]
                    fig = self.viz.plot_transition_dynamics(
                        epsilon_values[:len(details['transition_probabilities'])],
                        details['transition_probabilities'],
                        title="认知跃迁动力学"
                    )
                    fig.savefig(os.path.join(self.output_dir, 'transition_dynamics.png'),
                               dpi=150, bbox_inches='tight')
                    print("  ✓ 跃迁动力学可视化已保存")
            
            if 'theorem5_details' in self.results:
                details = self.results['theorem5_details']
                if 'aubry_set' in details and len(details['aubry_set']) > 0:
                    fig = self.viz.plot_aubry_set(
                        details['aubry_set'],
                        details.get('barrier_matrix', None),
                        title="Aubry集与Peierls势垒"
                    )
                    fig.savefig(os.path.join(self.output_dir, 'aubry_set.png'),
                               dpi=150, bbox_inches='tight')
                    print("  ✓ Aubry集可视化已保存")
            
            summary_fig = self.viz.plot_verification_summary(self.results)
            summary_fig.savefig(os.path.join(self.output_dir, 'verification_summary.png'),
                               dpi=150, bbox_inches='tight')
            print("  ✓ 验证汇总可视化已保存")
            
        except Exception as e:
            print(f"  ✗ 可视化生成失败: {e}")
    
    def _save_results(self):
        """保存验证结果"""
        serializable_results = {}
        
        for key, value in self.results.items():
            if isinstance(value, bool):
                serializable_results[key] = value
            elif isinstance(value, dict):
                serializable_results[key] = self._make_serializable(value)
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = str(value)
        
        serializable_results['timing'] = self.timing
        
        results_file = os.path.join(self.output_dir, 
                                   f'verification_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {results_file}")
        
        report = create_summary_report(self.results)
        report_file = os.path.join(self.output_dir,
                                  f'verification_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"报告已保存到: {report_file}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _print_summary(self):
        """打印验证结果汇总"""
        theorems = [
            ("定理1", "离散到连续的桥接定理（Γ-收敛）"),
            ("定理2", "认知最优控制定理（动态HJB方程）"),
            ("定理3", "认知静态传播定理（静态Eikonal方程）"),
            ("定理4", "认知几何涌现定理（Finsler度量）"),
            ("定理5", "认知相变与分类定理（弱KAM结构）"),
            ("定理6", "认知跃迁定理（大偏差理论）")
        ]
        
        print("\n定理验证状态:")
        print("-" * 80)
        
        for i, (name, desc) in enumerate(theorems, 1):
            status = self.results.get(f'theorem{i}', False)
            status_str = "✓ 通过" if status else "✗ 未通过"
            time_str = f"{self.timing.get(f'theorem{i}', 0):.2f}s"
            print(f"{name}: {status_str} ({time_str})")
            print(f"  {desc}")
        
        print("-" * 80)
        
        passed_count = sum(1 for i in range(1, 7) if self.results.get(f'theorem{i}', False))
        total_count = 6
        
        print(f"\n总体结果: {passed_count}/{total_count} 定理通过验证")
        
        if passed_count == total_count:
            print("\n🎉 恭喜！所有核心定理均通过数值验证！")
            print("认知与规律涌现框架的数学结构在数值层面得到了确认。")
        elif passed_count >= total_count * 0.7:
            print(f"\n✓ 大部分定理通过验证（{passed_count}/{total_count}）。")
            print("框架的核心结构基本得到确认。")
        else:
            print(f"\n⚠ 部分定理验证未通过（{passed_count}/{total_count}）。")
            print("需要进一步调整参数或改进算法。")


def run_quick_verification():
    """运行快速验证（简化版）"""
    print("\n" + "=" * 80)
    print("认知与规律涌现框架 - 快速验证模式")
    print("=" * 80)
    
    results = {}
    
    print("\n验证定理1（Γ-收敛）...")
    try:
        t1_results = verify_theorem1_discrete_to_continuous()
        results['theorem1'] = t1_results.get('gamma_convergence', False)
        print(f"  结果: {'通过' if results['theorem1'] else '未通过'}")
    except Exception as e:
        results['theorem1'] = False
        print(f"  错误: {e}")
    
    print("\n验证定理2（HJB方程）...")
    try:
        t2_results = verify_theorem2_hjb_equation()
        results['theorem2'] = t2_results.get('converged', False)
        print(f"  结果: {'通过' if results['theorem2'] else '未通过'}")
    except Exception as e:
        results['theorem2'] = False
        print(f"  错误: {e}")
    
    print("\n验证定理4（Finsler度量）...")
    try:
        t4_results = verify_theorem4_finsler_emergence()
        results['theorem4'] = t4_results.get('converged', False)
        print(f"  结果: {'通过' if results['theorem4'] else '未通过'}")
    except Exception as e:
        results['theorem4'] = False
        print(f"  错误: {e}")
    
    print("\n验证定理6（认知跃迁）...")
    try:
        t6_results = verify_theorem6_cognitive_transition()
        results['theorem6'] = t6_results.get('converged', False)
        print(f"  结果: {'通过' if results['theorem6'] else '未通过'}")
    except Exception as e:
        results['theorem6'] = False
        print(f"  错误: {e}")
    
    passed = sum(1 for k, v in results.items() if v)
    print(f"\n快速验证结果: {passed}/{len(results)} 定理通过")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='认知与规律涌现框架验证系统')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'quick', 'single'],
                       help='验证模式: full=完整验证, quick=快速验证, single=单定理验证')
    parser.add_argument('--theorem', type=int, default=None,
                       help='单定理验证模式下要验证的定理编号(1-6)')
    parser.add_argument('--no-viz', action='store_true',
                       help='禁用可视化')
    parser.add_argument('--output', type=str, default='./verification_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_verification()
    
    elif args.mode == 'single':
        if args.theorem is None or args.theorem < 1 or args.theorem > 6:
            print("错误: 单定理验证模式需要指定定理编号(1-6)")
            return
        
        print(f"\n验证定理{args.theorem}...")
        
        verify_funcs = {
            1: verify_theorem1_discrete_to_continuous,
            2: verify_theorem2_hjb_equation,
            3: verify_theorem3_eikonal_equation,
            4: verify_theorem4_finsler_emergence,
            5: verify_theorem5_weak_kam,
            6: verify_theorem6_cognitive_transition
        }
        
        try:
            result = verify_funcs[args.theorem]()
            print(f"\n定理{args.theorem}验证完成")
        except Exception as e:
            print(f"\n定理{args.theorem}验证失败: {e}")
    
    else:
        verifier = CognitiveEmergenceVerification(output_dir=args.output)
        verifier.run_all_verifications(
            enable_visualization=not args.no_viz,
            save_results=True
        )


if __name__ == "__main__":
    main()
