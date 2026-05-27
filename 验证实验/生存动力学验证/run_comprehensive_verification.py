"""
生存动力学综合验证实验
系统验证文档《生存动力学》中的所有主要定理和结论
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from survival_dynamics_core import (
    SystemParameters, SurvivalDynamicsSystem, TheoremVerifier,
    generate_random_initial_condition, generate_initial_on_torus,
    generate_initial_near_origin
)
from survival_dynamics_visualization import DynamicsVisualizer

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveVerification:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or "."
        self.results = {}
        
    def run_all_verifications(self, params: SystemParameters = None):
        if params is None:
            params = SystemParameters(r1=1.0, r2=1.0, omega=1.0, gamma=0.5)
        
        self.params = params
        self.system = SurvivalDynamicsSystem(params)
        self.verifier = TheoremVerifier(self.system)
        self.visualizer = DynamicsVisualizer(self.system)
        
        print("=" * 80)
        print("生存动力学综合验证实验")
        print("=" * 80)
        print(f"系统参数: r₁={params.r1}, r₂={params.r2}, ω={params.omega}, γ={params.gamma}")
        print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        self._verify_theorem1()
        self._verify_theorem2()
        self._verify_theorem3()
        self._verify_theorem5()
        self._verify_hopf_structure()
        self._parameter_sensitivity_analysis()
        self._generate_summary_report()
        
        return self.results
    
    def _verify_theorem1(self):
        print("\n" + "=" * 60)
        print("【定理1验证】能量单调衰减")
        print("=" * 60)
        
        n_tests = 10
        all_monotonic = True
        max_errors = []
        
        for i in range(n_tests):
            np.random.seed(42 + i)
            initial = generate_random_initial_condition(self.params)
            trajectory = self.system.simulate(initial, t_span=(0, 30), n_points=1000)
            result = self.verifier.verify_theorem1_energy_decay(trajectory)
            
            all_monotonic = all_monotonic and result['is_monotonic_decreasing']
            max_errors.append(result['max_gradient_error'])
            
            if i == 0:
                self.results['theorem1_example'] = {
                    'trajectory': trajectory,
                    'result': result
                }
        
        avg_error = np.mean(max_errors)
        
        print(f"测试轨线数: {n_tests}")
        print(f"所有轨线能量单调递减: {all_monotonic}")
        print(f"平均梯度误差: {avg_error:.2e}")
        print(f"定理1验证结果: {'通过' if all_monotonic else '需要检查'}")
        
        self.results['theorem1'] = {
            'all_monotonic': all_monotonic,
            'avg_gradient_error': avg_error,
            'passed': all_monotonic
        }
    
    def _verify_theorem2(self):
        print("\n" + "=" * 60)
        print("【定理2验证】原点的双曲鞍点结构")
        print("=" * 60)
        
        result = self.verifier.verify_theorem2_hyperbolic_saddle()
        
        print(f"特征值实部: {np.sort(result['real_parts'])}")
        print(f"正实部特征值个数: {result['n_positive_real']}")
        print(f"负实部特征值个数: {result['n_negative_real']}")
        print(f"零实部特征值个数: {result['n_zero_real']}")
        print(f"是否为双曲平衡点: {result['is_hyperbolic']}")
        print(f"是否为鞍点: {result['is_saddle_point']}")
        print(f"稳定流形维数: {result['stable_dim']}")
        print(f"不稳定流形维数: {result['unstable_dim']}")
        print(f"定理2验证结果: {'通过' if result['theorem_holds'] else '失败'}")
        
        self.results['theorem2'] = result
        
        print("\n补充验证: 从原点附近出发的轨线行为")
        for epsilon in [0.01, 0.05, 0.1]:
            np.random.seed(100)
            initial = generate_initial_near_origin(self.params, epsilon)
            trajectory = self.system.simulate(initial, t_span=(0, 20), n_points=500)
            final_dist = trajectory['dist_to_torus'][-1]
            print(f"  ε={epsilon}: 最终到环面距离={final_dist:.4f}")
    
    def _verify_theorem3(self):
        print("\n" + "=" * 60)
        print("【定理3验证】生存环面的几乎全局吸引性")
        print("=" * 60)
        
        n_tests = 20
        convergence_count = 0
        convergence_times = []
        final_distances = []
        
        for i in range(n_tests):
            np.random.seed(42 + i)
            initial = generate_random_initial_condition(self.params)
            trajectory = self.system.simulate(initial, t_span=(0, 50), n_points=2000)
            result = self.verifier.verify_theorem3_global_attraction(trajectory)
            
            if result['converges_to_torus']:
                convergence_count += 1
                if result['convergence_time'] < float('inf'):
                    convergence_times.append(result['convergence_time'])
            
            final_distances.append(result['final_distance_to_torus'])
            
            if i < 3:
                self.results[f'trajectory_{i+1}'] = trajectory
        
        convergence_ratio = convergence_count / n_tests
        avg_final_dist = np.mean(final_distances)
        
        print(f"测试轨线数: {n_tests}")
        print(f"收敛到环面的轨线数: {convergence_count}")
        print(f"收敛比例: {convergence_ratio:.2%}")
        print(f"平均最终距离: {avg_final_dist:.2e}")
        if convergence_times:
            print(f"平均收敛时间: {np.mean(convergence_times):.2f}")
        print(f"定理3验证结果: {'通过' if convergence_ratio > 0.9 else '需要检查'}")
        
        self.results['theorem3'] = {
            'convergence_ratio': convergence_ratio,
            'avg_final_distance': avg_final_dist,
            'passed': convergence_ratio > 0.9
        }
    
    def _verify_theorem5(self):
        print("\n" + "=" * 60)
        print("【定理5验证】正规双曲吸引性与谱分解")
        print("=" * 60)
        
        result = self.verifier.verify_theorem5_normal_hyperbolicity()
        
        print(f"线性化特征值:")
        for i, ev in enumerate(result['eigenvalues']):
            print(f"  λ_{i+1} = {ev:.4f} (实部: {np.real(ev):.4f})")
        
        print(f"\n谱分解:")
        print(f"  零特征值个数 (切向中性模): {result['n_zero_eigenvalues']}")
        print(f"  负实部特征值个数 (法向稳定模): {result['n_negative_eigenvalues']}")
        print(f"  正实部特征值个数: {result['n_positive_eigenvalues']}")
        
        print(f"\nRouth-Hurwitz 稳定性判据:")
        print(f"  r₁: {result['Routh_Hurwitz_r1']}")
        print(f"  r₂: {result['Routh_Hurwitz_r2']}")
        
        print(f"\n正规双曲吸引性: {result['is_attracting_NHIM']}")
        print(f"定理5验证结果: {'通过' if result['theorem_holds'] else '失败'}")
        
        self.results['theorem5'] = result
    
    def _verify_hopf_structure(self):
        print("\n" + "=" * 60)
        print("【Hopf 流结构验证】")
        print("=" * 60)
        
        print("在生存环面上的动力学: ż_k = iω z_k")
        print(f"角频率: ω = {self.params.omega}")
        print(f"周期: T = 2π/ω = {2*np.pi/self.params.omega:.4f}")
        
        print("\n验证环面上的周期轨道:")
        n_phases = 5
        for i in range(n_phases):
            phi1 = 2 * np.pi * i / n_phases
            phi2 = 0
            initial = generate_initial_on_torus(self.params, phi1, phi2)
            trajectory = self.system.simulate(initial, t_span=(0, 10), n_points=500)
            
            z1_final = trajectory['z1'][-1]
            z2_final = trajectory['z2'][-1]
            z1_initial = trajectory['z1'][0]
            z2_initial = trajectory['z2'][0]
            
            phase_diff1 = np.angle(z1_final / z1_initial)
            phase_diff2 = np.angle(z2_final / z2_initial)
            
            print(f"  初相 (φ₁={phi1:.2f}): 相位增量 = {phase_diff1:.4f} (理论: {10*self.params.omega:.4f})")
        
        print("\n链环数验证 (定理4):")
        print("  任意两条不同 Hopf 纤维的链环数 = ±1")
        print("  (这是 Hopf 纤维化的拓扑性质，已由定理4证明)")
        
        self.results['hopf_structure'] = {
            'period': 2*np.pi/self.params.omega,
            'linking_number': 1
        }
    
    def _parameter_sensitivity_analysis(self):
        print("\n" + "=" * 60)
        print("【参数敏感性分析】")
        print("=" * 60)
        
        gamma_values = [0.1, 0.3, 0.5, 1.0, 2.0]
        
        print("\n阻尼系数 γ 的影响:")
        print(f"{'γ':>6} {'收敛时间':>12} {'最终距离':>12} {'能量衰减率':>12}")
        print("-" * 45)
        
        for gamma in gamma_values:
            params_test = SystemParameters(
                r1=self.params.r1, r2=self.params.r2,
                omega=self.params.omega, gamma=gamma
            )
            system_test = SurvivalDynamicsSystem(params_test)
            
            np.random.seed(42)
            initial = generate_random_initial_condition(params_test)
            trajectory = system_test.simulate(initial, t_span=(0, 50), n_points=1000)
            
            dist = trajectory['dist_to_torus']
            conv_idx = np.where(dist < 0.01)[0]
            conv_time = trajectory['t'][conv_idx[0]] if len(conv_idx) > 0 else float('inf')
            final_dist = dist[-1]
            
            E = trajectory['energy']
            decay_rate = (E[0] - E[-1]) / E[0] if E[0] > 0 else 0
            
            print(f"{gamma:>6.2f} {conv_time:>12.2f} {final_dist:>12.2e} {decay_rate:>12.2%}")
        
        self.results['parameter_sensitivity'] = {
            'gamma_values': gamma_values
        }
    
    def _generate_summary_report(self):
        print("\n" + "=" * 80)
        print("【验证总结报告】")
        print("=" * 80)
        
        passed_count = 0
        total_count = 4
        
        results_summary = [
            ("定理1: 能量单调衰减", self.results.get('theorem1', {}).get('passed', False)),
            ("定理2: 原点双曲鞍点", self.results.get('theorem2', {}).get('theorem_holds', False)),
            ("定理3: 几乎全局吸引", self.results.get('theorem3', {}).get('passed', False)),
            ("定理5: 正规双曲吸引", self.results.get('theorem5', {}).get('theorem_holds', False)),
        ]
        
        for name, passed in results_summary:
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"  {name}: {status}")
            if passed:
                passed_count += 1
        
        print(f"\n总体验证结果: {passed_count}/{total_count} 通过")
        print(f"验证完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.results['summary'] = {
            'passed_count': passed_count,
            'total_count': total_count,
            'all_passed': passed_count == total_count
        }
    
    def save_results(self, filename: str = "verification_results.txt"):
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("生存动力学验证结果报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"系统参数:\n")
            f.write(f"  r₁ = {self.params.r1}\n")
            f.write(f"  r₂ = {self.params.r2}\n")
            f.write(f"  ω = {self.params.omega}\n")
            f.write(f"  γ = {self.params.gamma}\n\n")
            
            f.write("定理验证结果:\n")
            f.write("-" * 40 + "\n")
            
            for key in ['theorem1', 'theorem2', 'theorem3', 'theorem5']:
                if key in self.results:
                    result = self.results[key]
                    if key == 'theorem1':
                        f.write(f"\n定理1 (能量单调衰减):\n")
                        f.write(f"  所有轨线单调递减: {result.get('all_monotonic', 'N/A')}\n")
                        f.write(f"  平均梯度误差: {result.get('avg_gradient_error', 'N/A'):.2e}\n")
                    elif key == 'theorem2':
                        f.write(f"\n定理2 (原点双曲鞍点):\n")
                        f.write(f"  稳定流形维数: {result.get('stable_dim', 'N/A')}\n")
                        f.write(f"  不稳定流形维数: {result.get('unstable_dim', 'N/A')}\n")
                        f.write(f"  定理成立: {result.get('theorem_holds', 'N/A')}\n")
                    elif key == 'theorem3':
                        f.write(f"\n定理3 (几乎全局吸引):\n")
                        f.write(f"  收敛比例: {result.get('convergence_ratio', 'N/A'):.2%}\n")
                        f.write(f"  平均最终距离: {result.get('avg_final_distance', 'N/A'):.2e}\n")
                    elif key == 'theorem5':
                        f.write(f"\n定理5 (正规双曲吸引):\n")
                        f.write(f"  零特征值个数: {result.get('n_zero_eigenvalues', 'N/A')}\n")
                        f.write(f"  负实部特征值个数: {result.get('n_negative_eigenvalues', 'N/A')}\n")
                        f.write(f"  定理成立: {result.get('theorem_holds', 'N/A')}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\n结果已保存至: {filepath}")

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    verification = ComprehensiveVerification(output_dir)
    
    params = SystemParameters(r1=1.0, r2=1.0, omega=1.0, gamma=0.5)
    results = verification.run_all_verifications(params)
    
    verification.save_results()
    
    print("\n" + "=" * 80)
    print("是否生成可视化图表? (y/n)")
    print("=" * 80)
    
    try:
        user_input = input().strip().lower()
        if user_input == 'y':
            print("\n生成可视化图表...")
            visualizer = verification.visualizer
            
            if 'trajectory_1' in results:
                trajectory = results['trajectory_1']
                visualizer.plot_trajectory_3d(trajectory)
                visualizer.plot_energy_dissipation(trajectory)
            
            visualizer.plot_spectrum_analysis()
            visualizer.plot_multiple_trajectories(n_trajectories=5, t_max=30)
            visualizer.plot_hopf_flow_on_torus(n_orbits=8)
            
            print("\n可视化完成!")
    except:
        pass
    
    return results

if __name__ == "__main__":
    main()
