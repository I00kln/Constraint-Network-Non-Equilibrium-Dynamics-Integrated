"""
可视化模块
为认知涌现验证系统提供可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class CognitiveVisualization:
    """认知涌现可视化工具"""
    
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
        
    def plot_potential_landscape(self,
                                potential_func,
                                x_range=(-2, 2),
                                y_range=(-2, 2),
                                title="认知势景观",
                                minima=None,
                                saddles=None):
        """绘制势函数景观"""
        fig = plt.figure(figsize=self.figsize)
        
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = potential_func(np.array([X[i, j], Y[i, j]]))
        
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_zlabel('V(x)')
        ax1.set_title(f'{title} - 3D视图')
        
        ax2 = fig.add_subplot(222)
        contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
        plt.colorbar(contour, ax=ax2, label='V(x)')
        
        if minima is not None and len(minima) > 0:
            minima = np.array(minima)
            ax2.scatter(minima[:, 0], minima[:, 1], 
                       c='red', s=100, marker='o', 
                       label='局部极小值', zorder=5)
        
        if saddles is not None and len(saddles) > 0:
            saddles = np.array(saddles)
            ax2.scatter(saddles[:, 0], saddles[:, 1],
                       c='yellow', s=100, marker='x',
                       label='鞍点', zorder=5)
        
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_title(f'{title} - 等高线图')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(223)
        grad_x = np.zeros_like(X)
        grad_y = np.zeros_like(Y)
        
        dx = (x_range[1] - x_range[0]) / 100
        dy = (y_range[1] - y_range[0]) / 100
        
        for i in range(1, X.shape[0] - 1):
            for j in range(1, X.shape[1] - 1):
                grad_x[i, j] = (Z[i, j+1] - Z[i, j-1]) / (2 * dx)
                grad_y[i, j] = (Z[i+1, j] - Z[i-1, j]) / (2 * dy)
        
        ax3.quiver(X[::5, ::5], Y[::5, ::5], 
                  -grad_x[::5, ::5], -grad_y[::5, ::5],
                  alpha=0.6)
        ax3.set_xlabel('x₁')
        ax3.set_ylabel('x₂')
        ax3.set_title('梯度流场')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(224)
        ax4.hist(Z.flatten(), bins=50, density=True, alpha=0.7, color='steelblue')
        ax4.set_xlabel('V(x)')
        ax4.set_ylabel('密度')
        ax4.set_title('势函数值分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_finsler_unit_ball(self,
                              unit_ball_points,
                              title="Finsler度量单位球"):
        """绘制Finsler度量的单位球"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        if len(unit_ball_points) > 0:
            unit_ball_points = np.array(unit_ball_points)
            ax1.fill(unit_ball_points[:, 0], unit_ball_points[:, 1],
                    alpha=0.3, color='steelblue')
            ax1.plot(unit_ball_points[:, 0], unit_ball_points[:, 1],
                    'b-', linewidth=2, label='单位球边界')
            
            circle = plt.Circle((0, 0), 1, fill=False, 
                               color='red', linestyle='--', 
                               linewidth=2, label='欧氏单位圆')
            ax1.add_patch(circle)
        
        ax1.set_xlabel('v₁')
        ax1.set_ylabel('v₂')
        ax1.set_title(f'{title}')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        if len(unit_ball_points) > 0:
            angles = np.arctan2(unit_ball_points[:, 1], unit_ball_points[:, 0])
            radii = np.linalg.norm(unit_ball_points, axis=1)
            
            sort_idx = np.argsort(angles)
            angles = angles[sort_idx]
            radii = radii[sort_idx]
            
            ax2.plot(np.degrees(angles), radii, 'b-', linewidth=2)
            ax2.axhline(y=1, color='red', linestyle='--', 
                       linewidth=2, label='欧氏参考')
        
        ax2.set_xlabel('方向角 (度)')
        ax2.set_ylabel('半径')
        ax2.set_title('各向异性分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cognitive_potential_field(self,
                                      V_field,
                                      x_coords,
                                      y_coords,
                                      title="认知势场"):
        """绘制认知势场"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        im1 = ax1.imshow(V_field, extent=[x_coords[0], x_coords[-1],
                                          y_coords[0], y_coords[-1]],
                        origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im1, ax=ax1, label='V(x)')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title(f'{title}')
        
        ax2 = axes[1]
        X, Y = np.meshgrid(x_coords, y_coords)
        
        grad_x = np.zeros_like(V_field)
        grad_y = np.zeros_like(V_field)
        
        dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1
        
        grad_x[:, 1:-1] = (V_field[:, 2:] - V_field[:, :-2]) / (2 * dx)
        grad_y[1:-1, :] = (V_field[2:, :] - V_field[:-2, :]) / (2 * dy)
        
        skip = max(1, len(x_coords) // 20)
        ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  -grad_x[::skip, ::skip].T, -grad_y[::skip, ::skip].T,
                  alpha=0.6)
        
        ax2.contour(X, Y, V_field.T, levels=15, colors='black', alpha=0.3)
        
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_title('认知势梯度流')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_transition_dynamics(self,
                                epsilon_values,
                                probabilities,
                                escape_times=None,
                                title="认知跃迁动力学"):
        """绘制跃迁动力学"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        ax1.semilogy(epsilon_values, probabilities, 'bo-', 
                    linewidth=2, markersize=8, label='数值结果')
        
        if len(epsilon_values) >= 2 and len(probabilities) >= 2:
            log_eps = np.log(epsilon_values)
            log_p = np.log(probabilities)
            coeffs = np.polyfit(log_eps, log_p, 1)
            
            eps_fit = np.linspace(min(epsilon_values), max(epsilon_values), 50)
            p_fit = np.exp(coeffs[1]) * eps_fit**coeffs[0]
            ax1.semilogy(eps_fit, p_fit, 'r--', 
                        linewidth=2, label=f'拟合: P ∝ ε^{coeffs[0]:.2f}')
        
        ax1.set_xlabel('噪声强度 ε')
        ax1.set_ylabel('跃迁概率 P')
        ax1.set_title('跃迁概率 vs 噪声强度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        if escape_times is not None and len(escape_times) > 0:
            ax2.semilogy(epsilon_values[:len(escape_times)], 
                        escape_times, 'go-',
                        linewidth=2, markersize=8, label='数值结果')
            
            if len(epsilon_values) >= 2:
                log_eps = np.log(epsilon_values[:len(escape_times)])
                log_t = np.log(escape_times)
                coeffs = np.polyfit(log_eps, log_t, 1)
                
                eps_fit = np.linspace(min(epsilon_values[:len(escape_times)]),
                                     max(epsilon_values[:len(escape_times)]), 50)
                t_fit = np.exp(coeffs[1]) * eps_fit**coeffs[0]
                ax2.semilogy(eps_fit, t_fit, 'r--',
                            linewidth=2, label=f'拟合: τ ∝ ε^{coeffs[0]:.2f}')
        
        ax2.set_xlabel('噪声强度 ε')
        ax2.set_ylabel('平均逃逸时间 τ')
        ax2.set_title('逃逸时间 vs 噪声强度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_aubry_set(self,
                      aubry_points,
                      peierls_matrix=None,
                      title="Aubry集与Peierls势垒"):
        """绘制Aubry集"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        if len(aubry_points) > 0:
            aubry_points = np.array(aubry_points)
            ax1.scatter(aubry_points[:, 0], aubry_points[:, 1],
                       c='red', s=50, alpha=0.7, label='Aubry集')
            
            if len(aubry_points) > 1:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(aubry_points)
                    for simplex in hull.simplices:
                        ax1.plot(aubry_points[simplex, 0],
                                aubry_points[simplex, 1], 'b-', alpha=0.3)
                except:
                    pass
        
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title(f'{title}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        ax2 = axes[1]
        if peierls_matrix is not None and len(peierls_matrix) > 0:
            im = ax2.imshow(peierls_matrix, cmap='hot', aspect='auto')
            plt.colorbar(im, ax=ax2, label='Peierls势垒')
            ax2.set_xlabel('状态索引')
            ax2.set_ylabel('状态索引')
            ax2.set_title('Peierls势垒矩阵')
        else:
            ax2.text(0.5, 0.5, '无Peierls势垒数据',
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        return fig
    
    def plot_geodesic(self,
                     geodesic_path,
                     straight_path=None,
                     title="测地线"):
        """绘制测地线"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        geodesic_path = np.array(geodesic_path)
        ax.plot(geodesic_path[:, 0], geodesic_path[:, 1],
               'b-', linewidth=3, label='测地线')
        ax.scatter(geodesic_path[0, 0], geodesic_path[0, 1],
                  c='green', s=100, marker='o', label='起点', zorder=5)
        ax.scatter(geodesic_path[-1, 0], geodesic_path[-1, 1],
                  c='red', s=100, marker='s', label='终点', zorder=5)
        
        if straight_path is not None:
            straight_path = np.array(straight_path)
            ax.plot(straight_path[:, 0], straight_path[:, 1],
                   'r--', linewidth=2, alpha=0.5, label='直线路径')
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'{title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_verification_summary(self,
                                 theorem_results: Dict):
        """绘制验证结果汇总"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        theorems = ['定理1\nΓ-收敛', '定理2\nHJB方程', '定理3\nEikonal方程',
                   '定理4\nFinsler度量', '定理5\n弱KAM结构', '定理6\n认知跃迁']
        
        ax = axes[0, 0]
        theorem_names = [t.split('\n')[0] for t in theorems]
        passed = [1 if theorem_results.get(f'theorem{i+1}', False) else 0 
                 for i in range(6)]
        colors = ['green' if p == 1 else 'red' for p in passed]
        
        bars = ax.bar(theorem_names, passed, color=colors, alpha=0.7)
        ax.set_ylim([0, 1.2])
        ax.set_ylabel('验证状态')
        ax.set_title('定理验证状态')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['未通过', '通过'])
        
        for i, (bar, p) in enumerate(zip(bars, passed)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{["未通过", "通过"][p]}',
                   ha='center', va='bottom', fontsize=9)
        
        ax = axes[0, 1]
        if 'gamma_convergence_errors' in theorem_results:
            errors = theorem_results['gamma_convergence_errors']
            ax.plot(errors, 'b-', linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('误差')
            ax.set_title('Γ-收敛误差演化')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes)
        
        ax = axes[0, 2]
        if 'finsler_eigenvalues' in theorem_results:
            eigenvalues = theorem_results['finsler_eigenvalues']
            ax.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.7)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('特征值索引')
            ax.set_ylabel('特征值')
            ax.set_title('基本张量特征值')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes)
        
        ax = axes[1, 0]
        if 'barrier_heights' in theorem_results:
            heights = theorem_results['barrier_heights']
            ax.bar(range(len(heights)), heights, color='orange', alpha=0.7)
            ax.set_xlabel('跃迁路径')
            ax.set_ylabel('势垒高度')
            ax.set_title('Peierls势垒高度')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes)
        
        ax = axes[1, 1]
        if 'transition_probabilities' in theorem_results:
            probs = theorem_results['transition_probabilities']
            eps = theorem_results.get('epsilon_values', range(len(probs)))
            ax.semilogy(eps, probs, 'bo-', linewidth=2, markersize=6)
            ax.set_xlabel('噪声强度 ε')
            ax.set_ylabel('跃迁概率')
            ax.set_title('跃迁概率衰减')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes)
        
        ax = axes[1, 2]
        summary_text = "验证结果汇总:\n\n"
        for i, theorem in enumerate(theorems):
            status = "✓ 通过" if theorem_results.get(f'theorem{i+1}', False) else "✗ 未通过"
            summary_text += f"{theorem}: {status}\n"
        
        ax.text(0.1, 0.5, summary_text, ha='left', va='center',
               transform=ax.transAxes, fontsize=11, family='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        return fig


def create_summary_report(results: Dict) -> str:
    """创建验证结果汇总报告"""
    report = []
    report.append("=" * 70)
    report.append("认知与规律涌现框架 - 数值验证报告")
    report.append("=" * 70)
    report.append("")
    
    report.append("一、核心定理验证状态")
    report.append("-" * 70)
    
    theorems = [
        ("定理1", "离散到连续的桥接定理（Γ-收敛）"),
        ("定理2", "认知最优控制定理（动态HJB方程）"),
        ("定理3", "认知静态传播定理（静态Eikonal方程）"),
        ("定理4", "认知几何涌现定理（Finsler度量）"),
        ("定理5", "认知相变与分类定理（弱KAM结构）"),
        ("定理6", "认知跃迁定理（大偏差理论）")
    ]
    
    for i, (name, desc) in enumerate(theorems, 1):
        status = results.get(f'theorem{i}', False)
        status_str = "✓ 通过" if status else "✗ 未通过"
        report.append(f"{name}: {desc}")
        report.append(f"  验证状态: {status_str}")
        report.append("")
    
    report.append("二、详细验证结果")
    report.append("-" * 70)
    
    if 'gamma_convergence' in results:
        report.append("Γ-收敛验证:")
        gc = results['gamma_convergence']
        report.append(f"  下界条件: {'通过' if gc.get('lower_bound', False) else '未通过'}")
        report.append(f"  恢复条件: {'通过' if gc.get('recovery', False) else '未通过'}")
        report.append("")
    
    if 'hjb_verification' in results:
        report.append("HJB方程验证:")
        hjb = results['hjb_verification']
        report.append(f"  粘性解验证: {'通过' if hjb.get('viscosity', False) else '未通过'}")
        report.append(f"  半群性质: {'通过' if hjb.get('semigroup', False) else '未通过'}")
        report.append("")
    
    if 'finsler_verification' in results:
        report.append("Finsler度量验证:")
        finsler = results['finsler_verification']
        report.append(f"  齐次性: {'通过' if finsler.get('homogeneity', False) else '未通过'}")
        report.append(f"  凸性: {'通过' if finsler.get('convexity', False) else '未通过'}")
        report.append(f"  正定性: {'通过' if finsler.get('positive_definite', False) else '未通过'}")
        report.append(f"  各向异性: {'通过' if finsler.get('anisotropy', False) else '未通过'}")
        report.append("")
    
    if 'weak_kam' in results:
        report.append("弱KAM结构验证:")
        kam = results['weak_kam']
        report.append(f"  Aubry集大小: {kam.get('aubry_size', 0)}")
        report.append(f"  Mañé临界值: {kam.get('mane_critical', 0):.6f}")
        report.append("")
    
    if 'transition' in results:
        report.append("认知跃迁验证:")
        trans = results['transition']
        report.append(f"  亚稳态数量: {trans.get('num_minima', 0)}")
        report.append(f"  鞍点数量: {trans.get('num_saddles', 0)}")
        if 'barrier_height' in trans:
            report.append(f"  势垒高度: {trans['barrier_height']:.6f}")
        report.append("")
    
    report.append("三、结论")
    report.append("-" * 70)
    
    all_passed = all(results.get(f'theorem{i}', False) for i in range(1, 7))
    
    if all_passed:
        report.append("所有核心定理均通过数值验证，认知与规律涌现框架的数学结构")
        report.append("在数值层面得到了确认。")
    else:
        report.append("部分定理验证未完全通过，需要进一步调整参数或改进算法。")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("认知涌现可视化模块")
    print("提供以下可视化功能:")
    print("  1. 势函数景观可视化")
    print("  2. Finsler度量单位球可视化")
    print("  3. 认知势场可视化")
    print("  4. 跃迁动力学可视化")
    print("  5. Aubry集可视化")
    print("  6. 测地线可视化")
    print("  7. 验证结果汇总可视化")
