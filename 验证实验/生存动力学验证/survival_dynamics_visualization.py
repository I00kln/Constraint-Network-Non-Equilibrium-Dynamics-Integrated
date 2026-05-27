"""
生存动力学可视化模块
展示轨线收敛到 Clifford 环面的过程、能量衰减、谱分析等
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from typing import Dict, Optional
from survival_dynamics_core import (
    SystemParameters, SurvivalDynamicsSystem, TheoremVerifier,
    generate_random_initial_condition, generate_initial_on_torus
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DynamicsVisualizer:
    def __init__(self, system: SurvivalDynamicsSystem):
        self.sys = system
        self.params = system.params
    
    def plot_trajectory_3d(self, trajectory: Dict, save_path: str = None):
        fig = plt.figure(figsize=(14, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        z1 = trajectory['z1']
        x1, y1 = np.real(z1), np.imag(z1)
        ax1.plot(x1, y1, np.abs(z1), 'b-', alpha=0.7, linewidth=1)
        ax1.scatter([x1[0]], [y1[0]], [np.abs(z1)[0]], color='green', s=50, label='起点')
        ax1.scatter([x1[-1]], [y1[-1]], [np.abs(z1)[-1]], color='red', s=50, label='终点')
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(self.params.r1 * np.cos(theta), self.params.r1 * np.sin(theta), 
                np.ones_like(theta) * self.params.r1, 'r--', linewidth=2, label='目标环')
        ax1.set_xlabel('Re(z₁)')
        ax1.set_ylabel('Im(z₁)')
        ax1.set_zlabel('|z₁|')
        ax1.set_title('z₁ 分量轨迹')
        ax1.legend()
        
        ax2 = fig.add_subplot(132, projection='3d')
        z2 = trajectory['z2']
        x2, y2 = np.real(z2), np.imag(z2)
        ax2.plot(x2, y2, np.abs(z2), 'b-', alpha=0.7, linewidth=1)
        ax2.scatter([x2[0]], [y2[0]], [np.abs(z2)[0]], color='green', s=50, label='起点')
        ax2.scatter([x2[-1]], [y2[-1]], [np.abs(z2)[-1]], color='red', s=50, label='终点')
        ax2.plot(self.params.r2 * np.cos(theta), self.params.r2 * np.sin(theta), 
                np.ones_like(theta) * self.params.r2, 'r--', linewidth=2, label='目标环')
        ax2.set_xlabel('Re(z₂)')
        ax2.set_ylabel('Im(z₂)')
        ax2.set_zlabel('|z₂|')
        ax2.set_title('z₂ 分量轨迹')
        ax2.legend()
        
        ax3 = fig.add_subplot(133)
        t = trajectory['t']
        dist = trajectory['dist_to_torus']
        ax3.semilogy(t, dist, 'b-', linewidth=1.5)
        ax3.set_xlabel('时间 t')
        ax3.set_ylabel('到环面距离 (对数尺度)')
        ax3.set_title('收敛到 Clifford 环面')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_energy_dissipation(self, trajectory: Dict, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t = trajectory['t']
        
        ax1 = axes[0, 0]
        ax1.plot(t, trajectory['energy'], 'b-', linewidth=1.5)
        ax1.set_xlabel('时间 t')
        ax1.set_ylabel('机械能 E')
        ax1.set_title('定理1: 能量单调衰减')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(t, trajectory['dissipation'], 'r-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('时间 t')
        ax2.set_ylabel('耗散率 dE/dt')
        ax2.set_title('能量耗散率 (始终非正)')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        v1_norm = np.abs(trajectory['v1'])
        v2_norm = np.abs(trajectory['v2'])
        ax3.plot(t, v1_norm, 'g-', linewidth=1.5, label='|v₁|')
        ax3.plot(t, v2_norm, 'm-', linewidth=1.5, label='|v₂|')
        ax3.set_xlabel('时间 t')
        ax3.set_ylabel('速度误差范数')
        ax3.set_title('速度误差衰减')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        r1_dev = np.abs(np.abs(trajectory['z1']) - self.params.r1)
        r2_dev = np.abs(np.abs(trajectory['z2']) - self.params.r2)
        ax4.semilogy(t, r1_dev, 'c-', linewidth=1.5, label='| |z₁| - r₁ |')
        ax4.semilogy(t, r2_dev, 'orange', linewidth=1.5, label='| |z₂| - r₂ |')
        ax4.set_xlabel('时间 t')
        ax4.set_ylabel('径向偏差 (对数尺度)')
        ax4.set_title('径向坐标收敛')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_spectrum_analysis(self, save_path: str = None):
        verifier = TheoremVerifier(self.sys)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        result_origin = verifier.verify_theorem2_hyperbolic_saddle()
        eigenvalues_origin = result_origin['eigenvalues']
        
        ax1 = axes[0]
        ax1.scatter(np.real(eigenvalues_origin), np.imag(eigenvalues_origin), 
                   c='blue', s=80, alpha=0.7, edgecolors='black')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('实部 Re(λ)')
        ax1.set_ylabel('虚部 Im(λ)')
        ax1.set_title(f'定理2: 原点处线性化谱\n(双曲鞍点: 4正+4负实部)')
        ax1.grid(True, alpha=0.3)
        
        for i, ev in enumerate(eigenvalues_origin):
            ax1.annotate(f'{i+1}', (np.real(ev), np.imag(ev)), 
                        textcoords="offset points", xytext=(5, 5))
        
        result_torus = verifier.verify_theorem5_normal_hyperbolicity()
        eigenvalues_torus = result_torus['eigenvalues']
        
        ax2 = axes[1]
        colors = ['red' if np.abs(np.real(ev)) < 1e-10 else 'blue' 
                  for ev in eigenvalues_torus]
        ax2.scatter(np.real(eigenvalues_torus), np.imag(eigenvalues_torus), 
                   c=colors, s=80, alpha=0.7, edgecolors='black')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('实部 Re(λ)')
        ax2.set_ylabel('虚部 Im(λ)')
        ax2.set_title(f'定理5: 环面上线性化谱\n(2个零特征值 + 6个负实部)')
        ax2.grid(True, alpha=0.3)
        
        for i, ev in enumerate(eigenvalues_torus):
            ax2.annotate(f'{i+1}', (np.real(ev), np.imag(ev)), 
                        textcoords="offset points", xytext=(5, 5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return result_origin, result_torus
    
    def plot_multiple_trajectories(self, n_trajectories: int = 5, 
                                   t_max: float = 30, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        
        for i in range(n_trajectories):
            np.random.seed(42 + i)
            initial = generate_random_initial_condition(self.params)
            trajectory = self.sys.simulate(initial, t_span=(0, t_max), n_points=1000)
            
            t = trajectory['t']
            
            axes[0, 0].plot(t, trajectory['energy'], color=colors[i], 
                           linewidth=1.5, alpha=0.7, label=f'轨线{i+1}')
            
            axes[0, 1].semilogy(t, trajectory['dist_to_torus'], color=colors[i], 
                               linewidth=1.5, alpha=0.7)
            
            axes[1, 0].plot(np.real(trajectory['z1']), np.imag(trajectory['z1']), 
                           color=colors[i], linewidth=1, alpha=0.7)
            
            axes[1, 1].plot(np.real(trajectory['z2']), np.imag(trajectory['z2']), 
                           color=colors[i], linewidth=1, alpha=0.7)
        
        axes[0, 0].set_xlabel('时间 t')
        axes[0, 0].set_ylabel('机械能 E')
        axes[0, 0].set_title('多条轨线的能量衰减')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('时间 t')
        axes[0, 1].set_ylabel('到环面距离')
        axes[0, 1].set_title('多条轨线收敛到环面')
        axes[0, 1].grid(True, alpha=0.3)
        
        theta = np.linspace(0, 2*np.pi, 100)
        axes[1, 0].plot(self.params.r1 * np.cos(theta), self.params.r1 * np.sin(theta), 
                       'r--', linewidth=2, label='目标环')
        axes[1, 0].set_xlabel('Re(z₁)')
        axes[1, 0].set_ylabel('Im(z₁)')
        axes[1, 0].set_title('z₁ 平面投影')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(self.params.r2 * np.cos(theta), self.params.r2 * np.sin(theta), 
                       'r--', linewidth=2, label='目标环')
        axes[1, 1].set_xlabel('Re(z₂)')
        axes[1, 1].set_ylabel('Im(z₂)')
        axes[1, 1].set_title('z₂ 平面投影')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_hopf_flow_on_torus(self, n_orbits: int = 8, save_path: str = None):
        fig = plt.figure(figsize=(14, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        t = np.linspace(0, 2*np.pi, 200)
        
        for i in range(n_orbits):
            phi1 = 2 * np.pi * i / n_orbits
            phi2 = 0
            
            z1 = self.params.r1 * np.exp(1j * (t + phi1))
            z2 = self.params.r2 * np.exp(1j * (t + phi2))
            
            x = np.real(z1)
            y = np.imag(z1)
            z = np.real(z2)
            
            ax1.plot(x, y, z, alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel('Re(z₁)')
        ax1.set_ylabel('Im(z₁)')
        ax1.set_zlabel('Re(z₂)')
        ax1.set_title('Clifford 环面上的 Hopf 流\n(周期轨道族)')
        
        ax2 = fig.add_subplot(132)
        for i in range(n_orbits):
            phi1 = 2 * np.pi * i / n_orbits
            z1 = self.params.r1 * np.exp(1j * (t + phi1))
            ax2.plot(np.real(z1), np.imag(z1), alpha=0.7, linewidth=1.5)
        
        ax2.set_xlabel('Re(z₁)')
        ax2.set_ylabel('Im(z₁)')
        ax2.set_title('z₁ 分量: 圆周轨道')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(133)
        for i in range(n_orbits):
            phi2 = 2 * np.pi * i / n_orbits
            z2 = self.params.r2 * np.exp(1j * (t + phi2))
            ax3.plot(np.real(z2), np.imag(z2), alpha=0.7, linewidth=1.5)
        
        ax3.set_xlabel('Re(z₂)')
        ax3.set_ylabel('Im(z₂)')
        ax3.set_title('z₂ 分量: 圆周轨道')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_animation(self, trajectory: Dict, save_path: str = None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        z1 = trajectory['z1']
        z2 = trajectory['z2']
        t = trajectory['t']
        
        ax1, ax2, ax3 = axes
        
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(self.params.r1 * np.cos(theta), self.params.r1 * np.sin(theta), 
                'r--', linewidth=2, label='目标环')
        line1, = ax1.plot([], [], 'b-', linewidth=1.5, alpha=0.5)
        point1, = ax1.plot([], [], 'go', markersize=10)
        ax1.set_xlim(-2*self.params.r1, 2*self.params.r1)
        ax1.set_ylim(-2*self.params.r1, 2*self.params.r1)
        ax1.set_xlabel('Re(z₁)')
        ax1.set_ylabel('Im(z₁)')
        ax1.set_title('z₁ 分量')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.params.r2 * np.cos(theta), self.params.r2 * np.sin(theta), 
                'r--', linewidth=2, label='目标环')
        line2, = ax2.plot([], [], 'b-', linewidth=1.5, alpha=0.5)
        point2, = ax2.plot([], [], 'go', markersize=10)
        ax2.set_xlim(-2*self.params.r2, 2*self.params.r2)
        ax2.set_ylim(-2*self.params.r2, 2*self.params.r2)
        ax2.set_xlabel('Re(z₂)')
        ax2.set_ylabel('Im(z₂)')
        ax2.set_title('z₂ 分量')
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        dist = trajectory['dist_to_torus']
        ax3.semilogy(t, dist, 'b-', linewidth=1, alpha=0.3)
        current_dist, = ax3.semilogy([], [], 'ro', markersize=8)
        ax3.set_xlabel('时间 t')
        ax3.set_ylabel('到环面距离')
        ax3.set_title('收敛过程')
        ax3.grid(True, alpha=0.3)
        
        time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
        
        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            line2.set_data([], [])
            point2.set_data([], [])
            current_dist.set_data([], [])
            time_text.set_text('')
            return line1, point1, line2, point2, current_dist, time_text
        
        def animate(frame):
            idx = frame * 5
            if idx >= len(t):
                idx = len(t) - 1
            
            line1.set_data(np.real(z1[:idx]), np.imag(z1[:idx]))
            point1.set_data([np.real(z1[idx])], [np.imag(z1[idx])])
            
            line2.set_data(np.real(z2[:idx]), np.imag(z2[:idx]))
            point2.set_data([np.real(z2[idx])], [np.imag(z2[idx])])
            
            current_dist.set_data([t[idx]], [dist[idx]])
            
            time_text.set_text(f't = {t[idx]:.2f}, 距离 = {dist[idx]:.2e}')
            
            return line1, point1, line2, point2, current_dist, time_text
        
        n_frames = len(t) // 5
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=50, blit=True)
        
        plt.tight_layout()
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.show()
        return anim

def run_visualization_demo():
    print("=" * 70)
    print("生存动力学可视化演示")
    print("=" * 70)
    
    params = SystemParameters(r1=1.0, r2=1.0, omega=1.0, gamma=0.5)
    system = SurvivalDynamicsSystem(params)
    visualizer = DynamicsVisualizer(system)
    
    print("\n1. 单条轨线 3D 可视化...")
    np.random.seed(42)
    initial = generate_random_initial_condition(params)
    trajectory = system.simulate(initial, t_span=(0, 40), n_points=2000)
    visualizer.plot_trajectory_3d(trajectory)
    
    print("\n2. 能量耗散分析...")
    visualizer.plot_energy_dissipation(trajectory)
    
    print("\n3. 谱分析...")
    visualizer.plot_spectrum_analysis()
    
    print("\n4. 多条轨线对比...")
    visualizer.plot_multiple_trajectories(n_trajectories=5, t_max=30)
    
    print("\n5. Clifford 环面上的 Hopf 流...")
    visualizer.plot_hopf_flow_on_torus(n_orbits=8)
    
    print("\n可视化演示完成!")

if __name__ == "__main__":
    run_visualization_demo()
