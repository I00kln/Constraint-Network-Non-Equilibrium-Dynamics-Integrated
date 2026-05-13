#!/usr/bin/env python3
"""
参数扫描脚本 - 寻找最优参数组合
"""

import subprocess
import sys
from itertools import product

def run_simulation(N, steps, eta, lambda_, kc, alpha, sigma):
    """运行单次模拟"""
    cmd = [
        './causal_sim_v2', 'single',
        '-N', str(N),
        '-steps', str(steps),
        '-eta', str(eta),
        '-lambda', str(lambda_),
        '-kc', str(kc),
        '-alpha', str(alpha),
        '-sigma', str(sigma)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # 解析输出
        lines = output.split('\n')
        final_entropy = None
        final_edges = None
        wilson_peaks = None
        ricci_avg = None
        converged = None
        
        for line in lines:
            if 'Step' in line and 'entropy=' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('entropy='):
                        final_entropy = float(part.split('=')[1])
                    if part.startswith('edges='):
                        final_edges = int(part.split('=')[1])
            
            if 'Wilson loops:' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('peaks='):
                        wilson_peaks = int(part.split('=')[1])
            
            if 'Ricci curvature:' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('avg='):
                        ricci_avg = float(part.split('=')[1])
            
            if 'Converged:' in line:
                converged = 'Yes' in line
        
        return {
            'entropy': final_entropy,
            'edges': final_edges,
            'wilson_peaks': wilson_peaks,
            'ricci_avg': ricci_avg,
            'converged': converged
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 70)
    print("  Parameter Scan for Causal Dynamics V2.0")
    print("=" * 70)
    print()
    
    # 参数范围
    N_values = [200, 300]
    eta_values = [0.05, 0.1, 0.15]
    lambda_values = [0.8, 1.0, 1.2]
    kc_values = [0.05, 0.1, 0.15]
    alpha_values = [0.3, 0.5]
    sigma_values = [0.1]
    
    steps = 500
    
    total_combinations = len(N_values) * len(eta_values) * len(lambda_values) * \
                        len(kc_values) * len(alpha_values) * len(sigma_values)
    
    print(f"Total combinations to test: {total_combinations}")
    print()
    
    results = []
    count = 0
    
    for N, eta, lambda_, kc, alpha, sigma in product(
        N_values, eta_values, lambda_values, kc_values, alpha_values, sigma_values
    ):
        count += 1
        print(f"[{count}/{total_combinations}] N={N}, eta={eta}, lambda={lambda_}, kc={kc}, alpha={alpha}")
        
        result = run_simulation(N, steps, eta, lambda_, kc, alpha, sigma)
        
        if 'error' not in result:
            results.append({
                'N': N,
                'eta': eta,
                'lambda': lambda_,
                'kc': kc,
                'alpha': alpha,
                'sigma': sigma,
                **result
            })
            
            print(f"  entropy={result['entropy']:.6f}, edges={result['edges']}, "
                  f"wilson_peaks={result['wilson_peaks']}, ricci={result['ricci_avg']:.4f}, "
                  f"converged={result['converged']}")
        else:
            print(f"  ERROR: {result['error']}")
        
        print()
    
    # 输出最佳结果
    print("=" * 70)
    print("  Top Results (by Wilson peaks)")
    print("=" * 70)
    print()
    
    # 按Wilson peaks排序
    sorted_results = sorted(results, key=lambda x: x.get('wilson_peaks', 0), reverse=True)
    
    print(f"{'Rank':<5} {'N':<5} {'eta':<8} {'lambda':<8} {'kc':<8} {'alpha':<8} "
          f"{'entropy':<12} {'edges':<8} {'W-peaks':<8} {'ricci':<8} {'conv':<6}")
    print("-" * 100)
    
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1:<5} {r['N']:<5} {r['eta']:<8.3f} {r['lambda']:<8.3f} {r['kc']:<8.3f} "
              f"{r['alpha']:<8.3f} {r['entropy']:<12.6f} {r['edges']:<8} "
              f"{r['wilson_peaks']:<8} {r['ricci_avg']:<8.4f} {str(r['converged']):<6}")
    
    print()
    print("=" * 70)
    print("  Top Results (by Ricci curvature)")
    print("=" * 70)
    print()
    
    # 按Ricci曲率排序
    sorted_results = sorted(results, key=lambda x: x.get('ricci_avg', 0), reverse=True)
    
    print(f"{'Rank':<5} {'N':<5} {'eta':<8} {'lambda':<8} {'kc':<8} {'alpha':<8} "
          f"{'entropy':<12} {'edges':<8} {'W-peaks':<8} {'ricci':<8} {'conv':<6}")
    print("-" * 100)
    
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1:<5} {r['N']:<5} {r['eta']:<8.3f} {r['lambda']:<8.3f} {r['kc']:<8.3f} "
              f"{r['alpha']:<8.3f} {r['entropy']:<12.6f} {r['edges']:<8} "
              f"{r['wilson_peaks']:<8} {r['ricci_avg']:<8.4f} {str(r['converged']):<6}")

if __name__ == '__main__':
    main()
