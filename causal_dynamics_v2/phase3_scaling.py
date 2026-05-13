#!/usr/bin/env python3
"""
Phase 3: Finite Size Scaling Analysis
根据数值仿真计划v5.md阶段3要求：
- 在普适区固定参数
- 运行N=400,500,600,700
- 收集有限尺度标度数据：谱簇数量、间隙、Wilson峰分布宽度
"""

import subprocess
import json
import os
import re
from datetime import datetime

def run_simulation(N, steps, eta, lambda_, kc, alpha, sigma, seed):
    """运行单个模拟"""
    cmd = [
        './causal_sim_v2', 'single',
        '-N', str(N),
        '-steps', str(steps),
        '-eta', str(eta),
        '-lambda', str(lambda_),
        '-kc', str(kc),
        '-alpha', str(alpha),
        '-sigma', str(sigma),
        '-seed', str(seed)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        
        data = {
            'N': N,
            'steps': steps,
            'seed': seed,
            'success': result.returncode == 0
        }
        
        lines = output.split('\n')
        for line in lines:
            if 'Step' in line and 'entropy=' in line:
                parts = line.replace(',', ' ').split()
                for part in parts:
                    if part.startswith('entropy='):
                        try:
                            data['entropy'] = float(part.split('=')[1])
                        except:
                            pass
                    elif part.startswith('total='):
                        try:
                            data['total_info'] = float(part.split('=')[1])
                        except:
                            pass
                    elif part.startswith('edges='):
                        try:
                            data['edges'] = int(part.split('=')[1])
                        except:
                            pass
                    elif part.startswith('clusters='):
                        try:
                            data['clusters'] = int(part.split('=')[1])
                        except:
                            pass
        
        for line in lines:
            if 'Wilson loops:' in line:
                match = re.search(r'peaks=(\d+)', line)
                if match:
                    data['wilson_peaks'] = int(match.group(1))
            elif 'Ricci curvature:' in line:
                match = re.search(r'avg=([0-9.]+)', line)
                if match:
                    data['ricci_avg'] = float(match.group(1))
            elif 'Spectral clusters:' in line:
                match = re.search(r'Spectral clusters:\s*(\d+)', line)
                if match:
                    data['spectral_clusters'] = int(match.group(1))
        
        return data
    except subprocess.TimeoutExpired:
        return {'N': N, 'success': False, 'error': 'timeout'}
    except Exception as e:
        return {'N': N, 'success': False, 'error': str(e)}

def main():
    print("=" * 60)
    print("Phase 3: Finite Size Scaling Analysis")
    print("=" * 60)
    
    eta = 0.1
    lambda_ = 1.0
    kc = 0.1
    alpha = 0.5
    sigma = 0.1
    
    N_values = [400, 500, 600, 700]
    steps = 1000
    seeds = [12345, 23456]
    
    results = []
    
    for N in N_values:
        print(f"\n{'='*60}")
        print(f"N = {N}")
        print(f"{'='*60}")
        
        N_results = []
        
        for seed in seeds:
            print(f"  Running seed={seed}...", end=' ', flush=True)
            result = run_simulation(N, steps, eta, lambda_, kc, alpha, sigma, seed)
            
            if result.get('success'):
                print(f"OK - clusters={result.get('spectral_clusters', 'N/A')}, "
                      f"peaks={result.get('wilson_peaks', 'N/A')}")
                N_results.append(result)
            else:
                print(f"FAILED - {result.get('error', 'unknown')}")
        
        if N_results:
            avg_clusters = sum(r.get('spectral_clusters', 0) for r in N_results) / len(N_results)
            avg_entropy = sum(r.get('entropy', 0) for r in N_results) / len(N_results)
            avg_ricci = sum(r.get('ricci_avg', 0) for r in N_results) / len(N_results)
            avg_peaks = sum(r.get('wilson_peaks', 0) for r in N_results) / len(N_results)
            
            results.append({
                'N': N,
                'avg_clusters': avg_clusters,
                'avg_entropy': avg_entropy,
                'avg_ricci': avg_ricci,
                'avg_peaks': avg_peaks,
                'samples': len(N_results)
            })
            
            print(f"\n  Summary for N={N}:")
            print(f"    Spectral clusters: {avg_clusters:.2f}")
            print(f"    Entropy: {avg_entropy:.6f}")
            print(f"    Ricci curvature: {avg_ricci:.4f}")
            print(f"    Wilson peaks: {avg_peaks:.1f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'phase3_scaling_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Finite Size Scaling Results")
    print(f"{'='*60}")
    print(f"{'N':>6} {'Clusters':>10} {'Entropy':>12} {'Ricci':>10} {'Peaks':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['N']:>6} {r['avg_clusters']:>10.2f} {r['avg_entropy']:>12.6f} "
              f"{r['avg_ricci']:>10.4f} {r['avg_peaks']:>8.1f}")
    
    print(f"\n{'='*60}")
    print("Scaling Analysis")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        clusters = [r['avg_clusters'] for r in results]
        Ns = [r['N'] for r in results]
        
        print(f"\nSpectral clusters vs N:")
        for i, r in enumerate(results):
            print(f"  N={r['N']}: {r['avg_clusters']:.2f} clusters")
        
        if max(clusters) - min(clusters) < 1.0:
            print("\n✓ Cluster number appears SIZE-INDEPENDENT (converged)")
        else:
            print("\n✗ Cluster number shows SIZE DEPENDENCE (not converged)")
        
        riccis = [r['avg_ricci'] for r in results]
        print(f"\nRicci curvature vs N:")
        for r in results:
            print(f"  N={r['N']}: {r['avg_ricci']:.4f}")
        
        if abs(max(riccis) - min(riccis)) < 0.1:
            print("\n✓ Ricci curvature appears SIZE-INDEPENDENT (converged)")
        else:
            print("\n✗ Ricci curvature shows SIZE DEPENDENCE")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
