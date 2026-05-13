#!/usr/bin/env python3
"""
Phase 1: Quick parameter scan for testing
"""

import subprocess
import sys
from datetime import datetime

def run_simulation(N, steps, eta, lambda_, kc, alpha, sigma, seed=12345):
    """Run single simulation and parse results"""
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        
        lines = output.split('\n')
        data = {
            'entropy': None,
            'total': None,
            'edges': None,
            'clusters': None,
            'wilson_loops': None,
            'wilson_mod': None,
            'wilson_peaks': None,
            'ricci_avg': None,
            'ricci_std': None,
            'einstein_slope': None,
            'einstein_r2': None,
            'converged': False
        }
        
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
                            data['total'] = float(part.split('=')[1])
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
            
            if 'Wilson loops:' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('found'):
                        data['wilson_loops'] = int(parts[i-1])
                    elif part.startswith('avg_mod='):
                        try:
                            data['wilson_mod'] = float(part.split('=')[1])
                        except:
                            pass
                    elif part.startswith('peaks='):
                        try:
                            data['wilson_peaks'] = int(part.split('=')[1])
                        except:
                            pass
            
            if 'Ricci curvature:' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('avg='):
                        try:
                            data['ricci_avg'] = float(part.split('=')[1])
                        except:
                            pass
                    elif part.startswith('std='):
                        try:
                            data['ricci_std'] = float(part.split('=')[1])
                        except:
                            pass
            
            if 'Einstein relation:' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('slope='):
                        try:
                            data['einstein_slope'] = float(part.split('=')[1])
                        except:
                            pass
                    elif part.startswith('R^2='):
                        try:
                            data['einstein_r2'] = float(part.split('=')[1])
                        except:
                            pass
            
            if 'Converged:' in line:
                data['converged'] = 'Yes' in line
        
        return data
    except subprocess.TimeoutExpired:
        return {'error': 'Timeout'}
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 70)
    print("  Phase 1: Quick Parameter Scan (N=300)")
    print("=" * 70)
    print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Quick scan with fewer combinations
    N = 300
    steps = 500
    
    eta_values = [0.05, 0.1, 0.15]
    lambda_values = [0.8, 1.0, 1.2]
    kc_values = [0.05, 0.1]
    alpha_values = [0.3, 0.5]
    sigma_values = [0.1]
    
    results = []
    count = 0
    total = len(eta_values) * len(lambda_values) * len(kc_values) * len(alpha_values) * len(sigma_values)
    
    print(f"Total: {total} combinations\n")
    
    from itertools import product
    for eta, lambda_, kc, alpha, sigma in product(
        eta_values, lambda_values, kc_values, alpha_values, sigma_values
    ):
        count += 1
        print(f"[{count}/{total}] eta={eta:.2f}, lambda={lambda_:.1f}, kc={kc:.2f}, alpha={alpha:.1f}")
        
        data = run_simulation(N, steps, eta, lambda_, kc, alpha, sigma)
        
        if 'error' not in data and data['entropy'] is not None:
            result = {
                'N': N,
                'eta': eta,
                'lambda': lambda_,
                'kc': kc,
                'alpha': alpha,
                'sigma': sigma,
                **data
            }
            results.append(result)
            
            print(f"  entropy={data['entropy']:.4f}, total={data['total']:.6f}, "
                  f"edges={data['edges']}, W-peaks={data['wilson_peaks']}")
        else:
            print(f"  ERROR: {data.get('error', 'Unknown')}")
    
    print()
    print("=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print()
    
    print(f"Valid: {len(results)}/{total}")
    print()
    
    if results:
        # Sort by Wilson peaks
        sorted_results = sorted(results, key=lambda x: x.get('wilson_peaks', 0) or 0, reverse=True)
        
        print(f"{'eta':<8} {'lambda':<8} {'kc':<8} {'alpha':<8} {'entropy':<12} {'total':<12} {'W-peaks':<10} {'Ricci':<10}")
        print("-" * 80)
        
        for r in sorted_results[:10]:
            print(f"{r['eta']:<8.2f} {r['lambda']:<8.1f} {r['kc']:<8.2f} {r['alpha']:<8.1f} "
                  f"{r['entropy']:<12.4f} {r['total']:<12.6f} "
                  f"{r['wilson_peaks']:<10} {r['ricci_avg']:<10.4f}")
        
        print()
        
        # Recommend parameters
        if sorted_results:
            best = sorted_results[0]
            print("Recommended for Phase 2:")
            print(f"  eta = {best['eta']:.2f}")
            print(f"  lambda = {best['lambda']:.1f}")
            print(f"  kc = {best['kc']:.2f}")
            print(f"  alpha = {best['alpha']:.1f}")
    
    print()
    print(f"End: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == '__main__':
    main()
