#!/usr/bin/env python3
"""
Phase 2: Main Scan and Phase Diagram
N=500, full parameter scan with multiple seeds
"""

import subprocess
import sys
import json
from datetime import datetime
from itertools import product

def run_simulation(N, steps, eta, lambda_, kc, alpha, sigma, seed):
    """Run single simulation"""
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout
        
        lines = output.split('\n')
        data = {
            'entropy': None,
            'total': None,
            'edges': None,
            'clusters': None,
            'wilson_peaks': None,
            'ricci_avg': None,
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
                for part in parts:
                    if part.startswith('peaks='):
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
            
            if 'Einstein relation:' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('R^2='):
                        try:
                            data['einstein_r2'] = float(part.split('=')[1])
                        except:
                            pass
            
            if 'Converged:' in line:
                data['converged'] = 'Yes' in line
        
        return data
    except:
        return {'error': 'Failed'}

def main():
    print("=" * 70)
    print("  Phase 2: Main Scan (N=500)")
    print("=" * 70)
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    N = 500
    steps = 800
    seeds = [12345, 23456, 34567]
    
    # Parameter ranges
    eta_values = [0.08, 0.1, 0.12]
    lambda_values = [0.9, 1.0, 1.1]
    kc_values = [0.08, 0.1, 0.12]
    alpha_values = [0.4, 0.5, 0.6]
    sigma_values = [0.1]
    
    total = len(eta_values) * len(lambda_values) * len(kc_values) * len(alpha_values) * len(sigma_values) * len(seeds)
    
    print(f"Parameter combinations: {total//len(seeds)}")
    print(f"Seeds per combination: {len(seeds)}")
    print(f"Total simulations: {total}")
    print()
    
    results = []
    count = 0
    
    for eta, lambda_, kc, alpha, sigma in product(
        eta_values, lambda_values, kc_values, alpha_values, sigma_values
    ):
        count += 1
        print(f"\n[{count}/{total//len(seeds)}] eta={eta:.2f}, lambda={lambda_:.1f}, kc={kc:.2f}, alpha={alpha:.1f}")
        
        seed_results = []
        for seed in seeds:
            data = run_simulation(N, steps, eta, lambda_, kc, alpha, sigma, seed)
            
            if 'error' not in data and data['entropy'] is not None:
                seed_results.append(data)
        
        if seed_results:
            avg_entropy = sum(r['entropy'] for r in seed_results) / len(seed_results)
            avg_total = sum(r['total'] for r in seed_results) / len(seed_results)
            avg_ricci = sum(r['ricci_avg'] for r in seed_results if r['ricci_avg']) / len(seed_results)
            
            result = {
                'eta': eta,
                'lambda': lambda_,
                'kc': kc,
                'alpha': alpha,
                'sigma': sigma,
                'avg_entropy': avg_entropy,
                'avg_total': avg_total,
                'avg_ricci': avg_ricci,
                'seeds_tested': len(seed_results)
            }
            results.append(result)
            
            print(f"  avg_entropy={avg_entropy:.4f}, avg_total={avg_total:.6f}, avg_ricci={avg_ricci:.4f}")
        else:
            print(f"  FAILED")
    
    print()
    print("=" * 70)
    print("  Phase 2 Results")
    print("=" * 70)
    print()
    
    print(f"Valid combinations: {len(results)}")
    print()
    
    if results:
        print(f"{'eta':<8} {'lambda':<8} {'kc':<8} {'alpha':<8} {'entropy':<12} {'total':<12} {'ricci':<10}")
        print("-" * 70)
        
        for r in sorted(results, key=lambda x: x['avg_entropy']):
            print(f"{r['eta']:<8.2f} {r['lambda']:<8.1f} {r['kc']:<8.2f} {r['alpha']:<8.1f} "
                  f"{r['avg_entropy']:<12.4f} {r['avg_total']:<12.6f} {r['avg_ricci']:<10.4f}")
        
        # Save results
        output_file = f"phase2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'N': N,
                'steps': steps,
                'results': results
            }, f, indent=2)
        
        print()
        print(f"Results saved to: {output_file}")
    
    print()
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
