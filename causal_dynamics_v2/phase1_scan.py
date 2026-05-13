#!/usr/bin/env python3
"""
Phase 1: Small-scale Exploration and Parameter Screening
N=500, coarse parameter scan to identify universal region candidates
"""

import subprocess
import sys
import json
from datetime import datetime
from itertools import product

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
                        val = part.split('=')[1]
                        try:
                            data['entropy'] = float(val)
                        except:
                            pass
                    elif part.startswith('total='):
                        val = part.split('=')[1]
                        try:
                            data['total'] = float(val)
                        except:
                            pass
                    elif part.startswith('edges='):
                        val = part.split('=')[1]
                        try:
                            data['edges'] = int(val)
                        except:
                            pass
                    elif part.startswith('clusters='):
                        val = part.split('=')[1]
                        try:
                            data['clusters'] = int(val)
                        except:
                            pass
            
            if 'Wilson loops:' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('found'):
                        data['wilson_loops'] = int(parts[i-1])
                    elif part.startswith('avg_mod='):
                        data['wilson_mod'] = float(part.split('=')[1])
                    elif part.startswith('peaks='):
                        data['wilson_peaks'] = int(part.split('=')[1])
            
            if 'Ricci curvature:' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('avg='):
                        data['ricci_avg'] = float(part.split('=')[1])
                    elif part.startswith('std='):
                        data['ricci_std'] = float(part.split('=')[1])
            
            if 'Einstein relation:' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('slope='):
                        data['einstein_slope'] = float(part.split('=')[1])
                    elif part.startswith('R^2='):
                        data['einstein_r2'] = float(part.split('=')[1])
            
            if 'Converged:' in line:
                data['converged'] = 'Yes' in line
        
        return data
    except subprocess.TimeoutExpired:
        return {'error': 'Timeout'}
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 70)
    print("  Phase 1: Small-scale Exploration (N=500)")
    print("=" * 70)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Phase 1 parameters (coarse scan)
    N = 500
    steps = 1000
    
    # Parameter ranges for coarse scan
    eta_values = [0.05, 0.1, 0.15, 0.2]
    lambda_values = [0.8, 1.0, 1.2, 1.5]
    kc_values = [0.05, 0.1, 0.15]
    alpha_values = [0.3, 0.5, 0.8]
    sigma_values = [0.05, 0.1]
    
    total_combinations = len(eta_values) * len(lambda_values) * len(kc_values) * \
                        len(alpha_values) * len(sigma_values)
    
    print(f"Total combinations: {total_combinations}")
    print(f"N = {N}, steps = {steps}")
    print()
    
    results = []
    count = 0
    errors = 0
    
    # Run parameter scan
    for eta, lambda_, kc, alpha, sigma in product(
        eta_values, lambda_values, kc_values, alpha_values, sigma_values
    ):
        count += 1
        print(f"[{count}/{total_combinations}] eta={eta:.2f}, lambda={lambda_:.1f}, "
              f"kc={kc:.2f}, alpha={alpha:.1f}, sigma={sigma:.2f}")
        
        data = run_simulation(N, steps, eta, lambda_, kc, alpha, sigma)
        
        if 'error' not in data:
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
                  f"edges={data['edges']}, clusters={data['clusters']}")
            print(f"  Wilson: loops={data['wilson_loops']}, peaks={data['wilson_peaks']}, "
                  f"mod={data['wilson_mod']:.4f}")
            print(f"  Ricci: {data['ricci_avg']:.4f}, Einstein R²={data['einstein_r2']:.4f}")
        else:
            errors += 1
            print(f"  ERROR: {data['error']}")
        
        print()
    
    # Analysis
    print("=" * 70)
    print("  Phase 1 Results Analysis")
    print("=" * 70)
    print()
    
    # Filter valid results
    valid_results = [r for r in results if r['entropy'] is not None and r['total'] is not None]
    
    print(f"Valid simulations: {len(valid_results)}/{total_combinations}")
    print(f"Errors: {errors}")
    print()
    
    # Find candidates for universal region
    print("=" * 70)
    print("  Universal Region Candidates")
    print("=" * 70)
    print()
    
    # Criteria for universal region:
    # 1. Stable entropy (not NaN, reasonable value)
    # 2. Total information conserved (close to 1.0)
    # 3. Wilson peaks present
    # 4. Reasonable Ricci curvature
    
    candidates = []
    for r in valid_results:
        score = 0
        
        # Entropy stability
        if r['entropy'] and 3.0 < r['entropy'] < 10.0:
            score += 1
        
        # Total conservation
        if r['total'] and abs(r['total'] - 1.0) < 0.01:
            score += 2
        
        # Wilson peaks
        if r['wilson_peaks'] and r['wilson_peaks'] >= 5:
            score += 1
        
        # Ricci curvature reasonable
        if r['ricci_avg'] and 0.5 < r['ricci_avg'] < 2.0:
            score += 1
        
        # Einstein relation
        if r['einstein_r2'] and r['einstein_r2'] > 0.1:
            score += 1
        
        if score >= 4:
            candidates.append((score, r))
    
    # Sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Found {len(candidates)} candidates (score >= 4)")
    print()
    
    if candidates:
        print(f"{'Rank':<5} {'Score':<6} {'eta':<8} {'lambda':<8} {'kc':<8} {'alpha':<8} "
              f"{'entropy':<10} {'total':<10} {'W-peaks':<8} {'Ricci':<8}")
        print("-" * 90)
        
        for i, (score, r) in enumerate(candidates[:15]):
            print(f"{i+1:<5} {score:<6} {r['eta']:<8.2f} {r['lambda']:<8.1f} "
                  f"{r['kc']:<8.2f} {r['alpha']:<8.1f} "
                  f"{r['entropy']:<10.4f} {r['total']:<10.6f} "
                  f"{r['wilson_peaks']:<8} {r['ricci_avg']:<8.4f}")
        
        print()
        
        # Suggested universal region
        print("=" * 70)
        print("  Suggested Universal Region")
        print("=" * 70)
        print()
        
        if len(candidates) >= 3:
            etas = [r['eta'] for _, r in candidates[:10]]
            lambdas = [r['lambda'] for _, r in candidates[:10]]
            kcs = [r['kc'] for _, r in candidates[:10]]
            alphas = [r['alpha'] for _, r in candidates[:10]]
            
            print(f"eta ∈ [{min(etas):.2f}, {max(etas):.2f}]")
            print(f"lambda ∈ [{min(lambdas):.1f}, {max(lambdas):.1f}]")
            print(f"kc ∈ [{min(kcs):.2f}, {max(kcs):.2f}]")
            print(f"alpha ∈ [{min(alphas):.1f}, {max(alphas):.1f}]")
            print()
            
            # Recommended parameters
            best = candidates[0][1]
            print("Recommended parameters for Phase 2:")
            print(f"  eta = {best['eta']:.2f}")
            print(f"  lambda = {best['lambda']:.1f}")
            print(f"  kc = {best['kc']:.2f}")
            print(f"  alpha = {best['alpha']:.1f}")
            print(f"  sigma = {best['sigma']:.2f}")
    
    # Save results
    output_file = f"phase1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'N': N,
            'steps': steps,
            'total_combinations': total_combinations,
            'valid_simulations': len(valid_results),
            'errors': errors,
            'results': results,
            'candidates': [(s, r) for s, r in candidates[:20]]
        }, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")
    print()
    print("=" * 70)
    print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == '__main__':
    main()
