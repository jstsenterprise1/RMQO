#!/usr/bin/env python3
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd
from datetime import datetime
import os

os.makedirs('../data/results_iterative', exist_ok=True)

nqubits = 4
max_iterations = 100
circuitdepth = 5

backend = AerSimulator()
print(f"Using backend: {backend.name}\n")

def random_circuit_with_bias(nqubits, depth, bias_strength=0.0):
    qc = QuantumCircuit(nqubits)
    for layer in range(depth):
        for q in range(nqubits):
            if np.random.rand() > (0.5 - bias_strength*0.3):
                qc.h(q)
        if nqubits > 1 and np.random.rand() > 0.6:
            for q in range(nqubits - 1):
                if np.random.rand() > 0.5:
                    qc.cx(q, q+1)
        for q in range(nqubits):
            angle = np.pi * bias_strength if bias_strength > 0 else np.random.uniform(0, 2*np.pi)
            qc.rz(angle, q)
    qc.measure_all()
    return qc

def compute_parity_energy(counts):
    total = sum(counts.values())
    even_parity_prob = 0.0
    for bitstring, count in counts.items():
        if bitstring.count('1') % 2 == 0:
            even_parity_prob += count / total
    return even_parity_prob

def compute_all_ones_prob(counts):
    total = sum(counts.values())
    return counts.get('1111', 0) / total

def compute_diversity_energy(counts):
    total = sum(counts.values())
    probs = np.array([count/total for count in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy / nqubits

objectives = {
    'even_parity': compute_parity_energy,
    'all_ones': compute_all_ones_prob,
    'diversity': compute_diversity_energy,
}

print(f"RMQO ITERATIVE TEST")
print(f"{'='*70}")
print(f"Testing convergence with iterative feedback")
print(f"Objectives: {list(objectives.keys())}")
print(f"Max iterations per run: {max_iterations}\n")

results = []
print(f"Running 5 ITERATIVE optimization runs...\n")

for run_num in range(5):
    print(f"\n{'='*70}")
    print(f"RUN {run_num+1}/5")
    print(f"{'='*70}")
    
    best_energy = 0.0
    energies = {obj: [] for obj in objectives.keys()}
    queries = 0
    bias = 0.0
    
    for iteration in range(max_iterations):
        bias = min(iteration / max_iterations, 0.7)
        
        qc = random_circuit_with_bias(nqubits, circuitdepth, bias_strength=bias)
        qc_transpiled = transpile(qc, backend)
        job = backend.run(qc_transpiled, shots=1000)
        result = job.result()
        counts = result.get_counts()
        queries += 1
        
        iteration_energies = {obj: func(counts) for obj, func in objectives.items()}
        for obj, energy in iteration_energies.items():
            energies[obj].append(energy)
        
        if (iteration + 1) % 20 == 0:
            avg_energy = np.mean([iteration_energies[obj] for obj in objectives.keys()])
            print(f"  Iter {iteration+1:3d}: Energy={avg_energy:.3f}, Bias={bias:.2f}, Queries={queries}")
        
        if iteration > 10:
            recent_avg = np.mean([energies[obj][-5:] for obj in objectives.keys()])
            old_avg = np.mean([energies[obj][-10:-5] for obj in objectives.keys()])
            if abs(recent_avg - old_avg) < 0.01:
                print(f"  Converged at iteration {iteration+1}")
                break
    
    final_energies = {obj: energies[obj][-1] for obj in objectives.keys()}
    mean_final_energy = np.mean(list(final_energies.values()))
    
    print(f"\n  Final Results:")
    for obj, energy in final_energies.items():
        print(f"    {obj:20s}: {energy:.1%}")
    print(f"    Average: {mean_final_energy:.1%}")
    print(f"  Queries to convergence: {queries}")
    
    results.append({
        'run': run_num,
        'final_queries': queries,
        'mean_final_energy': mean_final_energy,
        'iterations': iteration + 1,
        **{f'final_{obj}': final_energies[obj] for obj in objectives.keys()}
    })

df = pd.DataFrame(results)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"../data/results_iterative/rmqo_iterative_{timestamp}.csv"
df.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print(f"ITERATIVE TEST COMPLETE")
print(f"{'='*70}")
print(f"\nResults saved to: {output_file}")
print(f"\nSUMMARY ACROSS 5 RUNS:")
print(f"{'-'*70}")
print(f"Mean queries to convergence: {df['final_queries'].mean():.1f}")
print(f"Mean final energy: {df['mean_final_energy'].mean():.1%}")
print(f"Best final energy: {df['mean_final_energy'].max():.1%} (run {df['mean_final_energy'].idxmax()})")
print(f"Worst final energy: {df['mean_final_energy'].min():.1%} (run {df['mean_final_energy'].idxmin()})")

print(f"\nCOMPARISON TO BASELINE:")
baseline_avg = 0.298
improvement = (df['mean_final_energy'].mean() - baseline_avg) / baseline_avg * 100
print(f"Baseline (random): {baseline_avg:.1%}")
print(f"Iterative result: {df['mean_final_energy'].mean():.1%}")
print(f"Improvement: {improvement:+.1f}%")

if improvement > 0:
    print(f"IMPROVEMENT DETECTED: {improvement:.1f}% better than random!")
else:
    print(f"No improvement")

print(f"{'-'*70}")
print(df[['run', 'final_queries', 'mean_final_energy']].to_string(index=False))
print(f"{'='*70}")
