# rmqo_iterative.py - Automated Feedback Loop (FIXED)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import os
import json

# ===== CONFIG =====
output_dir = "rmqo_feedback_study"
os.makedirs(output_dir, exist_ok=True)

n_qubits = 4
n_trials_per_iteration = 100
max_iterations = 10
threshold = -0.5

# ===== HAMILTONIANS =====
hamiltonians = {
    'H_AFM': SparsePauliOp.from_list([("ZZII", 1.0), ("IIZZ", 1.0), ("ZIZI", -1.0)]),
    'H_FM': SparsePauliOp.from_list([("ZZII", -1.0), ("IIZZ", -1.0), ("ZIZI", -1.0)]),
    'H_TF': SparsePauliOp.from_list([("XIII", -0.5), ("IXII", -0.5), ("IIXI", -0.5), ("IIIX", -0.5)]),
    'H_XY': SparsePauliOp.from_list([("XXII", 0.5), ("YYII", 0.5), ("IIXX", 0.5), ("IIYY", 0.5)]),
    'H_RF': SparsePauliOp.from_list([("ZIII", -0.7), ("IZII", 0.3), ("IIZI", -0.9), ("IIIZ", 0.5)])
}

backend = AerSimulator()

# ===== FUNCTIONS =====
def random_circuit(n_qubits, depth, bit_prefs=None, bias_strength=0.0):
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        for qubit in range(n_qubits):
            if bit_prefs and np.random.rand() < bias_strength * bit_prefs.get(qubit, {}).get('1', 0.5):
                qc.rx(np.pi / 4, qubit)
            else:
                qc.h(qubit)
        
        if np.random.rand() < 0.6:
            control = np.random.randint(0, n_qubits)
            target = np.random.randint(0, n_qubits)
            while target == control:
                target = np.random.randint(0, n_qubits)
            qc.cx(control, target)
        
        qc.rz(np.random.uniform(0, 2*np.pi), np.random.randint(0, n_qubits))
    qc.measure_all()
    return qc

def compute_energy(counts, hamiltonian):
    total_shots = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        prob = count / total_shots
        state_energy = sum([(-1)**int(bit) for bit in bitstring])
        energy += prob * state_energy
    return energy

def extract_patterns(results):
    """Extract bit preferences from successful trials"""
    successful_bitstrings = []
    for r in results:
        if any(e < threshold for e in r['energies'].values()):
            counts = r['counts']
            most_common = max(counts.items(), key=lambda x: x[1])[0]
            successful_bitstrings.append(most_common)
    
    if not successful_bitstrings:
        return None
    
    bit_prefs = {}
    for pos in range(len(successful_bitstrings[0])):
        bit_prefs[pos] = {'0': 0, '1': 0}
        for bitstring in successful_bitstrings:
            bit_prefs[pos][bitstring[pos]] += 1
    
    # Normalize
    for pos in bit_prefs:
        total = bit_prefs[pos]['0'] + bit_prefs[pos]['1']
        if total > 0:
            bit_prefs[pos]['1'] /= total
            bit_prefs[pos]['0'] /= total
    
    return bit_prefs, successful_bitstrings

# ===== MAIN LOOP =====
all_results = []
success_rates = []
bit_prefs = None

print(f"{'='*70}")
print(f"RMQO ITERATIVE FEEDBACK LOOP WITH ANNEALING")
print(f"{'='*70}\n")

for iteration in range(max_iterations):
    # ANNEALING BIAS DECAY - Define ONCE and use throughout
    bias_strength = 0.7 * np.exp(-0.2 * iteration)
    
    print(f"ITERATION {iteration + 1}/{max_iterations}")
    print(f"Bias Strength (annealed): {bias_strength:.3f}")
    
    iteration_results = []
    
    for t in range(n_trials_per_iteration):
        if (t+1) % 25 == 0:
            print(f"  Trial {t+1}/{n_trials_per_iteration}")
        
        # Use the ANNEALED bias_strength (do NOT override it)
        qc = random_circuit(n_qubits, depth=6, bit_prefs=bit_prefs, bias_strength=bias_strength)
        qc_transpiled = transpile(qc, backend)
        
        job = backend.run(qc_transpiled, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        energies = {name: compute_energy(counts, H) for name, H in hamiltonians.items()}
        iteration_results.append({'trial': t, 'counts': counts, 'energies': energies, 'iteration': iteration})
    
    all_results.extend(iteration_results)
    
    # Analyze this iteration
    df_iter = pd.DataFrame(iteration_results)
    df_iter['energies_dict'] = df_iter['energies'].apply(lambda x: x)
    df_iter['solved'] = df_iter['energies_dict'].apply(lambda x: any(e < threshold for e in x.values()))
    
    success_rate = df_iter['solved'].sum() / len(df_iter)
    success_rates.append(success_rate)
    
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Successful Trials: {df_iter['solved'].sum()}/{len(df_iter)}\n")
    
    # Extract patterns for next iteration
    pattern_result = extract_patterns(iteration_results)
    if pattern_result:
        bit_prefs, successful_bitstrings = pattern_result
        print(f"  Patterns Extracted: {len(set(successful_bitstrings))} unique solutions")
        print(f"  Qubit 1 preference: |1⟩={bit_prefs[1]['1']:.1%}")
        print(f"  Qubit 2 preference: |1⟩={bit_prefs[2]['1']:.1%}\n")
    
    # Save iteration results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iter_csv = f"{output_dir}/iteration_{iteration+1:02d}_{timestamp}.csv"
    df_iter.to_csv(iter_csv, index=False)
    print(f"  Saved: {iter_csv}\n")

# ===== FINAL ANALYSIS =====
print(f"\n{'='*70}")
print(f"FINAL RESULTS")
print(f"{'='*70}\n")

for i, rate in enumerate(success_rates):
    print(f"Iteration {i+1}: {rate*100:.1f}%")

print(f"\nInitial (random): {success_rates[0]*100:.1f}%")
if len(success_rates) > 1:
    final = success_rates[-1]
    improvement = (final - success_rates[0]) / success_rates[0] * 100
    print(f"Final (after {max_iterations-1} feedback loops): {final*100:.1f}%")
    print(f"Improvement: +{improvement:.0f}%")

# Plot convergence
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(success_rates)+1), [r*100 for r in success_rates], 'o-', linewidth=2, markersize=8)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.title('RMQO Feedback Loop Convergence with Annealing', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_file = f"{output_dir}/convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(plot_file, dpi=150)
print(f"\nConvergence plot saved: {plot_file}")

# Save summary
summary_file = f"{output_dir}/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(summary_file, 'w') as f:
    f.write(f"RMQO ITERATIVE FEEDBACK LOOP STUDY\n")
    f.write(f"{'='*70}\n\n")
    f.write(f"Iterations: {max_iterations}\n")
    f.write(f"Trials per iteration: {n_trials_per_iteration}\n")
    f.write(f"Total trials: {len(all_results)}\n\n")
    f.write(f"Success Rates by Iteration:\n")
    for i, rate in enumerate(success_rates):
        f.write(f"  Iteration {i+1}: {rate*100:.1f}%\n")
    if len(success_rates) > 1:
        f.write(f"\nImprovement: +{improvement:.0f}%\n")

print(f"Summary saved: {summary_file}")
