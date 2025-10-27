#!/usr/bin/env python3
# rmqo_advanced.py - WORKING VERSION

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd
from datetime import datetime
import os

os.makedirs('../data/results_advanced', exist_ok=True)

nqubits = 4
ntrials = 50
circuitdepth = 5

backend = AerSimulator()
print(f"Using backend: {backend.name}\n")

def random_circuit(nqubits, depth):
    """Generate random quantum circuit."""
    qc = QuantumCircuit(nqubits)
    for layer in range(depth):
        for q in range(nqubits):
            if np.random.rand() > 0.5:
                qc.h(q)
        
        if nqubits > 1 and np.random.rand() > 0.6:
            for q in range(nqubits - 1):
                if np.random.rand() > 0.5:
                    qc.cx(q, q+1)
        
        for q in range(nqubits):
            qc.rz(np.random.uniform(0, 2*np.pi), q)
    
    qc.measure_all()
    return qc

objectives = {
    'all_zeros': lambda bs: 1.0 if bs == '0'*nqubits else 0.0,
    'all_ones': lambda bs: 1.0 if bs == '1'*nqubits else 0.0,
    'even_parity': lambda bs: 1.0 if bs.count('1') % 2 == 0 else 0.0,
    'odd_parity': lambda bs: 1.0 if bs.count('1') % 2 == 1 else 0.0,
    'alternating': lambda bs: 1.0 if bs in ['0101', '1010', '01010101', '10101010'] else 0.0,
    'majority_ones': lambda bs: 1.0 if bs.count('1') > len(bs) / 2 else 0.0,
    'majority_zeros': lambda bs: 1.0 if bs.count('0') > len(bs) / 2 else 0.0,
    'weight_2': lambda bs: 1.0 if bs.count('1') == 2 else 0.0,
    'weight_3': lambda bs: 1.0 if bs.count('1') == 3 else 0.0,
    'first_qubit_one': lambda bs: 1.0 if bs[0] == '1' else 0.0,
}

results = []
print(f"Running {ntrials} ADVANCED trials with {len(objectives)} objectives...\n")

for trial in range(ntrials):
    if (trial + 1) % 10 == 0:
        print(f"Trial {trial+1}/{ntrials}")
    
    qc = random_circuit(nqubits, circuitdepth)
    qc_transpiled = transpile(qc, backend)
    job = backend.run(qc_transpiled, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    trial_data = {'trial': trial, 'num_states': len(counts)}
    
    for obj_name, obj_func in objectives.items():
        total_success = 0.0
        for bitstring, count in counts.items():
            total_success += obj_func(bitstring) * (count / 1000)
        trial_data[f'obj_{obj_name}'] = total_success
    
    results.append(trial_data)

df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"../data/results_advanced/rmqo_advanced_{timestamp}.csv"
df.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print(f"ADVANCED TEST COMPLETE")
print(f"{'='*70}")
print(f"\nTrials completed: {len(df)}")
print(f"Objectives tested: {len(objectives)}")
print(f"\nObjective Success Rates:")
print(f"{'-'*70}")

for col in df.columns:
    if col.startswith('obj_'):
        obj_name = col.replace('obj_', '')
        success_rate = df[col].mean() * 100
        print(f"  {obj_name:20s}: {success_rate:5.1f}%")

print(f"{'-'*70}")
print(f"Data saved to: {output_file}")
print(f"{'='*70}")
