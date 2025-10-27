# rmqo_feedback_loop.py - Retrocausal Feedback

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import pandas as pd

# ===== PREVIOUS RESULTS (from parser) =====
successful_patterns = {
    '1111': 7,
    '0110': 6,
    '0111': 5,
    '1011': 5,
    '1110': 5,
    '1101': 4,
    '1010': 3,
    '0101': 3,
    '1001': 2,
    '0100': 1
}

# Pattern preferences (from your parser output)
bit_prefs = {
    0: {'0': 0.378, '1': 0.622},  # Position 0 prefers |1⟩
    1: {'0': 0.289, '1': 0.711},  # Position 1 prefers |1⟩ (STRONG)
    2: {'0': 0.289, '1': 0.711},  # Position 2 prefers |1⟩ (STRONG)
    3: {'0': 0.378, '1': 0.622}   # Position 3 prefers |1⟩
}

# ===== CONFIG =====
n_qubits = 4
n_trials = 100
threshold = -0.5
backend = AerSimulator()

# ===== BIASED CIRCUIT GENERATION =====
def biased_circuit(n_qubits, depth, bit_prefs, bias_strength=0.8):
    """
    bias_strength: 0.0 = random, 1.0 = 100% follow preferences
    """
    qc = QuantumCircuit(n_qubits)
    
    for _ in range(depth):
        # Initialize gates biased toward successful patterns
        for qubit in range(n_qubits):
            pref_1 = bit_prefs.get(qubit, {}).get('1', 0.5)
            
            # If this qubit prefers |1⟩, bias H gate application
            if np.random.rand() < bias_strength * pref_1:
                # Apply RX(π/4) to bias toward |1⟩
                qc.rx(np.pi / 4, qubit)
            else:
                qc.h(qubit)
        
        # Entanglement (biased toward successful pairs)
        if np.random.rand() < 0.6:  # More frequent CNOT
            # Prefer qubits that correlate in successful trials
            control = np.random.choice([1, 2], p=[0.5, 0.5])  # Qubits 1,2 are key
            target = np.random.choice([q for q in range(n_qubits) if q != control])
            qc.cx(control, target)
        
        # Rotation
        qc.rz(np.random.uniform(0, 2*np.pi), np.random.randint(0, n_qubits))
    
    qc.measure_all()
    return qc

# ===== DEFINE HAMILTONIANS (same as before) =====
hamiltonians = {
    'H_AFM': SparsePauliOp.from_list([("ZZII", 1.0), ("IIZZ", 1.0), ("ZIZI", -1.0)]),
    'H_FM': SparsePauliOp.from_list([("ZZII", -1.0), ("IIZZ", -1.0), ("ZIZI", -1.0)]),
    'H_TF': SparsePauliOp.from_list([("XIII", -0.5), ("IXII", -0.5), ("IIXI", -0.5), ("IIIX", -0.5)]),
    'H_XY': SparsePauliOp.from_list([("XXII", 0.5), ("YYII", 0.5), ("IIXX", 0.5), ("IIYY", 0.5)]),
    'H_RF': SparsePauliOp.from_list([("ZIII", -0.7), ("IZII", 0.3), ("IIZI", -0.9), ("IIIZ", 0.5)])
}

# ===== ENERGY COMPUTATION =====
def compute_energy(counts, hamiltonian):
    total_shots = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        prob = count / total_shots
        state_energy = sum([(-1)**int(bit) for bit in bitstring])
        energy += prob * state_energy
    return energy

# ===== RUN BIASED TRIALS =====
print(f"Running {n_trials} BIASED trials (using successful patterns)...\n")

biased_results = []
for t in range(n_trials):
    if (t+1) % 20 == 0:
        print(f"Trial {t+1}/{n_trials}")
    
    # Use biased circuit generation
    qc = biased_circuit(n_qubits, depth=6, bit_prefs=bit_prefs, bias_strength=0.7)
    qc_transpiled = transpile(qc, backend)
    
    job = backend.run(qc_transpiled, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    energies = {name: compute_energy(counts, H) for name, H in hamiltonians.items()}
    biased_results.append({'trial': t, 'counts': counts, 'energies': energies})

# ===== COMPARE RESULTS =====
biased_solved = [r for r in biased_results if any(e < threshold for e in r['energies'].values())]
biased_success_rate = len(biased_solved) / n_trials

print(f"\n{'='*60}")
print(f"BIASED TRIALS (100):")
print(f"  Success rate: {biased_success_rate * 100:.1f}%")
print(f"  Successful trials: {len(biased_solved)}/100")
print(f"\nCOMPARISON:")
print(f"  Random (baseline): ~9%")
print(f"  Biased (with parser): {biased_success_rate * 100:.1f}%")
if biased_success_rate > 0.09:
    improvement = (biased_success_rate - 0.09) / 0.09 * 100
    print(f"  IMPROVEMENT: +{improvement:.0f}%")
print(f"{'='*60}\n")

# Save results
df = pd.DataFrame(biased_results)
df.to_csv('rmqo_biased_results.csv', index=False)
print("Results saved to rmqo_biased_results.csv")

# Show successful bitstrings
successful_bitstrings = []
for r in biased_results:
    if any(e < threshold for e in r['energies'].values()):
        counts = r['counts']
        most_common = max(counts.items(), key=lambda x: x[1])[0]
        successful_bitstrings.append(most_common)

from collections import Counter
print("\nTop successful bitstrings (biased run):")
for bitstring, count in Counter(successful_bitstrings).most_common(5):
    print(f"  {bitstring}: {count} times")
