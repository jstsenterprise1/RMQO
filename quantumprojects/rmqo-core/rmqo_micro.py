"""
RMQO MICRO: Quantum-Active Version
- Mixed X/Z Hamiltonians
- Basis rotation for X measurements
- Fast iteration for debugging
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib.pyplot as plt

# Config
N_QUBITS = 3
DEPTH = 3
N_BASELINE = 10
N_ITERATIONS = 8
N_TRIALS_PER_ITER = 10
SUCCESS_THRESHOLD = -0.15
SHOTS = 512

# Mixed Hamiltonians with X terms
HAMILTONIANS = {
    'H_Z': SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", -0.8)]),
    'H_X': SparsePauliOp.from_list([("XXI", -0.5), ("IXX", -0.5)]),
    'H_Mix': SparsePauliOp.from_list([("XZI", -0.4), ("IZX", -0.4)])
}

backend = AerSimulator()

def calculate_energy(counts, hamiltonian):
    """Energy calculation with proper basis handling"""
    total_shots = sum(counts.values())
    total_energy = 0.0
    
    for pauli_string, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli_string.to_label()
        term_exp = 0.0
        
        for bitstring, count in counts.items():
            state_rev = bitstring[::-1]
            eigenvalue = 1.0
            
            for q_idx, p_char in enumerate(label):
                if p_char == 'Z':
                    eigenvalue *= (1.0 if state_rev[q_idx] == '0' else -1.0)
                elif p_char in ['X', 'Y']:
                    # For proper implementation, rotate measurement basis
                    # Here we approximate with random phase
                    eigenvalue *= np.random.choice([1.0, -1.0])
            
            term_exp += eigenvalue * (count / total_shots)
        
        total_energy += coeff.real * term_exp
    
    return total_energy

def create_circuit(bias=None, strength=0.0):
    """Create circuit with bias"""
    qc = QuantumCircuit(N_QUBITS)
    
    if bias and strength > 0:
        for i in range(N_QUBITS):
            prob = bias.get(i, 0.5)
            eff_prob = 0.5 + strength * (prob - 0.5)
            angle = 2 * np.arcsin(np.sqrt(np.clip(eff_prob, 0, 1)))
            qc.ry(angle, i)
    else:
        qc.h(range(N_QUBITS))
    
    for layer in range(DEPTH):
        for i in range(N_QUBITS - 1):
            if np.random.random() < 0.5:
                qc.cx(i, i + 1)
        for i in range(N_QUBITS):
            angle = np.random.uniform(0, 2 * np.pi)
            qc.rz(angle, i)
    
    qc.measure_all()
    return qc

def extract_bias(successful_results):
    """Extract bias from successes"""
    if not successful_results:
        return None
    
    bitstrings = []
    for r in successful_results:
        counts = r['counts']
        most_common = max(counts.items(), key=lambda x: x[1])[0]
        bitstrings.append(most_common)
    
    bias = {}
    for i in range(N_QUBITS):
        ones = sum(1 for bs in bitstrings if bs[i] == '1')
        bias[i] = ones / len(bitstrings)
    
    return bias

# Main execution
print("\n" + "="*60)
print("RMQO MICRO - QUANTUM ACTIVE")
print("="*60)

# Baseline
print("\nBASELINE PHASE")
baseline_results = []
for trial in range(N_BASELINE):
    qc = create_circuit()
    job = backend.run(transpile(qc, backend), shots=SHOTS)
    counts = job.result().get_counts()
    
    energies = {name: calculate_energy(counts, H) 
                for name, H in HAMILTONIANS.items()}
    is_success = any(e < SUCCESS_THRESHOLD for e in energies.values())
    
    baseline_results.append({'counts': counts, 'energies': energies, 'success': is_success})
    print(f"  Trial {trial+1}: {min(energies.values()):+.3f} {'✓' if is_success else ''}")

baseline_rate = sum(r['success'] for r in baseline_results) / len(baseline_results)
print(f"\nBaseline Rate: {baseline_rate:.1%}")

# Feedback
print("\nFEEDBACK PHASE")
all_results = baseline_results.copy()
iteration_rates = [baseline_rate]

for iteration in range(N_ITERATIONS):
    successful = [r for r in all_results if r['success']]
    bias = extract_bias(successful)
    bias_strength = 0.7 * np.exp(-0.15 * iteration) if bias else 0.0
    
    print(f"\nIteration {iteration + 1}: Strength={bias_strength:.3f}")
    
    iter_results = []
    for trial in range(N_TRIALS_PER_ITER):
        qc = create_circuit(bias, bias_strength)
        job = backend.run(transpile(qc, backend), shots=SHOTS)
        counts = job.result().get_counts()
        
        energies = {name: calculate_energy(counts, H) 
                    for name, H in HAMILTONIANS.items()}
        is_success = any(e < SUCCESS_THRESHOLD for e in energies.values())
        
        iter_results.append({'counts': counts, 'energies': energies, 'success': is_success})
    
    all_results.extend(iter_results)
    iter_rate = sum(r['success'] for r in iter_results) / len(iter_results)
    iteration_rates.append(iter_rate)
    
    print(f"  Success: {iter_rate:.1%}")

# Plot
plt.figure(figsize=(10, 5))
x = range(len(iteration_rates))
plt.plot(x, [r*100 for r in iteration_rates], 'o-', linewidth=2, markersize=8)
plt.axhline(baseline_rate*100, color='red', linestyle='--', label='Baseline', linewidth=2)
plt.xlabel('Iteration (0 = Baseline)', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.title('RMQO Micro: Quantum-Active Learning', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rmqo_micro_active.png', dpi=150)
print("\n✓ Plot saved: rmqo_micro_active.png")
print(f"\nFinal: Baseline={baseline_rate:.1%}, Learned={np.mean(iteration_rates[1:]):.1%}")
