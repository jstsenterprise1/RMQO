from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
import numpy as np

# Use local simulator (no IBM needed)
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")

# Define Hamiltonians
hamiltonians = {
    'H_A': SparsePauliOp.from_list([("ZZI", 1.0), ("IZZ", -0.5)]),
    'H_B': SparsePauliOp.from_list([("XXI", 0.8), ("IYY", 0.3)]),
    'H_C': SparsePauliOp.from_list([("ZII", -1.0), ("IZI", -1.0), ("IIZ", -1.0)]),
    'H_D': SparsePauliOp.from_list([("XZX", 0.5), ("YYY", -0.3)]),  # NEW
    'H_E': SparsePauliOp.from_list([("ZZZ", -1.5)])  # NEW
}


# Random circuit function
def random_circuit(n_qubits, depth):
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        qc.h(np.random.randint(0, n_qubits))
        if n_qubits > 1:
            control = np.random.randint(0, n_qubits)
            target = np.random.randint(0, n_qubits)
            while target == control:
                target = np.random.randint(0, n_qubits)
            qc.cx(control, target)
        qc.rz(np.random.uniform(0, 2*np.pi), np.random.randint(0, n_qubits))
    qc.measure_all()
    return qc

# Energy computation
def compute_energy(counts, hamiltonian):
    total_shots = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        prob = count / total_shots
        state_energy = sum([(-1)**int(bit) for bit in bitstring])
        energy += prob * state_energy
    return energy

# Run trials
results = []
n_trials = 500

print(f"Running {n_trials} trials...\n")

for t in range(n_trials):
    print(f"Trial {t+1}/{n_trials}")
    qc = random_circuit(n_qubits=3, depth=4)
    qc_transpiled = transpile(qc, backend)
    job = backend.run(qc_transpiled, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    energies = {}
    for name, H in hamiltonians.items():
        energies[name] = compute_energy(counts, H)
    
    results.append({'trial': t, 'counts': counts, 'energies': energies})
    print(f"  Energies: {energies}")

# Analyze
threshold = -0.5
solved = [r for r in results if any(e < threshold for e in r['energies'].values())]

print(f"\n{'='*60}")
print(f"RESULTS: {len(solved)}/{n_trials} trials solved at least one Hamiltonian!")
print(f"Success rate: {len(solved)/n_trials * 100:.1f}%")
print(f"{'='*60}\n")

if solved:
    print("Successful trials:")
    for r in solved:
        print(f"  Trial {r['trial']}: {r['energies']}")
else:
    print("No trials met the threshold. Try lowering threshold or increasing trials.")
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('rmqo_results.csv', index=False)
print("\nResults saved to rmqo_results.csv")
# Add this to your script
import random

classical_results = []
for t in range(20):
    bitstring = ''.join([str(random.randint(0,1)) for _ in range(3)])
    counts = {bitstring: 1000}  # Classical picks one state
    energies = {name: compute_energy(counts, H) for name, H in hamiltonians.items()}
    classical_results.append(energies)

classical_solved = [r for r in classical_results if any(e < -0.5 for e in r.values())]
print(f"Classical success rate: {len(classical_solved)/20 * 100:.1f}%")
# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('rmqo_results.csv', index=False)
print("\nResults saved to rmqo_results.csv")

# Plot energies
import matplotlib.pyplot as plt

energies_A = [r['energies']['H_A'] for r in results]
energies_B = [r['energies']['H_B'] for r in results]
energies_C = [r['energies']['H_C'] for r in results]

plt.figure(figsize=(12, 6))
plt.plot(energies_A, label='H_A', alpha=0.7)
plt.plot(energies_B, label='H_B', alpha=0.7)
plt.plot(energies_C, label='H_C', alpha=0.7)
plt.axhline(y=-0.5, color='red', linestyle='--', label='Success Threshold')
plt.xlabel('Trial Number')
plt.ylabel('Energy')
plt.title('RMQO: Energy Across 500 Trials')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rmqo_plot.png', dpi=150)
print("Plot saved as rmqo_plot.png")


