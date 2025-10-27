#!/usr/bin/env python3
"""
RMQO Simple Demo - Shows the Core Concept
This is a simplified version to demonstrate how bias annealing works
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("RMQO SIMPLE DEMO - ANNEALING CONCEPT")
print("="*60)

# Setup
backend = AerSimulator()
nqubits = 3  # Simple: 3 qubits = 8 possible states
max_iterations = 20  # Short run for quick demo
shots = 1000

def create_circuit_with_bias(nqubits, bias_strength):
    """
    Create quantum circuit where bias_strength controls how structured it is.
    bias_strength = 0.0 -> completely random
    bias_strength = 1.0 -> highly structured
    """
    qc = QuantumCircuit(nqubits)
    
    # Apply gates based on bias
    for q in range(nqubits):
        # More bias = more likely to apply Hadamard (superposition)
        if np.random.rand() > (0.3 - bias_strength * 0.2):
            qc.h(q)
    
    # Entanglement (controlled by bias)
    if nqubits > 1 and np.random.rand() > (0.5 - bias_strength * 0.3):
        for q in range(nqubits - 1):
            if np.random.rand() > 0.5:
                qc.cx(q, q+1)
    
    # Rotation gates (angle depends on bias)
    for q in range(nqubits):
        angle = np.pi * bias_strength if bias_strength > 0 else np.random.uniform(0, 2*np.pi)
        qc.rz(angle, q)
    
    qc.measure_all()
    return qc

def compute_energy(counts):
    """
    Simple energy function: we want even parity (even number of 1s).
    Returns probability of measuring states with even parity.
    """
    total = sum(counts.values())
    even_parity_prob = 0.0
    
    for bitstring, count in counts.items():
        if bitstring.count('1') % 2 == 0:  # Even number of 1s
            even_parity_prob += count / total
    
    return even_parity_prob

# Store results
iterations = []
biases = []
energies = []

print("\nRunning annealing process...")
print(f"{'Iteration':<12} {'Bias':<10} {'Energy':<10}")
print("-" * 35)

for iteration in range(max_iterations):
    # Gradually increase bias from 0 to 0.7
    bias = min(iteration / max_iterations, 0.7)
    
    # Create and run circuit
    qc = create_circuit_with_bias(nqubits, bias)
    qc_transpiled = transpile(qc, backend)
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Calculate energy
    energy = compute_energy(counts)
    
    # Store
    iterations.append(iteration)
    biases.append(bias)
    energies.append(energy)
    
    # Print every 5 iterations
    if iteration % 5 == 0:
        print(f"{iteration:<12} {bias:<10.3f} {energy:<10.3f}")

print("-" * 35)
print(f"Final energy: {energies[-1]:.3f}")
print(f"Improvement: {(energies[-1] - energies[0])/energies[0] * 100:.1f}%")

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Energy over iterations
ax1.plot(iterations, energies, 'b-', linewidth=2, label='Energy')
ax1.axhline(y=0.5, color='r', linestyle='--', label='Target (50%)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Energy (Even Parity Probability)')
ax1.set_title('RMQO: Energy Convergence with Annealing')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Bias schedule
ax2.plot(iterations, biases, 'g-', linewidth=2, label='Bias Strength')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Bias Strength')
ax2.set_title('Bias Annealing Schedule')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rmqo_demo.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print("SUCCESS!")
print("Plot saved as: rmqo_demo.png")
print(f"{'='*60}")

# Show plot (will open window)
plt.show()
