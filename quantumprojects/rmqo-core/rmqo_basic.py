from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd
from datetime import datetime

# CONFIG
nqubits = 3
ntrials = 50  # Start small
circuitdepth = 4

# BACKEND
backend = AerSimulator()
print(f"Using backend: {backend.name}")

def random_circuit(nqubits, depth):
    qc = QuantumCircuit(nqubits)
    for _ in range(depth):
        qc.h(np.random.randint(0, nqubits))
        if nqubits > 1:
            control = np.random.randint(0, nqubits)
            target = np.random.randint(0, nqubits)
            while target == control:
                target = np.random.randint(0, nqubits)
            qc.cx(control, target)
        qc.rz(np.random.uniform(0, 2*np.pi), np.random.randint(0, nqubits))
    qc.measure_all()
    return qc

# RUN TRIALS
results = []
print(f"Running {ntrials} trials...\n")
for t in range(ntrials):
    if (t+1) % 10 == 0:
        print(f"Trial {t+1}/{ntrials}")
    
    qc = random_circuit(nqubits, circuitdepth)
    qc_transpiled = transpile(qc, backend)
    job = backend.run(qc_transpiled, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    results.append({'trial': t, 'counts': str(counts)})

# SAVE
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"../data/results_basic/rmqo_basic_{timestamp}.csv"

df = pd.DataFrame(results)
df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"Success! Results saved to: {output_file}")
print(f"{'='*60}")
