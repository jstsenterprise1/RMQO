from qiskit import QuantumCircuit
from qiskit_optimization.applications import MaxCut
from scipy.optimize import minimize

# 1. Define Max-Cut problem
# Graph: 6 vertices, 8 random edges (weighted)
graph = {
    'num_nodes': 6,
    'edges': [(0,1,1.5), (0,2,2.1), (1,3,1.2), 
              (2,4,1.8), (3,5,0.9), ...] 
}

# 2. Create cost Hamiltonian from edges
maxcut = MaxCut(graph)
qp = maxcut.to_quadratic_program()

# 3. QAOA ansatz (standard layers)
def qaoa_ansatz(theta_gamma, theta_beta, p=1):
    # theta_gamma: cost phase, theta_beta: mixer phase
    # Returns expectation value of cost Hamiltonian
    ...

# 4. Optimize classically
result_gd = minimize(qaoa_ansatz, x0=init_params, 
                     method='BFGS', options={'maxiter': 500})

result_spsa = minimize(qaoa_ansatz, x0=init_params,
                       method='SPSA', options={'maxiter': 500})

# METRIC: Number of function evaluations to reach 95% of optimal
queries_gd = result_gd.nfev
queries_spsa = result_spsa.nfev
