# UPDATED rmqo_iterative.py with ANNEALING BIAS DECAY

# ===== MAIN LOOP (REPLACE THIS SECTION) =====
all_results = []
success_rates = []
bit_prefs = None

print(f"{'='*70}")
print(f"RMQO ITERATIVE FEEDBACK LOOP WITH ANNEALING")
print(f"{'='*70}\n")

for iteration in range(max_iterations):
    print(f"ITERATION {iteration + 1}/{max_iterations}")
    
    # ANNEALING: Bias strength DECAYS over iterations
    # Start high (0.7) → End low (0.3) to consolidate solutions
    bias_strength = 0.7 * np.exp(-0.2 * iteration)
    
    print(f"Bias Strength (annealed): {bias_strength:.3f}")
    print(f"Bias Strength Formula: 0.7 * exp(-0.2 * {iteration}) = {bias_strength:.3f}")
    
    iteration_results = []
    
    for t in range(n_trials_per_iteration):
        if (t+1) % 25 == 0:
            print(f"  Trial {t+1}/{n_trials_per_iteration}")
        
        # Generate circuit with ANNEALED bias strength
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
    print(f"  Bias Strength Decay Schedule:")
    print(f"    Iteration 1: {0.7 * np.exp(-0.2 * 0):.3f} (high - explore)")
    print(f"    Iteration 5: {0.7 * np.exp(-0.2 * 4):.3f} (medium)")
    print(f"    Iteration 10: {0.7 * np.exp(-0.2 * 9):.3f} (low - consolidate)\n")

# Rest of analysis remains the same...