# rmqo_parser.py - Extract and amplify successful patterns

import pandas as pd
import numpy as np
from collections import Counter

# Load results
df = pd.read_csv('rmqo_advanced_results.csv')

# Parse successful bitstrings
threshold = -0.5
successful_bitstrings = []

for idx, row in df.iterrows():
    energies = eval(row['energies'])
    if any(e < threshold for e in energies.values()):
        counts = eval(row['counts'])
        # Get most probable bitstring
        most_common = max(counts.items(), key=lambda x: x[1])[0]
        successful_bitstrings.append(most_common)

print(f"Found {len(successful_bitstrings)} successful bitstrings\n")

# Analyze patterns
print("=== Pattern Analysis ===\n")

# 1. Bit frequency analysis
bit_positions = {}
for bitstring in successful_bitstrings:
    for i, bit in enumerate(bitstring):
        if i not in bit_positions:
            bit_positions[i] = {'0': 0, '1': 0}
        bit_positions[i][bit] += 1

print("Bit Position Preferences (successful states):")
for pos in sorted(bit_positions.keys()):
    total = bit_positions[pos]['0'] + bit_positions[pos]['1']
    pref_0 = bit_positions[pos]['0'] / total
    pref_1 = bit_positions[pos]['1'] / total
    print(f"  Position {pos}: |0⟩={pref_0:.1%}, |1⟩={pref_1:.1%}")

# 2. Pairwise correlations
print("\n=== Pairwise Correlations (which bits align) ===\n")
pairs = {}
for bitstring in successful_bitstrings:
    for i in range(len(bitstring)):
        for j in range(i+1, len(bitstring)):
            pair = (i, j, bitstring[i], bitstring[j])
            pairs[pair] = pairs.get(pair, 0) + 1

# Top correlated pairs
top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
for (i, j, bi, bj), count in top_pairs:
    prob = count / len(successful_bitstrings)
    print(f"  Qubits {i},{j}: |{bi}{bj}⟩ appears {count}x ({prob:.1%})")

# 3. Entropy (how "structured" are solutions?)
bitstring_counts = Counter(successful_bitstrings)
entropy = -sum((c/len(successful_bitstrings)) * np.log2(c/len(successful_bitstrings)) 
               for c in bitstring_counts.values())
max_entropy = np.log2(len(bitstring_counts))
print(f"\n=== Solution Diversity ===")
print(f"  Unique solutions: {len(bitstring_counts)}")
print(f"  Entropy: {entropy:.2f} bits (max={max_entropy:.2f})")
print(f"  Structure: {(1 - entropy/max_entropy)*100:.1f}% (higher = more ordered)")

print("\n=== Top 10 Recurring Successful Bitstrings ===")
for bitstring, count in bitstring_counts.most_common(10):
    print(f"  {bitstring}: {count} times")

print("\n" + "="*60)
print("KEY INSIGHT: If solutions are REPEATING across trials")
print("without being trained for them = RETROCAUSAL EMERGENCE")
print("="*60)
