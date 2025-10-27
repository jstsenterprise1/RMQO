import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

df = pd.read_csv('../data/results_iterative/rmqo_iterative_20251024_063707.csv')

data = []
for idx, row in df.iterrows():
    for obj in ['even_parity', 'all_ones', 'diversity']:
        col_name = f'final_{obj}'
        if col_name in df.columns:
            data.append([row[obj] for obj in ['even_parity', 'all_ones', 'diversity']])

if len(data) < 5:
    print("Not enough data points. Generating synthetic data for demo.")
    np.random.seed(42)
    data = np.random.rand(50, 3)
else:
    data = np.array(data)

print(f"Data shape: {data.shape}")

reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(data)

print(f"Embedding shape: {embedding.shape}")

plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=range(len(embedding)), cmap='viridis', s=50)
plt.colorbar(label='Data Point Index')
plt.title('RMQO Results: UMAP Visualization')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('umap_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to umap_plot.png")
plt.show()
