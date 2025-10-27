# RMQO: Retrocausal Multi-Target Quantum Optimization

**A self-guiding paradigm for emergent solution discovery through retrocausal feedback loops, iterative annealing, and attractor-based dynamics.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org)

---

## Overview

RMQO (Retrocausal Multi-Target Quantum Optimization) is an experimental quantum algorithm framework designed to explore **non-Markovian information recurrence** and **temporal symmetry** in adaptive quantum systems. 

The algorithm introduces:
- **Retrocausal feedback loops** that bias future trials based on parsed historical data
- **Multi-Hamiltonian optimization** across competing energy landscapes
- **Emergent attractor dynamics** where convergence patterns exhibit cross-iteration correlation

RMQO operationalizes concepts from the **Transactional Interpretation** of quantum mechanics and explores whether quantum optimization algorithms can encode temporal bidirectionality into their solution-search mechanisms.

---

## Key Features

- **Iterative Convergence Tracking**: Monitor energy minimization across multiple Hamiltonians over successive quantum annealing cycles
- **Retrocausal Biasing**: Parse historical trial data and apply adaptive parameter shifts to subsequent runs
- **Non-Markovian Correlation Testing**: Quantify whether iteration \( n \) shows statistically significant alignment with iteration \( n-1 \) beyond random drift
- **Modular Architecture**: Easy-to-extend framework for custom Hamiltonians, backend simulators, and real quantum hardware

---

## Installation

### Prerequisites
- Python 3.8+
- Qiskit 0.43+
- NumPy, Matplotlib

### Setup

Clone the repository:
```bash
git clone https://github.com/jstsenterprise1/RMQO.git
cd RMQO
```

Create a virtual environment and install dependencies:
```bash
python3 -m venv rmqo_env
source rmqo_env/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### Run a basic RMQO simulation:
```bash
python src/rmqo_core/rmqo_iterative.py
```

This executes 10 iterations of multi-Hamiltonian optimization with automatic convergence tracking and CSV export.

### Customize Hamiltonians:
Edit `src/utils/hamiltonian_generators.py` to define custom target states or energy functions.

### Visualize Results:
```bash
python src/analysis/umap_visualization.py
```

Generates convergence plots and dimensionality-reduced attractor maps from your trial data.

---

## Repository Structure

```
RMQO/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── citation.cff               # Citation metadata
│
├── src/                       # Core algorithm implementations
│   ├── rmqo_core/
│   ├── analysis/
│   └── utils/
│
├── data/                      # Experimental output datasets
├── results/                   # Plots and visualizations
├── docs/                      # Research papers and documentation
└── tests/                     # Unit tests
```

---

## Data & Results

All experimental outputs (CSV files, convergence plots, summary statistics) are stored in:
- `data/` — Raw trial results
- `results/` — Processed visualizations and graphs

### Sample Output
Each run generates:
- `iteration_XX_YYYYMMDD_HHMMSS.csv` — Per-iteration energy measurements
- `summary_YYYYMMDD_HHMMSS.txt` — Final statistics (mean, variance, convergence rate)
- `convergence_YYYYMMDD_HHMMSS.png` — Iteration-to-iteration energy plots

---

## Research & Publications

### Preprints
- **RMQO WhitePaper** *(arXiv pending)*  
  *"Retrocausal Multi-Target Quantum Optimization: A Self-Guiding Paradigm for Emergent Solution Discovery"*

### Related Documentation
- [Retrocausal Theory Foundation](docs/Retrocausal_Theory_Foundation.md)
- [Standard Model Mapping](docs/Standard_Model_Mapping.md)
- [Anomalous Intuition Analysis](docs/Anomalous_Intuition_Analysis.pdf)

---

## Citation

If you use RMQO in your research, please cite:

```bibtex
@software{rmqo2025,
  author = {Oestreich, Jacob},
  title = {RMQO: Retrocausal Multi-Target Quantum Optimization},
  year = {2025},
  url = {https://github.com/jstsenterprise1/RMQO},
  version = {1.0}
}
```

You can also use GitHub's **"Cite this repository"** button at the top of this page.

---

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -m "Add new analysis module"`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Contact

**Jacob Oestreich**  
JSTS Enterprise Quantum Group  
Email: jacob.o@jstsenterpriseinc.com  
GitHub: [@jstsenterprise1](https://github.com/jstsenterprise1)

---

## Acknowledgements

This research was developed independently with theoretical grounding in:
- Transactional Interpretation of Quantum Mechanics (Cramer, 1986)
- Retrocausality and Time-Symmetric Quantum Mechanics (Aharonov, Tollaksen)
- Quantum Annealing and Optimization (D-Wave, IBM Quantum)