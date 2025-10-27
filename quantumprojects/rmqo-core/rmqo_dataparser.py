"""
RMQO Data Parser and Analysis Tool
Parses CSV outputs and generates comprehensive reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path

class RMQODataParser:
    def __init__(self, csv_path):
        """Initialize parser with CSV file"""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.hamiltonians = self._detect_hamiltonians()
        print(f"✓ Loaded {len(self.df)} trials from {Path(csv_path).name}")
        print(f"  Phases: {self.df['phase'].unique().tolist()}")
        print(f"  Hamiltonians: {self.hamiltonians}")
    
    def _detect_hamiltonians(self):
        """Detect which Hamiltonians are in the dataset"""
        first_energy = self.df['energies'].iloc[0]
        try:
            clean = first_energy.replace('np.float64(', '').replace(')', '')
            energy_dict = ast.literal_eval(clean)
            return list(energy_dict.keys())
        except:
            return []
    
    def parse_energies(self, energy_str):
        """Parse energy dictionary from string"""
        try:
            clean = energy_str.replace('np.float64(', '').replace(')', '')
            return ast.literal_eval(clean)
        except:
            return {}
    
    def success_summary(self):
        """Generate success rate summary"""
        print("\n" + "="*70)
        print("SUCCESS RATE SUMMARY")
        print("="*70)
        
        summary = self.df.groupby('phase')['success'].agg([
            ('total_trials', 'count'),
            ('successes', 'sum'),
            ('success_rate', 'mean')
        ])
        
        print(summary)
        print()
        
        for phase in summary.index:
            rate = summary.loc[phase, 'success_rate']
            count = summary.loc[phase, 'successes']
            total = summary.loc[phase, 'total_trials']
            print(f"{phase:15s}: {rate:6.1%}  ({count:3.0f}/{total:3.0f})")
        
        return summary
    
    def energy_distributions(self):
        """Analyze energy distributions per Hamiltonian"""
        print("\n" + "="*70)
        print("ENERGY DISTRIBUTIONS")
        print("="*70)
        
        all_energies = {h: [] for h in self.hamiltonians}
        
        for _, row in self.df.iterrows():
            energies = self.parse_energies(str(row['energies']))
            for h_name, energy in energies.items():
                if h_name in all_energies:
                    all_energies[h_name].append(energy)
        
        stats = {}
        for h_name, values in all_energies.items():
            if values:
                arr = np.array(values)
                stats[h_name] = {
                    'mean': arr.mean(),
                    'std': arr.std(),
                    'min': arr.min(),
                    'max': arr.max(),
                    'median': np.median(arr)
                }
                
                print(f"\n{h_name}:")
                print(f"  Mean:   {stats[h_name]['mean']:+.4f}")
                print(f"  Std:    {stats[h_name]['std']:.4f}")
                print(f"  Min:    {stats[h_name]['min']:+.4f}")
                print(f"  Max:    {stats[h_name]['max']:+.4f}")
                print(f"  Median: {stats[h_name]['median']:+.4f}")
        
        return stats
    
    def iteration_analysis(self):
        """Analyze performance by iteration"""
        print("\n" + "="*70)
        print("ITERATION-BY-ITERATION ANALYSIS")
        print("="*70)
        
        # Filter to only learned and random phases
        feedback_df = self.df[self.df['phase'].isin(['learned', 'random_control'])]
        
        if 'iteration' not in feedback_df.columns:
            print("No iteration data found")
            return None
        
        iter_analysis = feedback_df.groupby(['phase', 'iteration'])['success'].mean().unstack(0)
        print("\nSuccess Rate by Iteration:")
        print(iter_analysis)
        
        return iter_analysis
    
    def visualize_convergence(self, save_path='rmqo_analysis_convergence.png'):
        """Create convergence visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Success rates by phase
        ax = axes[0, 0]
        phase_rates = self.df.groupby('phase')['success'].mean()
        colors = {'baseline': 'coral', 'learned': 'lightgreen', 'random_control': 'gold'}
        bars = ax.bar(range(len(phase_rates)), phase_rates.values, 
                     color=[colors.get(p, 'gray') for p in phase_rates.index])
        ax.set_xticks(range(len(phase_rates)))
        ax.set_xticklabels(phase_rates.index, rotation=45, ha='right')
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate by Phase', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, phase_rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Iteration convergence
        ax = axes[0, 1]
        feedback_df = self.df[self.df['phase'].isin(['learned', 'random_control'])]
        if 'iteration' in feedback_df.columns:
            iter_data = feedback_df.groupby(['phase', 'iteration'])['success'].mean().unstack(0)
            for phase in iter_data.columns:
                ax.plot(iter_data.index, iter_data[phase], 
                       'o-', label=phase, linewidth=2, markersize=6)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Success Rate', fontsize=12)
            ax.set_title('Learning Convergence', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Energy distributions
        ax = axes[1, 0]
        all_energies = {h: [] for h in self.hamiltonians}
        for _, row in self.df.iterrows():
            energies = self.parse_energies(str(row['energies']))
            for h_name, energy in energies.items():
                if h_name in all_energies:
                    all_energies[h_name].append(energy)
        
        positions = range(len(self.hamiltonians))
        bp = ax.boxplot([all_energies[h] for h in self.hamiltonians],
                        positions=positions, labels=self.hamiltonians)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title('Energy Distributions', fontsize=14, fontweight='bold')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Success vs iteration (scatter)
        ax = axes[1, 1]
        if 'iteration' in self.df.columns:
            for phase in ['learned', 'random_control']:
                phase_df = self.df[self.df['phase'] == phase]
                if not phase_df.empty and 'iteration' in phase_df.columns:
                    iter_success = phase_df.groupby('iteration')['success'].agg(['mean', 'std'])
                    ax.errorbar(iter_success.index, iter_success['mean'],
                               yerr=iter_success['std'], fmt='o-',
                               label=phase, linewidth=2, capsize=5)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Success Rate', fontsize=12)
            ax.set_title('Success Rate with Error Bars', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {save_path}")
    
    def generate_report(self, output_path='rmqo_analysis_report.txt'):
        """Generate comprehensive text report"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RMQO DATA ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset: {Path(self.csv_path).name}\n")
            f.write(f"Total Trials: {len(self.df)}\n")
            f.write(f"Phases: {self.df['phase'].unique().tolist()}\n")
            f.write(f"Hamiltonians: {self.hamiltonians}\n\n")
            
            # Success summary
            summary = self.df.groupby('phase')['success'].agg(['count', 'sum', 'mean'])
            f.write("SUCCESS SUMMARY:\n")
            f.write(summary.to_string())
            f.write("\n\n")
            
            # Energy stats
            all_energies = {h: [] for h in self.hamiltonians}
            for _, row in self.df.iterrows():
                energies = self.parse_energies(str(row['energies']))
                for h_name, energy in energies.items():
                    if h_name in all_energies:
                        all_energies[h_name].append(energy)
            
            f.write("ENERGY STATISTICS:\n")
            for h_name, values in all_energies.items():
                if values:
                    arr = np.array(values)
                    f.write(f"\n{h_name}:\n")
                    f.write(f"  Mean: {arr.mean():+.4f}\n")
                    f.write(f"  Std:  {arr.std():.4f}\n")
                    f.write(f"  Range: [{arr.min():+.4f}, {arr.max():+.4f}]\n")
        
        print(f"✓ Report saved: {output_path}")

# ===== MAIN USAGE =====

def analyze_rmqo_data(csv_path):
    """Main analysis function"""
    parser = RMQODataParser(csv_path)
    parser.success_summary()
    parser.energy_distributions()
    parser.iteration_analysis()
    parser.visualize_convergence()
    parser.generate_report()
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    # Example usage - replace with your actual CSV filename
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to most recent file in current directory
        csv_files = sorted(Path('.').glob('complete_data_*.csv'))
        if csv_files:
            csv_file = str(csv_files[-1])
            print(f"Using most recent CSV: {csv_file}")
        else:
            print("No CSV files found. Please specify a file:")
            print("python rmqo_parser.py <path_to_csv>")
            sys.exit(1)
    
    analyze_rmqo_data(csv_file)
