import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
import argparse
import re

# ACM-compatible font settings
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



# Colors and labels from existing scripts
custom_colors = {
    "random": "gray",
    "randomwt": "#2c89d9",
    "oort": "#E69F00",
    "oortwt": "#7f7fff",
    "oortca": '#E69F00',
    "oortcawt": "cornflowerblue"
}

label_mapping = {
    "random": "Random",
    "randomwt": "RandomWT",
    "oort": "Oort",
    "oortwt": "OortWT",
    "oortca": "OortCA",
    "oortcawt": "OortCAWT"
}

def extract_method_and_budget(filename):
    """Extract method name and budget percentage from filename"""
    # Handle patterns like "OortCA 30%" or "Oort 100%"
    match = re.search(r'([A-Za-z]+)\s+(\d+)%', filename)
    if match:
        method = match.group(1).lower()
        budget = int(match.group(2))
        return method, budget
    return None, None

def load_final_results(folder_path):
    """
    Load final accuracies and carbon used from pkl files in folder.
    Returns dictionary with (method, budget) as keys and (final_accuracy, carbon_used) as values.
    """
    files = os.listdir(folder_path)
    results = {}
    
    # Get accuracy files
    accuracy_files = [f for f in files if not any(exclude in f for exclude in 
                     ['_carbon_used', '_intensity_matrix', '_selected_clients', '_cumulative_usage'])]
    
    # Get carbon used files
    carbon_files = [f for f in files if '_carbon_used.pkl' in f]
    
    for acc_file in accuracy_files:
        if not acc_file.endswith('.pkl'):
            continue
            
        method, budget = extract_method_and_budget(acc_file)
        if method is None or budget is None:
            continue
            
        try:
            # Load accuracy data
            with open(os.path.join(folder_path, acc_file), 'rb') as f:
                history = pickle.load(f)
                
            final_accuracy = 0.0
            if hasattr(history, "metrics_centralized") and "accuracy" in history.metrics_centralized:
                accuracy_data = history.metrics_centralized["accuracy"]
                if accuracy_data:
                    final_accuracy = accuracy_data[-1][1] * 100  # Convert to percentage
            
            # Find matching carbon file
            carbon_used = 0.0
            for carbon_file in carbon_files:
                # More precise matching: extract method and budget from carbon filename
                carbon_method, carbon_budget = extract_method_and_budget(carbon_file)
                if (carbon_method and carbon_budget is not None and 
                    carbon_method.lower() == method.lower() and 
                    carbon_budget == budget):
                    try:
                        with open(os.path.join(folder_path, carbon_file), 'rb') as f:
                            carbon_data = pickle.load(f)
                            if isinstance(carbon_data, (int, float)):
                                carbon_used = carbon_data
                            elif hasattr(carbon_data, '__len__') and len(carbon_data) > 0:
                                carbon_used = carbon_data[-1]
                        print(f"[DEBUG] Found carbon file: {carbon_file} -> {carbon_used}")
                        break
                    except Exception as e:
                        print(f"[WARN] Error reading carbon file {carbon_file}: {e}")
            
            results[(method, budget)] = (final_accuracy, carbon_used)
            print(f"[INFO] {method.capitalize()} {budget}%: Accuracy={final_accuracy:.1f}%, Carbon={carbon_used:.0f} gCO2eq")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {acc_file}: {e}")
    
    return results



def plot_results_combined(folder_results, folder_names, output_name="budget_accuracy_combined"):
    """
    Plot simple bars for carbon and dots/lines for accuracy in 2x2 layout.
    First row: first dataset, Second row: second dataset.
    """
    num_folders = len(folder_results)
    
    # Check if we have 4 folders for 2x2 layout
    if num_folders == 4:
        fig, axes = plt.subplots(2, 2, figsize=(26, 10), sharey=True)
        dataset_mode = True
    else:
        # Fallback to side-by-side if not 4 folders
        fig, axes = plt.subplots(1, num_folders, figsize=(8 * num_folders, 6), sharey=True)
        if num_folders == 1:
            axes = [axes]
        dataset_mode = False
    
    for folder_idx, (results, folder_name) in enumerate(zip(folder_results, folder_names)):
        if dataset_mode:
            # 2x2 layout
            row = folder_idx // 2
            col = folder_idx % 2
            ax = axes[row, col]
        else:
            # Side-by-side layout
            ax = axes[folder_idx]
            
        ax2 = ax.twinx()  # Second y-axis for carbon
        
        # Collect all methods and sort by budget (0%, 30%, 100%)
        method_list = []
        for (method, budget), (accuracy, carbon) in results.items():
            method_list.append((budget, method, accuracy, carbon))
        
        # Sort by budget percentage
        method_list.sort(key=lambda x: x[0])  # Sort by budget
        
        # Prepare data for plotting
        methods = []
        accuracies = []
        carbons = []
        colors = []
        
        for budget, method, accuracy, carbon in method_list:
            # Create method label with sentence case
            if budget == 0:
                if method.lower() == 'oortcawt':
                    label = f"OortCAWT 0%"
                else:
                    label = f"OortCA 0%"
            elif budget == 30:
                if method.lower() == 'oortcawt':
                    label = f"OortCAWT 30%"
                else:
                    label = f"OortCA 30%"
            elif budget == 100:
                label = f"Oort"  # Always "Oort" for 100%
            else:
                if method.lower() == 'oortcawt':
                    label = f"OortCAWT {budget}%"
                else:
                    label = f"OortCA {budget}%"
            
            methods.append(label)
            accuracies.append(accuracy)
            carbons.append(carbon / 1000)  # Convert to kg
            
            # Get color based on method
            method_key = method.lower()
            color = custom_colors.get(method_key, '#E69F00')
            colors.append(color)
        
        # Plot carbon bars with gray color (following plot_budgets.py style)
        x_pos = range(len(methods))
        
        # All emission bars are darker gray with transparency (like plot_budgets.py)
        bars = ax2.bar(x_pos, carbons, color='#555555', alpha=0.4, width=0.6)
        
        # Plot accuracy line with dots using condition-based color
        if dataset_mode:
            # For 2x2 layout: assign colors based on Clean/Noisy for accuracy lines only
            if "Clean" in folder_name:
                line_color = '#E69F00'  # Yellow/orange (from plot_budgets oort color)
            else:  # Noisy
                line_color = 'cornflowerblue'  # Blue (from plot_budgets oortcawt color)
        else:
            # Fallback for side-by-side
            line_color = 'cornflowerblue'
        
        line = ax.plot(x_pos, accuracies, 'o-', color=line_color, 
                      linewidth=3, markersize=8, label='Max Accuracy')
        
        # Add accuracy values below each dot
        for i, (x, acc) in enumerate(zip(x_pos, accuracies)):
            ax.text(x, acc - 3, f'{acc:.1f}%', ha='center', va='top', 
                   fontsize=28, color='black', weight='normal')
        
        # Styling with updated limits
        ax.set_ylim(0, 70)
        ax2.set_ylim(0, 400)
        
        # Y-axis ticks only (labels will be added as figure text)
        if dataset_mode:
            if col == 0:  # Left column - show left y-axis ticks
                ax.tick_params(axis='y', labelcolor='black', labelsize=36)
            else:  # Right column - hide left y-axis ticks and labels
                ax.tick_params(axis='y', labelleft=False, left=False)
            
            if col == 1:  # Right column - show right y-axis ticks
                ax2.tick_params(axis='y', labelsize=36)
            else:  # Left column - hide right y-axis ticks and labels
                ax2.tick_params(axis='y', labelright=False, right=False)
        else:
            # Original logic for side-by-side
            if folder_idx == 0:
                ax.tick_params(axis='y', labelcolor='black', labelsize=36)
            else:
                ax.tick_params(axis='y', labelleft=False, left=False)
            
            if folder_idx == num_folders - 1:
                ax2.tick_params(axis='y', labelsize=26)
            else:
                ax2.tick_params(axis='y', labelright=False, right=False)
        
        # X-axis - only on bottom row
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=36)
        
        if dataset_mode:
            if row == 1:  # Bottom row - show x-axis labels but no individual xlabel
                pass
            else:  # Top row - hide x-axis labels and ticks
                ax.tick_params(axis='x', labelbottom=False, bottom=False)
        # No individual xlabel for side-by-side mode either
        
        # Title - only show Clean/Noisy on first row
        if dataset_mode:
            if row == 0:  # First row only
                ax.set_title(folder_name, fontsize=38, pad=25)
            else:  # Second row - no title
                ax.set_title('', fontsize=38, pad=25)
        else:
            ax.set_title(folder_name, fontsize=38, pad=25)
        
        # Show all spines for full box around plot
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['left'].set_visible(True)
        
        # Hide y-axis labels for right subplots in each row (moved this logic above)
    
    # Add dataset labels on the outer side (moved slightly right)
    if dataset_mode:
        fig.text(-0.08, 0.72, 'TinyImageNet', va='center', rotation='vertical', 
                fontsize=44, ha='center', weight='bold')
        fig.text(-0.08, 0.25, 'CIFAR-100', va='center', rotation='vertical', 
                fontsize=44, ha='center', weight='bold')
        
        # Add single axis labels following plot_budgets.py style
        # Left y-axis label (accuracy) - positioned outside left edge, neutral color
        fig.text(-0.02, 0.5, 'Accuracy (%)', va='center', rotation='vertical', 
                fontsize=48, ha='center', color='black', weight='normal')
        
        # Right y-axis label (emissions) - positioned outside right edge
        fig.text(1.02, 0.5, 'kgCO$_2$eq', va='center', rotation='vertical',
                fontsize=48, ha='center', weight='normal')
        
        # Single centered x-axis label
        fig.text(0.5, -0.02, 'Method', va='center', ha='center', 
                fontsize=48, weight='normal')
    
    # Create legend with gray bars and colored lines (following plot_budgets.py)
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='#E69F00', marker='o', linestyle='-',
                  markersize=10, linewidth=3, label='Clean Data End Accuracy'),
        plt.Line2D([0], [0], color='cornflowerblue', marker='o', linestyle='-',
                  markersize=10, linewidth=3, label='Corrupted Data End Accuracy'),
        Patch(facecolor='#555555', alpha=0.4, label='Emissions')
    ]
    
    legend_y = 1.12 if dataset_mode else 1.16
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, legend_y), ncol=3, fontsize=38, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/{output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Combined budget-accuracy plot saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Create budget vs accuracy plot with carbon bars.")
    parser.add_argument("folders", nargs='+', help="Folders containing .pkl files")
    parser.add_argument("--output_name", type=str, default="budget_accuracy_combined",
                        help="Name of the output file (without extension)")
    parser.add_argument("--folder_names", nargs='+', 
                        help="Custom names for folders (default: use folder names)")
    
    args = parser.parse_args()
    
    # Load results from folders
    folder_results = []
    folder_names = args.folder_names if args.folder_names else []
    
    for idx, folder in enumerate(args.folders):
        if not os.path.exists(folder):
            print(f"[ERROR] Folder {folder} does not exist")
            continue
            
        print(f"[INFO] Processing folder: {folder}")
        results = load_final_results(folder)
        folder_results.append(results)
        
        # Use custom name if provided, otherwise use folder name
        if idx < len(folder_names):
            display_name = folder_names[idx]
        else:
            display_name = os.path.basename(folder.rstrip('/'))
        
        if idx >= len(folder_names):
            folder_names.append(display_name)
    
    if not folder_results:
        print("[ERROR] No valid data found")
        return
    
    # Create plot
    plot_results_combined(folder_results, folder_names, args.output_name)

if __name__ == "__main__":
    main()
