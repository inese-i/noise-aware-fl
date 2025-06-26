import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

def load_budget_data(folder_path, exclude_keywords=None):
    """Load budget data files, excluding certain keywords"""
    exclude_keywords = exclude_keywords or ["intensity_matrix", "selected_clients", "_carbon_used", "_cumulative_usage"]
    files = os.listdir(folder_path)
    included_files = [
        os.path.join(folder_path, f) for f in files
        if f.endswith(".pkl") and not any(keyword in f for keyword in exclude_keywords)
    ]
    return included_files

def load_carbon_data(folder_path):
    """Load carbon usage data from _carbon_used.pkl files"""
    files = os.listdir(folder_path)
    carbon_files = [
        os.path.join(folder_path, f) for f in files
        if f.endswith("_carbon_used.pkl")
    ]
    return carbon_files

def load_data(folder_path, exclude_keywords=None):
    exclude_keywords = exclude_keywords or ["intensity_matrix", "selected_clients"]
    files = os.listdir(folder_path)
    included_files = [
        os.path.join(folder_path, f) for f in files
        if f.endswith(".pkl") and not any(keyword in f for keyword in exclude_keywords)
    ]
    return included_files

def simplify_label(label):
    """Convert file labels to display labels"""
    label = label.lower()
    
    # Extract method and budget information
    if "random" in label:
        if "20%" in label:
            return "Random 20%"
        elif "30%" in label:
            return "Random 30%"
        else:
            return "Random"
    elif "oort" in label:
        if "100%" in label:
            return "Oort"
        elif "30%" in label:
            return "OortCA 30%"
        elif "0%" in label:
            return "OortCA 0%"
        else:
            return "Oort"
    
    return label.replace("_", " ").title()

def get_method_and_budget(file_path):
    """Extract method name and budget percentage from filename"""
    filename = os.path.basename(file_path)
    # Pattern: methodname budget.pkl or methodname_cumulative_usage.pkl
    if "_cumulative_usage" in filename:
        base_name = filename.replace("_cumulative_usage.pkl", "")
    else:
        base_name = filename.replace(".pkl", "")
        # For files like cifar100budget_Oort 100%.pkl, extract the method part
        if "cifar100budget_" in base_name:
            base_name = base_name.replace("cifar100budget_", "")
    
    return base_name

def plot_carbon_and_final_accuracy(carbon_files, budget_files, ax, collected_lines, collected_labels):
    """Plot carbon emissions as bars with final accuracy as dotted line"""
    custom_colors = {
        "Oort": "#E69F00",
        "OortCA 30%": "#56B4E9", 
        "OortCA 0%": "#009E73",
        "Random": "black"
    }
    
    # Process carbon files
    carbon_data = {}
    for file_path in carbon_files:
        try:
            with open(file_path, 'rb') as f:
                carbon_value = pickle.load(f)
            
            # Extract the scalar carbon value
            if hasattr(carbon_value, 'item'):
                emission_value = carbon_value.item()
            else:
                emission_value = float(carbon_value)
            
            base_label = get_method_and_budget(file_path)
            display_label = simplify_label(base_label)
            carbon_data[display_label] = emission_value
            
        except Exception as e:
            print(f"Carbon file {file_path}: {e}. Skipping...")
    
    # Process budget files to get final accuracy
    accuracy_data = {}
    for file_path in budget_files:
        try:
            with open(file_path, 'rb') as f:
                history = pickle.load(f)

            if hasattr(history, "metrics_centralized") and "accuracy" in history.metrics_centralized:
                global_accuracy_centralized = history.metrics_centralized["accuracy"]
                # Get final accuracy (last entry)
                final_acc = global_accuracy_centralized[-1][1] * 100.0
                
                base_label = get_method_and_budget(file_path)
                display_label = simplify_label(base_label)
                accuracy_data[display_label] = final_acc
                
        except Exception as e:
            print(f"Budget file {file_path}: {e}. Skipping...")
    
    # Combine data for methods that have both carbon and accuracy data
    methods = []
    emissions = []
    final_accuracies = []
    colors = []
    
    # Sort methods to ensure consistent ordering
    sorted_methods = sorted(set(carbon_data.keys()) & set(accuracy_data.keys()))
    
    for method in sorted_methods:
        methods.append(method)
        emissions.append(carbon_data[method])
        final_accuracies.append(accuracy_data[method])
        colors.append(custom_colors.get(method, "#333333"))
    
    if methods and emissions:
        # Create secondary y-axis for carbon emissions (on the right)
        ax2 = ax.twinx()
        
        # Create bars for carbon emissions on the right axis (convert to kg like plot_budgets.py)
        bars = ax2.bar(methods, [e/1000 for e in emissions], color=colors, alpha=0.8)
        
        # Remove value labels on bars (commented out)
        # for bar, emission in zip(bars, emissions):
        #     height = bar.get_height()
        #     ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
        #            f'{emission/1000:.1f}', ha='center', va='bottom', fontsize=20)
        
        # Plot final accuracy as dotted line on the left axis
        x_pos = range(len(methods))
        line, = ax.plot(x_pos, final_accuracies, 'k--', linewidth=3, marker='o', 
                        markersize=8, label='Final Accuracy')
        
        # Add accuracy value labels
        for i, acc in enumerate(final_accuracies):
            ax.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=18, fontweight='bold')
        
        # Format primary y-axis (accuracy) - left side
        # ax.set_ylabel("Final Accuracy (%)", fontsize=30)
        ax.set_xlabel("Methods", fontsize=30)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=25, width=1.5, length=6)
        # ax.tick_params(axis='x', rotation=45)  # Removed rotation
        ax.set_ylim(55, 65)
        ax.set_yticks([55, 60, 65])
        
        # Format secondary y-axis (carbon) - right side, similar to plot_budgets.py
        ax2.set_ylabel("kgCO$_2$eq", fontsize=30)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(True)
        ax2.tick_params(axis='y', labelsize=25, width=1.5, length=6)
        
        # Set y-limits and formatting similar to plot_budgets.py
        from matplotlib.ticker import FuncFormatter
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax2.set_ylim(0, 400)  # Set maximum to 400 kg as requested
        
        # Add the accuracy line to collected items for legend
        collected_lines.append(line)
        collected_labels.append('Final Accuracy')

def plot_accuracy_convergence(file_paths, ax, collected_lines, collected_labels):
    """Plot accuracy convergence over rounds"""
    MAX_ROUNDS = 100
    custom_colors = {
        "Oort": "#E69F00",
        "OortCA 30%": "#56B4E9",
        "OortCA 0%": "#009E73", 
        "Random": "black"
    }

    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                history = pickle.load(f)

            if hasattr(history, "metrics_centralized") and "accuracy" in history.metrics_centralized:
                global_accuracy_centralized = history.metrics_centralized["accuracy"]
            else:
                raise ValueError(f"File {file_path} does not contain expected metrics structure.")

            round_global = [data[0] for data in global_accuracy_centralized]
            acc_global = [100.0 * data[1] for data in global_accuracy_centralized]

            if len(round_global) > MAX_ROUNDS:
                round_global = round_global[:MAX_ROUNDS]
                acc_global = acc_global[:MAX_ROUNDS]

            base_label = get_method_and_budget(file_path)
            display_label = simplify_label(base_label)
            
            color = custom_colors.get(display_label, "#333333")
            linestyle = '--' if "random" in display_label.lower() else '-'

            line, = ax.plot(round_global, acc_global, label=display_label, 
                          color=color, linestyle=linestyle, linewidth=2)

            if display_label not in collected_labels:
                collected_lines.append(line)
                collected_labels.append(display_label)

        except Exception as e:
            print(f"Accuracy file {file_path}: {e}. Skipping...")

    ax.set_xlabel("Training Rounds", fontsize=30)
    ax.set_ylabel("Accuracy (%)", fontsize=30)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=25, width=1.5, length=6)
    ax.set_xticks(np.arange(0, MAX_ROUNDS + 1, step=25))
    ax.set_yticks(np.arange(30, 72, 20))
    ax.set_ylim(30, 72)

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results with carbon emissions and accuracy.")
    parser.add_argument("folder", type=str, help="Path to the folder containing .pkl files.")
    parser.add_argument("--output_name", type=str, default="evaluation_plot",
                        help="Name of the output file (without extension).")
    args = parser.parse_args()

    folder_path = args.folder
    
    # Load budget data (accuracy files)
    budget_files = load_budget_data(folder_path)
    
    # Load carbon data
    carbon_files = load_carbon_data(folder_path)
    
    print(f"Found {len(budget_files)} budget files and {len(carbon_files)} carbon files")

    collected_lines = []
    collected_labels = []

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Plot accuracy convergence on the left (first subplot, like plot_acc.py)
    if budget_files:
        plot_accuracy_convergence(budget_files, axes[0], collected_lines, collected_labels)
    else:
        print("No budget files found")

    # Plot carbon emissions with final accuracy on the right (second subplot)
    if carbon_files and budget_files:
        plot_carbon_and_final_accuracy(carbon_files, budget_files, axes[1], collected_lines, collected_labels)
    else:
        print("No carbon or budget files found for combined plot")

    # Add dataset name label on the left edge for both subplots
    fig.text(-0.02, 0.5, "CIFAR-100", rotation=90, fontsize=30,  
             ha='center', va='center')

    # Create global legend with unique labels
    if collected_lines and collected_labels:
        unique_items = {}
        for line, label in zip(collected_lines, collected_labels):
            if label not in unique_items:
                unique_items[label] = line
        
        fig.legend(unique_items.values(), unique_items.keys(),
                   loc="upper center",
                   ncol=min(3, len(unique_items)),
                   fontsize=25,
                   frameon=False,
                   bbox_to_anchor=(0.5, 1.12))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/{args.output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
