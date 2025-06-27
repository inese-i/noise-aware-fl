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
    elif "oortcawt" in label or "oortwt" in label:
        if "100%" in label:
            return "OortCAWT"
        elif "30%" in label:
            return "OortCAWT 30%"
        elif "0%" in label:
            return "OortCAWT 0%"
        else:
            return "OortCAWT"
    elif "oortca" in label:
        if "100%" in label:
            return "OortCA"
        elif "30%" in label:
            return "OortCA 30%"
        elif "0%" in label:
            return "OortCA 0%"
        else:
            return "OortCA"
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
        # For files like nnoisecifar100_OortWT 30%.pkl, extract the method part
        elif "nnoisecifar100_" in base_name:
            base_name = base_name.replace("nnoisecifar100_", "")
    
    return base_name

def plot_carbon_and_final_accuracy(carbon_files, budget_files, ax, collected_lines, collected_labels, show_xlabel=True, custom_ylim=None, custom_yticks=None):
    """Plot carbon emissions as bars with final accuracy as dotted line"""
    # Use same colors as plot_budgets.py
    custom_colors = {
        "Oort": "gray",  # Oort 100% should always be gray
        "OortCA": "#E69F00",  # Match plot_budgets.py
        "OortCA 30%": "#E69F00",  
        "OortCA 0%": "#009E73",
        "OortCAWT": "cornflowerblue",  # Match plot_budgets.py
        "OortCAWT 30%": "cornflowerblue",
        "OortCAWT 0%": "cornflowerblue",
        "Random": "gray"  # Changed from "black" to "gray" to match plot_budgets.py
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
            print(f"DEBUG Carbon: {file_path} -> base: '{base_label}' -> display: '{display_label}'")
            
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
    
    # For emissions bars: include all methods that have carbon data
    # For accuracy line: only include methods that have both carbon and accuracy data (and are not 100%)
    all_carbon_methods = set(carbon_data.keys())
    accuracy_methods = set(carbon_data.keys()) & set(accuracy_data.keys())
    
    # Define custom ordering: 0% first, then 100%
    def get_method_order(method):
        if "0%" in method:
            return (0, method)  # 0% methods come first
        elif "100%" in method or method in ["Oort", "OortCA", "OortCAWT"]:
            return (2, method)  # 100% methods come last
        else:
            return (1, method)  # Other methods (like 30%) in between
    
    # Sort all carbon methods for the bars
    sorted_methods = sorted(all_carbon_methods, key=get_method_order)
    
    # Collect all methods for emissions bars
    all_methods_for_bars = []
    all_emissions = []
    all_colors = []
    
    # Collect only non-100% methods for accuracy line
    methods_for_accuracy = []
    accuracies_for_line = []
    
    for method in sorted_methods:
        # Add all methods to emissions bars (as long as they have carbon data)
        all_methods_for_bars.append(method)
        all_emissions.append(carbon_data[method])
        all_colors.append(custom_colors.get(method, "#333333"))
        
        # Only add methods to accuracy line if they have both carbon and accuracy data AND are not 100%
        if (method in accuracy_data and 
            not ("100%" in method or method in ["Oort", "OortCA", "OortCAWT"])):
            methods_for_accuracy.append(method)
            accuracies_for_line.append(accuracy_data[method])
    
    if all_methods_for_bars and all_emissions:
        # Create secondary y-axis for carbon emissions (on the right)
        ax2 = ax.twinx()
        
        # Create bars for carbon emissions on the right axis (convert to kg like plot_budgets.py)
        # Use same styling as plot_budgets.py: alpha=0.4, width=5
        bars = ax2.bar(all_methods_for_bars, [e/1000 for e in all_emissions], color=all_colors, alpha=0.4, width=0.6)
        
        # Remove value labels on bars (commented out)
        # for bar, emission in zip(bars, emissions):
        #     height = bar.get_height()
        #     ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
        #            f'{emission/1000:.1f}', ha='center', va='bottom', fontsize=20)
        
        # Plot final accuracy as black solid line on the left axis (only for non-100% methods)
        # Use same styling as plot_budgets.py: marker='o', markersize=10, linewidth=2
        if methods_for_accuracy and accuracies_for_line:
            # Map accuracy line to the correct x-positions in the full bar chart
            accuracy_x_positions = []
            for acc_method in methods_for_accuracy:
                if acc_method in all_methods_for_bars:
                    accuracy_x_positions.append(all_methods_for_bars.index(acc_method))
                else:
                    print(f"[WARN] Method {acc_method} not found in all_methods_for_bars")
            
            line, = ax.plot(accuracy_x_positions, accuracies_for_line, 'k-', linewidth=2, marker='o', 
                            markersize=10, label='Final Accuracy')
        
        # Add horizontal line for Oort 100% (unconstrained availability) similar to plot_budgets.py
        # Look for Oort 100% file in budget_files to get the accuracy
        dashed_line_for_legend = None
        for file_path in budget_files:
            fname = os.path.basename(file_path)
            if fname.endswith("100%.pkl") and "carbon" not in fname:
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if hasattr(data, "metrics_centralized") and "accuracy" in data.metrics_centralized:
                        # Get final accuracy (last entry) like plot_budgets.py
                        acc = 100 * data.metrics_centralized["accuracy"][-1][1]
                        ax.axhline(y=acc, color='black', linestyle='--', linewidth=1.5)
                        # Position text at left edge like in plot_budgets.py
                        ax.text(x=-0.4, y=acc + 0.3, s="Oort", fontsize=30, va='bottom', ha='left')
                        # Create line object for legend
                        import matplotlib.lines as mlines
                        dashed_line_for_legend = mlines.Line2D([], [], color='black', linestyle='--', label='Oort')
                        break  # Only need to add this once
                except Exception as e:
                    print(f"[WARN] Could not read accuracy from {fname}: {e}")
        
        # Add accuracy value labels (consistent for all rows)
        for i, (x_pos, acc) in enumerate(zip(accuracy_x_positions, accuracies_for_line)):
            ax.text(x_pos, acc - 1, f'{acc:.1f}%', ha='center', va='top', 
                    fontsize=int(35 * 1.2))  # Updated fontsize
        
        # Format primary y-axis (accuracy) - left side
        # ax.set_ylabel("Final Accuracy (%)", fontsize=30)
        if show_xlabel:
            ax.set_xlabel("Methods", fontsize=int(42 * 1.2))  # Updated fontsize
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=int(40 * 1.2), width=2, length=6)  # Updated fontsize
        # ax.tick_params(axis='x', rotation=45)  # Removed rotation
        
        # Apply custom y-axis limits and ticks if provided (for second dataset)
        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)
        else:
            ax.set_ylim(55, 65)
            
        if custom_yticks is not None:
            ax.set_yticks(custom_yticks)
        else:
            ax.set_yticks([55, 60, 65])
        
        # Extract percentage values for x-axis labels (use all methods for bars)
        percentage_labels = []
        for method in all_methods_for_bars:
            if "30%" in method:
                percentage_labels.append("30%")
            elif "20%" in method:
                percentage_labels.append("20%")
            elif "0%" in method:
                percentage_labels.append("0%")
            elif "100%" in method or "Oort" == method:
                percentage_labels.append("100%")
            else:
                # For methods without clear percentage, use original label
                percentage_labels.append(method)
        
        # Set x-tick labels to show only percentages (for all methods including 100%)
        ax.set_xticks(range(len(all_methods_for_bars)))
        if show_xlabel:
            ax.set_xticklabels(percentage_labels, fontsize=40)
        else:
            ax.set_xticklabels([])  # Hide x-tick labels
        
        # Format secondary y-axis (carbon) - right side, similar to plot_budgets.py
        # Y-axis label will be set as shared label in main function
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(True)
        ax2.tick_params(axis='y', labelsize=int(40 * 1.2), width=2, length=6)  # Updated fontsize
        
        # Set y-limits and formatting similar to plot_budgets.py
        from matplotlib.ticker import FuncFormatter
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax2.set_ylim(0, 400)  # Set maximum to 400 kg as requested
        
        # Add the accuracy line to collected items for legend
        collected_lines.append(line)
        collected_labels.append('Final Accuracy')
        
        # Add the dashed line to collected items for legend if it exists
        if dashed_line_for_legend:
            collected_lines.append(dashed_line_for_legend)
            collected_labels.append('Oort')

def plot_accuracy_convergence(file_paths, ax, collected_lines, collected_labels, show_xlabel=True, custom_ylim=None, custom_yticks=None):
    """Plot accuracy convergence over rounds"""
    MAX_ROUNDS = 100
    # Use same colors as plot_budgets.py
    custom_colors = {
        "Oort": "gray",  # Oort 100% should always be gray
        "OortCA": "#E69F00",  # Match plot_budgets.py
        "OortCA 30%": "#E69F00",  
        "OortCA 0%": "#009E73", 
        "OortCAWT": "cornflowerblue",  # Match plot_budgets.py
        "OortCAWT 30%": "cornflowerblue",
        "OortCAWT 0%": "cornflowerblue",
        "Random": "gray"  # Changed from "black" to "gray" to match plot_budgets.py
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
            # Oort 100% should use dashed lines, Random should use dashed lines
            linestyle = '--' if "random" in display_label.lower() or display_label == "Oort" else '-'

            line, = ax.plot(round_global, acc_global, label=display_label, 
                          color=color, linestyle=linestyle, linewidth=2)

            if display_label not in collected_labels:
                collected_lines.append(line)
                collected_labels.append(display_label)

        except Exception as e:
            print(f"Accuracy file {file_path}: {e}. Skipping...")

    if show_xlabel:
        ax.set_xlabel("Training Rounds", fontsize=42)
    # Y-axis label will be set as shared label in main function
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=40, width=2, length=6)
    ax.set_xticks(np.arange(0, MAX_ROUNDS + 1, step=25))
    if not show_xlabel:
        ax.set_xticklabels([])  # Hide x-tick labels
    
    # Set custom y-axis limits and ticks if provided, otherwise use defaults
    if custom_ylim and custom_yticks:
        ax.set_ylim(custom_ylim)
        ax.set_yticks(custom_yticks)
    else:
        ax.set_yticks(np.arange(30, 72, 20))
        ax.set_ylim(30, 72)

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results with carbon emissions and accuracy.")
    parser.add_argument("folder", type=str, help="Path to the folder containing .pkl files.")
    parser.add_argument("--folder2", type=str, default=None,
                        help="Path to the second folder for additional row of plots.")
    parser.add_argument("--output_name", type=str, default="evaluation_plot",
                        help="Name of the output file (without extension).")
    args = parser.parse_args()

    folder_path = args.folder
    folder2_path = args.folder2
    
    # Determine number of rows based on whether second folder is provided
    num_rows = 2 if folder2_path else 1
    
    collected_lines = []
    collected_labels = []

    fig, axes = plt.subplots(num_rows, 2, figsize=(16, 5.5 * num_rows))
    
    # Handle case where we have only one row (axes is 1D)
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot first folder (first row)
    print(f"Processing first folder: {folder_path}")
    budget_files = load_budget_data(folder_path)
    carbon_files = load_carbon_data(folder_path)
    
    print(f"Found {len(budget_files)} budget files and {len(carbon_files)} carbon files")

    # Determine if this is the bottom row (show x-axis labels)
    is_bottom_row = (num_rows == 1) or (folder2_path is None)

    # Plot accuracy convergence on the left (first subplot)
    if budget_files:
        plot_accuracy_convergence(budget_files, axes[0, 0], collected_lines, collected_labels, show_xlabel=is_bottom_row)
    else:
        print("No budget files found in first folder")

    # Plot carbon emissions with final accuracy on the right (second subplot)
    if carbon_files and budget_files:
        plot_carbon_and_final_accuracy(carbon_files, budget_files, axes[0, 1], collected_lines, collected_labels, show_xlabel=is_bottom_row)
    else:
        print("No carbon or budget files found for combined plot in first folder")

    # Add subplot label (a) in left margin for first row
    fig.text(
        -0.15,  # X: very close to left edge
        axes[0, 0].get_position().y1 - 0.1,  # Y: just above top of first row
        "(a)",
        fontsize=42,
        fontweight="bold",
        ha="left",
        va="top"
    )

    # If only one row, add shared y-axis labels here
    if num_rows == 1:
        y_center = 0.5
        fig.text(-0.05, y_center, "Accuracy (%)", va='center', rotation='vertical', fontsize=42)
        fig.text(1.02, y_center, "kgCO$_2$eq", va='center', rotation='vertical', fontsize=42)

    # Plot second folder if provided (second row)
    if folder2_path:
        print(f"Processing second folder: {folder2_path}")
        budget_files2 = load_budget_data(folder2_path)
        carbon_files2 = load_carbon_data(folder2_path)
        
        print(f"Found {len(budget_files2)} budget files and {len(carbon_files2)} carbon files in second folder")

        # Plot accuracy convergence on the left (third subplot)
        if budget_files2:
            plot_accuracy_convergence(budget_files2, axes[1, 0], collected_lines, collected_labels, 
                                     show_xlabel=True,  # Second row is always bottom row
                                     custom_ylim=(0, 70), custom_yticks=[0, 35, 70])
        else:
            print("No budget files found in second folder")

        # Plot carbon emissions with final accuracy on the right (fourth subplot)
        if carbon_files2 and budget_files2:
            # Use custom y-axis limits for TinyImageNet (0-70) with 3 tick labels
            plot_carbon_and_final_accuracy(carbon_files2, budget_files2, axes[1, 1], collected_lines, collected_labels, 
                                          show_xlabel=True, custom_ylim=(0, 70), custom_yticks=[0, 35, 70])  # Second row is always bottom row
        else:
            print("No carbon or budget files found for combined plot in second folder")

        # Add subplot label (b) in left margin for second row
        fig.text(
            -0.15,  # X: very close to left edge
            axes[1, 0].get_position().y1 - 0.1,  # Y: just above top of second row
            "(b)",
            fontsize=42,
            fontweight="bold",
            ha="left",
            va="top"
        )

    # Add shared y-axis labels (matching plot_budgets.py style)
    y_center = 0.5 if num_rows == 1 else 0.5  # Center position for y-labels
    fig.text(-0.05, y_center, "Accuracy (%)", va='center', rotation='vertical', fontsize=42)
    fig.text(1.02, y_center, "kgCO$_2$eq", va='center', rotation='vertical', fontsize=42)

    # Create global legend with unique labels
    if collected_lines and collected_labels:
        unique_items = {}
        for line, label in zip(collected_lines, collected_labels):
            if label not in unique_items:
                unique_items[label] = line
        
        fig.legend(unique_items.values(), unique_items.keys(),
                   loc="upper center",
                   ncol=min(2, len(unique_items)),
                   fontsize=40,
                   frameon=False,
                   bbox_to_anchor=(0.5, 1.25))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/{args.output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
