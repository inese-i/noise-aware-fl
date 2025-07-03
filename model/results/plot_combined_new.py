import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.colors as mcolors
import argparse
import re

# Copied plotting functions instead of importing

def load_data(folder_path):
    """
    Load intensity and selection matrices from separate files based on their names.
    """
    files = os.listdir(folder_path)

    # Separate files by type
    intensity_files = [f for f in files if "intensity_matrix" in f and f.endswith(".pkl")]
    selected_files = [f for f in files if "selected_clients" in f and f.endswith(".pkl")]

    data = []

    for selected_file in selected_files:
        try:
            # Load selection matrix
            with open(os.path.join(folder_path, selected_file), "rb") as f:
                selection_matrix = np.array(pickle.load(f))

            # Match intensity file
            intensity_file = selected_file.replace("selected_clients", "intensity_matrix")
            if intensity_file in intensity_files:
                with open(os.path.join(folder_path, intensity_file), "rb") as f:
                    intensity_matrix = np.array(pickle.load(f))

                data.append((selection_matrix, intensity_matrix, selected_file.replace(".pkl", "")))
            else:
                print(f"[WARNING] No matching intensity file for {selected_file}.")
        except Exception as e:
            print(f"[ERROR] Could not process file {selected_file}: {e}")

    return data

def load_budget_data(folder_path, exclude_keywords=None):
    """Load budget data files, excluding certain keywords"""
    exclude_keywords = exclude_keywords or ["intensity_matrix", "selected_clients", "_carbon_used", "_cumulative_usage"]
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

def plot_accuracy_convergence(file_paths, ax, collected_lines, collected_labels, show_xlabel=True, custom_ylim=None, custom_yticks=None, max_rounds=100, show_xticks=True):
    """Plot accuracy convergence over rounds"""
    MAX_ROUNDS = max_rounds
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
    
    # Set x-axis ticks based on MAX_ROUNDS
    if MAX_ROUNDS == 50:
        ax.set_xticks(np.arange(0, MAX_ROUNDS + 1, step=10))  # 0, 10, 20, 30, 40, 50
    else:
        ax.set_xticks(np.arange(0, MAX_ROUNDS + 1, step=25))  # 0, 25, 50, 75, 100
    
    # Control x-tick label visibility
    if not show_xticks:
        ax.set_xticklabels([])  # Hide x-tick labels
    
    # Set custom y-axis limits and ticks if provided, otherwise use defaults
    if custom_ylim and custom_yticks:
        ax.set_ylim(custom_ylim)
        ax.set_yticks(custom_yticks)
    else:
        ax.set_yticks([0, 35, 70])
        ax.set_ylim(0, 70)

def plot_client_selection_counts_with_intensity(data, output_dir="plots", output_name="client_selection_counts_with_intensity", show_legend=True, ax=None, show_title=True):
    if not data:
        print("[ERROR] No selection data provided.")
        return

    data.sort(key=lambda x: x[2])  # sort by file_name
    num_methods = len(data)
    total_clients = data[0][0].shape[1]

    client_counts_by_method = []
    avg_intensities_by_method = []
    global_max = 0

    for selection_matrix, intensity_matrix, _ in data:
        # Mask intensity to only consider selected clients
        masked_intensity = np.ma.masked_where(selection_matrix.T == 0, intensity_matrix)
        avg_intensity = np.mean(masked_intensity, axis=1)  # Average intensity for selected clients
        counts = np.sum(selection_matrix, axis=0)
        client_counts_by_method.append(counts)

        avg_intensities_by_method.append(avg_intensity)
        global_max = max(global_max, np.max(counts))

    # Set a fixed figure size based on your preferred proportions
    if ax is None:
        fig, axes = plt.subplots(1, num_methods, figsize=(7 * num_methods, 5), sharey=True)
        if num_methods == 1:
            axes = [axes]
    else:
        fig = None
        # When using external axis, we still want separate subplots for each method
        # This will be handled at the calling level
        axes = [ax]

    # Create a custom colormap 
    semi_transparent_green = [0.75, 0.85, 0.95]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_to_orange_darker",
        [semi_transparent_green, "gray", "lightgray","#E69F00"]
    )
    norm = Normalize(vmin=0, vmax=700)  # Adjust this to the maximum possible intensity range

    # Create a ScalarMappable to associate the colormap with the normalized intensity values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array just for colorbar creation

    for idx, (counts, avg_intensity, (selection_matrix, intensity_matrix, file_name)) in enumerate(
            zip(client_counts_by_method, avg_intensities_by_method, data)):
        if ax is None:
            # Normal plotting with separate axes for each method
            current_ax = axes[idx]
        else:
            # When using external axis, only plot the first method on the provided axis
            if idx == 0:
                current_ax = ax
            else:
                break  # Skip other methods when external axis is provided
        
        client_ids = np.arange(total_clients)
        
        # Color bars based on average intensity
        colors = cmap(norm(avg_intensity))

        current_ax.bar(client_ids, counts, color=colors, edgecolor='none')
        current_ax.set_ylim(0, global_max + 5)

        if ax is None:
            current_ax.set_xlabel("Client ID", fontsize=42, labelpad=15)
            # Handle y-axis label only when not using external axes
            if idx == 0:
                current_ax.set_ylabel("Selection Count", fontsize=42, labelpad=15)
        # When using external axis, don't set xlabel/ylabel here - it will be handled by calling code
        
        current_ax.set_xticks([0, 15, 30])
        current_ax.tick_params(axis='x', labelsize=40)
        current_ax.tick_params(axis='y', labelsize=40)

        title = file_name.split('_')[0]
        # Change "Oort 100%" to just "Oort"
        if "Oort 100%" in title:
            title = "Oort"
        
        # Only show title if requested
        if show_title:
            current_ax.set_title(title, fontsize=42, pad=20, ha='center')

def get_method_column_mapping(selection_data_list):
    """Create a mapping of methods to column indices based on method type"""
    # Define standard column order: 100%, 30%, 0%
    column_order = ['100%', '30%', '0%']
    
    # Collect all unique method types across all folders
    all_method_types = set()
    for selection_data in selection_data_list:
        for _, _, file_name in selection_data:
            for method_type in column_order:
                if method_type in file_name:
                    all_method_types.add(method_type)
                    break
    
    # Create column mapping
    method_to_column = {}
    column_index = 0
    for method_type in column_order:
        if method_type in all_method_types:
            method_to_column[method_type] = column_index
            column_index += 1
    
    return method_to_column, column_index
def sort_by_percentage(data):
    """Sort data by percentage in descending order (100%, 30%, 0%)"""
    def get_percentage(item):
        if len(item) == 3:
            _, _, file_name = item
        else:
            _, file_name = item
        # Extract all numbers from filename
        numbers = re.findall(r'\d+', file_name)
        if numbers:
            # Return negative of the first number to reverse sort order
            return -int(numbers[0])
        return -999  # Put files without numbers at the end
    
    return sorted(data, key=get_percentage)

def plot_combined_visualization(folder1, folder2=None, folder3=None, folder4=None, output_name="combined_plot", titles=None):
    """
    Create a combined visualization with accuracy convergence plots on the left
    and selection/intensity plots on the right for each folder.
    
    Args:
        folder1: Path to first folder (dataset 1, condition 1)
        folder2: Path to second folder (dataset 1, condition 2)
        folder3: Path to third folder (dataset 2, condition 1) - optional
        folder4: Path to fourth folder (dataset 2, condition 2) - optional
        output_name: Output filename without extension
        titles: Custom row titles (list of strings) - NOT USED ANYMORE
    """
    # Determine number of rows based on datasets
    has_second_dataset = folder3 is not None
    num_rows = 4 if has_second_dataset else 2
    
    # Load data for first dataset
    selection_data1 = load_data(folder1)
    selection_data1 = sort_by_percentage(selection_data1)
    budget_files1 = load_budget_data(folder1)
    
    selection_data2 = load_data(folder2) if folder2 else []
    selection_data2 = sort_by_percentage(selection_data2)
    budget_files2 = load_budget_data(folder2) if folder2 else []
    
    # Load data for second dataset if provided
    if has_second_dataset:
        selection_data3 = load_data(folder3)
        selection_data3 = sort_by_percentage(selection_data3)
        budget_files3 = load_budget_data(folder3)
        
        selection_data4 = load_data(folder4) if folder4 else []
        selection_data4 = sort_by_percentage(selection_data4)
        budget_files4 = load_budget_data(folder4) if folder4 else []
        
        # Create method-to-column mapping for all datasets
        all_selection_data = [selection_data1, selection_data2, selection_data3, selection_data4]
        method_to_column, max_methods = get_method_column_mapping(all_selection_data)
    else:
        # Create method-to-column mapping for first dataset only
        method_to_column, max_methods = get_method_column_mapping([selection_data1, selection_data2])
    
    # Create figure with custom layout
    total_cols = 1 + 1 + max_methods  # 1 for accuracy + 1 spacer + methods for selection
    
    # Use gridspec for flexible layout
    fig_height = 20 if has_second_dataset else 10  # Double height for second dataset
    fig = plt.figure(figsize=(7 * total_cols + 1, fig_height))
    
    # Define width ratios: accuracy column, spacer, selection columns, colorbar
    width_ratios = [1.5, 0.6] + [1] * max_methods + [0.05]
    
    # Define height ratios: add extra space between datasets
    if has_second_dataset:
        height_ratios = [1, 1, 0.3, 1, 1]  # rows 1, 2, spacer, 3, 4
        actual_num_rows = 5  # Including spacer row
    else:
        height_ratios = [1] * num_rows
        actual_num_rows = num_rows
    
    gs = gridspec.GridSpec(actual_num_rows, total_cols + 1, 
                          width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          wspace=0.05, hspace=0.3)
    
    # Create axes arrays
    accuracy_axes = []
    selection_axes = []
    
    for row in range(num_rows):
        # Map logical row to actual grid row (skip spacer row)
        if has_second_dataset and row >= 2:
            grid_row = row + 1  # Skip spacer row (row 2 in grid)
        else:
            grid_row = row
        
        # Accuracy axis (left side - column 0)
        ax_acc = fig.add_subplot(gs[grid_row, 0])
        accuracy_axes.append(ax_acc)
        
        # Selection axes (right side - columns 2 onwards, skipping column 1 which is spacer)
        selection_row = []
        for col in range(max_methods):
            ax = fig.add_subplot(gs[grid_row, col + 2])  # +2 to skip accuracy column (0) and spacer (1)
            selection_row.append(ax)
        selection_axes.append(selection_row)
    
    # Colorbar axis for selection plots - span all rows
    cax = fig.add_subplot(gs[:, -1])
    
    # Create shared colormap and norm for intensity
    semi_transparent_green = [0.75, 0.85, 0.95]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_to_orange_darker",
        [semi_transparent_green, "gray", "lightgray","#E69F00"]
    )
    norm = Normalize(vmin=0, vmax=700)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Track collected lines and labels for accuracy plots
    collected_lines = []
    collected_labels = []
    
    # Helper function to plot a row of data
    def plot_row(row_idx, selection_data, budget_files, folder_name):
        print(f"Processing folder {row_idx + 1}: {folder_name}")
        
        # Accuracy plot (left side)
        is_last_row = (row_idx == num_rows - 1)
        if budget_files:
            # Determine max_rounds based on dataset: Dataset 1 (rows 0,1) = 100 rounds, Dataset 2 (rows 2,3) = 50 rounds
            if has_second_dataset and row_idx >= 2:
                max_rounds = 50  # Second dataset
                # Show x-tick labels only on bottom row of second dataset (row 3)
                show_xticks = (row_idx == 3)
            else:
                max_rounds = 100  # First dataset
                # Show x-tick labels only on bottom row of first dataset (row 1), or last row if single dataset
                if has_second_dataset:
                    show_xticks = (row_idx == 1)  # Bottom row of first dataset
                else:
                    show_xticks = (row_idx == 1)  # For single dataset, show on row 1 (bottom row)
                
            plot_accuracy_convergence(budget_files, accuracy_axes[row_idx], 
                                    collected_lines, collected_labels, 
                                    show_xlabel=False, max_rounds=max_rounds, show_xticks=show_xticks)
        
        # Selection plots (right side)
        for selection_matrix, intensity_matrix, file_name in selection_data:
            # Determine which column this method should go to
            method_column = None
            for method_type in ['100%', '30%', '0%']:
                if method_type in file_name:
                    method_column = method_to_column.get(method_type)
                    break
            
            if method_column is not None and method_column < max_methods:
                ax = selection_axes[row_idx][method_column]
                single_data = [(selection_matrix, intensity_matrix, file_name)]
                
                # Show titles only on the first row (row 0)
                show_title = (row_idx == 0)
                
                plot_client_selection_counts_with_intensity(
                    single_data, output_name=f"temp_{row_idx}_{method_column}", ax=ax, show_title=show_title
                )
                
                # Hide x-axis labels for non-last row
                if not is_last_row:
                    ax.set_xlabel("")
                    ax.tick_params(axis='x', labelbottom=False)
                
                # Hide y-axis labels for non-leftmost columns
                if method_column > 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelleft=False)
    
    # Plot all rows
    plot_row(0, selection_data1, budget_files1, folder1)
    if folder2:
        plot_row(1, selection_data2, budget_files2, folder2)
    if has_second_dataset:
        plot_row(2, selection_data3, budget_files3, folder3)
        if folder4:
            plot_row(3, selection_data4, budget_files4, folder4)
    
    # Add subplot labels (a), (b), (c), (d) for each row
    import string
    subplot_labels = list(string.ascii_lowercase)
    
    for row_idx in range(num_rows):
        ax = accuracy_axes[row_idx]  # Use accuracy axes for positioning
        # Add subplot label in left margin (matching plot_budgets.py style)
        fig.text(
            0.04,  # X: positioned in left margin, moved closer to plots (+15 points approximately)
            ax.get_position().y1 - 0.09,  # Y: just above top of this subplot
            f"({subplot_labels[row_idx]})",
            fontsize=42,
            ha="left",
            va="top"
        )
    
    # Hide unused selection subplots for all rows
    all_data_sets = [selection_data1, selection_data2]
    if has_second_dataset:
        all_data_sets.extend([selection_data3, selection_data4])
    
    for row_idx, selection_data in enumerate(all_data_sets):
        if row_idx >= num_rows:
            break
        used_columns = set()
        for selection_matrix, intensity_matrix, file_name in selection_data:
            for method_type in ['100%', '30%', '0%']:
                if method_type in file_name:
                    col = method_to_column.get(method_type)
                    if col is not None:
                        used_columns.add(col)
                    break
        
        # Hide unused subplots for this row
        for col in range(max_methods):
            if col not in used_columns:
                selection_axes[row_idx][col].set_visible(False)
    
    # Add shared labels
    if has_second_dataset:
        # Y-axis labels for each dataset (moved 10 points to the right)
        # Dataset 1 labels
        fig.text(0.08, 0.75, 'Accuracy (%)', va='center', rotation='vertical', fontsize=42)
        fig.text(0.37, 0.75, 'Selection Count', va='center', rotation='vertical', fontsize=42)
        
        # Dataset 2 labels  
        fig.text(0.08, 0.25, 'Accuracy (%)', va='center', rotation='vertical', fontsize=42)
        fig.text(0.37, 0.25, 'Selection Count', va='center', rotation='vertical', fontsize=42)
    else:
        # Y-axis labels for single dataset (centered, moved 10 points to the right)
        fig.text(0.08, 0.5, 'Accuracy (%)', va='center', rotation='vertical', fontsize=42)
        fig.text(0.40, 0.5, 'Selection Count', va='center', rotation='vertical', fontsize=42)
    
    # X-axis labels - positioned below the plots and their tick labels
    y_pos = -0.02 if has_second_dataset else -0.01  # Lower position for 4-row layout
    fig.text(0.25, y_pos, 'Training Rounds', ha='center', va='bottom', fontsize=42)
    fig.text(0.65, y_pos, 'Client ID', ha='center', va='bottom', fontsize=42)
    
    # Add dataset labels if second dataset is present (moved 10 points to the right)
    if has_second_dataset:
        fig.text(0.015, 0.75, 'CIFAR100', va='center', rotation='vertical', fontsize=42, weight='bold')
        fig.text(0.015, 0.25, 'TinyImageNet', va='center', rotation='vertical', fontsize=42, weight='bold')
    
    # Add column titles above the plots
    # Get the position of the first row for title placement
    first_acc_ax = accuracy_axes[0]
    first_sel_ax = selection_axes[0][0] if selection_axes[0] else None
    
    # Title above accuracy column
    fig.text(first_acc_ax.get_position().x0 + first_acc_ax.get_position().width/2, 
             first_acc_ax.get_position().y1 + 0.08, 
             'Convergence', 
             ha='center', va='bottom', fontsize=42)
    
    # Title above selection columns
    if first_sel_ax:
        # Calculate center position across all selection columns
        first_sel_pos = selection_axes[0][0].get_position()
        last_sel_pos = selection_axes[0][-1].get_position() if len(selection_axes[0]) > 1 else first_sel_pos
        center_x = (first_sel_pos.x0 + last_sel_pos.x1) / 2
        
        fig.text(center_x, 
                 first_sel_pos.y1 + 0.08, 
                 'Selection Count with Average Carbon Intensity', 
                 ha='center', va='bottom', fontsize=42)
    
    # Add intensity colorbar for selection plots
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=35)
    cbar.set_label("gCO$_2$eq/kWh", fontsize=40, labelpad=20)
    
    # Create legend for accuracy plots
    if collected_lines and collected_labels:
        unique_items = {}
        for line, label in zip(collected_lines, collected_labels):
            if label not in unique_items:
                unique_items[label] = line
        
        # Position legend centered above all plots
        legend_y = 1.08 if has_second_dataset else 1.15  # Adjust for taller figure
        fig.legend(unique_items.values(), unique_items.keys(),
                   loc="upper center",
                   ncol=min(5, len(unique_items)),
                   fontsize=35,
                   frameon=False,
                   bbox_to_anchor=(0.5, legend_y))
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/{output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Combined plot saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Create combined visualization with accuracy and selection plots.")
    parser.add_argument("folders", nargs='+', help="2 or 4 folders containing .pkl files. For 2 folders: dataset1_condition1 dataset1_condition2. For 4 folders: dataset1_condition1 dataset1_condition2 dataset2_condition1 dataset2_condition2")
    parser.add_argument("--second_dataset", action='store_true', 
                        help="Flag to indicate that 4 folders are provided for 2 datasets comparison")
    parser.add_argument("--output_name", type=str, default="combined_plot",
                        help="Name of the output file (without extension).")
    
    args = parser.parse_args()
    
    if args.second_dataset:
        if len(args.folders) != 4:
            print("Error: --second_dataset flag requires exactly 4 folders")
            return
        plot_combined_visualization(
            folder1=args.folders[0],
            folder2=args.folders[1], 
            folder3=args.folders[2],
            folder4=args.folders[3],
            output_name=args.output_name
        )
    else:
        if len(args.folders) != 2:
            print("Error: Without --second_dataset flag, exactly 2 folders are required")
            return
        plot_combined_visualization(
            folder1=args.folders[0],
            folder2=args.folders[1],
            output_name=args.output_name
        )

if __name__ == "__main__":
    main()
