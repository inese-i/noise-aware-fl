import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import os
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import re

def load_selection_data(folder_path):
    """
    Load only *_selected_clients.pkl files.
    Returns list of (selection_matrix, filename_without_extension).
    """
    selected_files = [
        f for f in os.listdir(folder_path)
        if f.endswith("_selected_clients.pkl")
    ]

    data = []
    for file_name in sorted(selected_files):
        try:
            with open(os.path.join(folder_path, file_name), "rb") as f:
                selection_matrix = np.array(pickle.load(f))
            data.append((selection_matrix, file_name.replace(".pkl", "")))
        except Exception as e:
            print(f"[ERROR] Failed to load {file_name}: {e}")
    return data

def load_data(folder_path):
    """
    Load intensity and selection matrices from separate files based on their names.

    Parameters:
    - folder_path (str): Path to the folder containing .pkl files.

    Returns:
    - list of tuple: Each tuple contains (selection_matrix, intensity_matrix, file_name).
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

gray_white_cmap = ListedColormap(["#B0B0B0", "white"])  # light gray and white

def plot_selection_with_intensity(data, output_dir="plots", output_name="selection_intensity_combined"):
    if not data:
        print("[ERROR] No data to plot.")
        return

    # Sort and reorder
    data_100 = [d for d in data if '100%' in d[2]]
    data_rest = [d for d in data if '100%' not in d[2]]
    data_ordered = sorted(data_100, key=lambda x: x[2]) + sorted(data_rest, key=lambda x: x[2])

    num_plots = len(data_ordered)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 9), gridspec_kw={'wspace': 0.4})

    if num_plots == 1:
        axes = [axes]

    #cmap = plt.get_cmap('viridis_r')
    semi_transparent_green = [0.75, 0.85, 0.95]
 #[0.6, 0.98, 0.6, 0.6]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_to_orange_darker",
        [semi_transparent_green, "gray", "lightgray","#E69F00"]
    )
    norm = Normalize(vmin=0, vmax=700)

    for idx, (selection_matrix, intensity_matrix, file_name) in enumerate(data_ordered):
        ax = axes[idx]
        min_rounds = selection_matrix.shape[0]
        total_clients = selection_matrix.shape[1]

        selection_matrix_flipped = selection_matrix[:min_rounds, :].T[::-1, :]
        intensity_matrix_flipped = intensity_matrix[::-1, :]

        selected_mask = selection_matrix_flipped
        intensity_masked = np.ma.masked_where(selected_mask == 0, intensity_matrix_flipped)

        ax.imshow(np.ones_like(selected_mask), aspect='auto', cmap="gray", norm=Normalize(vmin=0, vmax=1))
        cax = ax.imshow(intensity_masked, aspect='auto', cmap=cmap, norm=norm, origin='upper')

        ax.set_xticks([0, min_rounds - 1])
        ax.set_xticklabels([0, min_rounds], fontsize=50)
        ax.set_xlabel("")

        ax.set_ylim(-0.5, total_clients - 0.5)
        if idx == 0:
            ax.set_yticks([0,15, 30])
            ax.set_yticklabels([30,15, 0], fontsize=50)
            ax.set_ylabel("Client ID", fontsize=50, labelpad=20)  # axis label padding

            # Add extra padding to tick labels (e.g., shift them left)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_x(-0.05)  # shift left (more negative = more padding)

        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False)

        # Identify and label titles
        if '100%' in file_name:
            title = 'Unconstrained\nAvailability'
        else:
            match = re.search(r'(\d+%)', file_name)
            title = match.group(1) if match else file_name
            if title == '0%':
                zero_index = idx
                title = 'Zero-Emissions\nAvailability'

        ax.set_title(title, fontsize=50, pad=30)

    # Add shared title above plots
    fig.suptitle("Carbon Budget", fontsize=60, y=1.00, x=0.645)

    # Shared x-axis label
    fig.supxlabel("Training Rounds", fontsize=55, y=0.01)

    # --- Add vertical dashed line between first and second subplot ---
    if num_plots > 2:
        bbox1 = axes[1].get_position()  # 2nd plot
        bbox2 = axes[2].get_position()  # 3rd plot
        x_between = (bbox1.x1 + bbox2.x0) / 2

        # Optional tweak: shift left (-) or right (+)
        x_between += 0.001  # ← Shift right (or use -0.005 to shift left)

        fig.lines.append(plt.Line2D(
            [x_between, x_between], [0.12, 0.99],
            transform=fig.transFigure,
            linestyle='--', color='black', linewidth=5
        ))

    plt.subplots_adjust(bottom=0.20, top=0.75)
    bbox = axes[-1].get_position()
    cbar_ax = fig.add_axes([bbox.x1 + 0.025, bbox.y0, 0.012, bbox.height])  # [left, bottom, width, height]

    cbar = fig.colorbar(cax, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=50)
    cbar.set_label("gCO$_2$eq/kWh", fontsize=50, labelpad=20)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{output_dir}/{output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300)
    plt.close(fig)
    print(f"[INFO] Combined plot saved to {plot_filename}.")

def plot_client_selection_counts(data, output_dir="plots", output_name="client_selection_counts", show_legend=True):

    if not data:
        print("[ERROR] No selection data provided.")
        return

    data.sort(key=lambda x: x[1])  # sort by file_name
    num_methods = len(data)
    total_clients = data[0][0].shape[1]

    client_counts_by_method = []
    global_max = 0
    for selection_matrix, _ in data:
        counts = np.sum(selection_matrix, axis=0)
        client_counts_by_method.append(counts)
        global_max = max(global_max, np.max(counts))

    fig, axes = plt.subplots(1, num_methods, figsize=(7 * num_methods, 10), sharey=True)
    if num_methods == 1:
        axes = [axes]

    for idx, (counts, (_, file_name)) in enumerate(zip(client_counts_by_method, data)):
        ax = axes[idx]
        client_ids = np.arange(total_clients)

        # Color: first 5 in yellow, others in gray
        colors = ['#E69F00' if i < 6 else 'gray' for i in client_ids]

        ax.bar(client_ids, counts, color=colors, edgecolor='none')
        ax.set_ylim(0, global_max + 5)
        ax.set_yticks([0, 50, 100])

        ax.set_xlabel("Client ID", fontsize=62, labelpad=15)
        if idx == 0:
            ax.set_ylabel("Selection Count", fontsize=65, labelpad=15)
        ax.set_xticks([0, 15, 30])
        ax.tick_params(axis='x', labelsize=58)
        ax.tick_params(axis='y', labelsize=58)

        title = file_name.split('_')[0]
        ax.set_title(title, fontsize=58, pad=20)

    if show_legend:
        handles = [
            Patch(facecolor="#E69F00", label="Corrupted (ID 0–5)"),
            Patch(facecolor="gray", label="Unmodified Data (ID 6–29)")
        ]
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            fontsize=58,
            frameon=False
        )

    plt.tight_layout(rect=[0, 0, 1, 0.75])  # Leave space at top for legend
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{output_dir}/{output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300)
    plt.close(fig)

    print(f"[INFO] Highlighted client selection count plot saved as {plot_filename}")


def plot_client_selection_counts_with_intensity(data, output_dir="plots", output_name="client_selection_counts_with_intensity", show_legend=True):
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
    fig, axes = plt.subplots(1, num_methods, figsize=(7 * num_methods, 5), sharey=True)
    if num_methods == 1:
        axes = [axes]

    # Create a custom colormap from gray to #E69F00
    cmap = mcolors.LinearSegmentedColormap.from_list("gray_white_orange", ["gray", "white", "#E69F00"])
    norm = Normalize(vmin=0, vmax=700)  # Adjust this to the maximum possible intensity range

    # Create a ScalarMappable to associate the colormap with the normalized intensity values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array just for colorbar creation

    for idx, (counts, avg_intensity, (selection_matrix, intensity_matrix, file_name)) in enumerate(
            zip(client_counts_by_method, avg_intensities_by_method, data)):
        ax = axes[idx]
        client_ids = np.arange(total_clients)

        # Color bars based on average intensity
        colors = cmap(norm(avg_intensity))

        ax.bar(client_ids, counts, color=colors, edgecolor='none')
        ax.set_ylim(0, global_max + 5)

        ax.set_xlabel("Client ID", fontsize=35, labelpad=15)
        if idx == 0:
            ax.set_ylabel("Selection Count", fontsize=35, labelpad=15)
        ax.set_xticks([0, 15, 30])
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

        title = file_name.split('_')[0]
        ax.set_title(title, fontsize=35, pad=20)

    # Add the colorbar (heatmap legend) outside the plot to the right
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("CO$_2$eq/kWh", fontsize=30)

    plt.subplots_adjust(left=0.12, right=0.85, bottom=0.25, top=0.85)  # Adjust margins for labels

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{output_dir}/{output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300)
    plt.close(fig)

    print(f"[INFO] Highlighted client selection count plot with intensity saved as {plot_filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot selection matrix visualizations.")
    parser.add_argument("folder", nargs="+", help="One or more folders with .pkl files.")
    parser.add_argument("--output_name", type=str, default="selection_plot", help="Output filename (no extension).")
    parser.add_argument("--titles", nargs="+", help="Optional custom row titles (must match number of folders).")
    parser.add_argument("--selection", action="store_true", help="Only plot selection counts or matrix (no intensity).")
    parser.add_argument("--intenscount", action="store_true", help="Plot selection count bars colored by average intensity.")

    args = parser.parse_args()

    if args.intenscount:
        if len(args.folder) > 1:
            print("[ERROR] --intenscount only supports one folder at a time.")
        else:
            data = load_data(args.folder[0])
            plot_client_selection_counts_with_intensity(data, output_name="client_bar_colored")

    elif args.selection:
        if len(args.folder) == 1:
            data = load_selection_data(args.folder[0])
            plot_client_selection_counts(data, output_name=args.output_name)


    else:
        if len(args.folder) > 1:
            print("[ERROR] Intensity overlay plotting only supported for one folder at a time.")
        else:
            data = load_data(args.folder[0])
            plot_selection_with_intensity(data, output_name=args.output_name)
