import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import re

def load_data(folder_path, exclude_keywords=None):
    exclude_keywords = exclude_keywords or ["intensity_matrix", "selected_clients"]
    files = os.listdir(folder_path)
    included_files = [
        os.path.join(folder_path, f) for f in files
        if f.endswith(".pkl") and not any(keyword in f for keyword in exclude_keywords)
    ]
    return included_files

def simplify_label(label):
    label = label.lower()
    if "oortwt c=" in label:
        return "OortWT"
    elif "oortwt" in label:
        return "OortWT"
    elif "oortca" in label:
        return "OortCA"
    elif "oort" in label:
        return "Oort"
    elif "randomwt" in label:
        return "RandomWT"
    elif "random" in label:
        return "Random"
    else:
        return label

def plot_data(file_paths, ax, collected_lines, collected_labels):
    MAX_ROUNDS = 100
    custom_colors = {
        "RandomWT": "#2c89d9",
        "Random": "black",
        "OortWT": '#7f7fff',
        "OortCA": '#7f7fff',
        "Oort": '#E69F00'
    }

    num_files = len(file_paths)
    color_map = plt.get_cmap('tab10')
    default_colors = [color_map(i) for i in np.linspace(0, 1, num_files)]

    labels = [
        os.path.splitext(os.path.basename(file))[0].split('_', 1)[-1] for file in file_paths
    ]

    simplified_labels = [simplify_label(lbl) for lbl in labels]
    sorted_data = sorted(zip(simplified_labels, file_paths), key=lambda x: x[0])
    labels, file_paths = zip(*sorted_data)

    for idx, file_path in enumerate(file_paths):
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

            label = labels[idx]
            color = custom_colors.get(label, default_colors[idx])
            linestyle = '--' if "random" in label.lower() else '-'

            line, = ax.plot(round_global, acc_global, label=label, color=color, linestyle=linestyle, linewidth=2)

            collected_lines.append(line)
            collected_labels.append(label)

        except Exception as e:
            print(f"{file_path}: {e}. Skipping...")

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=25, width=1.5, length=6)
    ax.set_xticks(np.arange(0, MAX_ROUNDS + 1, step=25))
    ax.set_yticks(np.arange(30, 72, 20))
    ax.set_ylim(30, 72)

def main():
    parser = argparse.ArgumentParser(description="Plot probing results from multiple folders.")
    parser.add_argument("folders", nargs='+', type=str, help="Paths to the folders containing .pkl files.")
    parser.add_argument("--titles", nargs='+', type=str, default=None,
                        help="Titles for each subplot (default: folder names).")
    parser.add_argument("--output_name", type=str, default="multi_accuracy",
                        help="Name of the output file (without extension).")
    parser.add_argument("--exclude", type=str, nargs="*", default=["intensity_matrix", "selected_clients"],
                        help="List of substrings to exclude files containing them.")
    args = parser.parse_args()

    folder_paths = args.folders
    folder_titles = args.titles or [os.path.basename(folder.rstrip('/')) for folder in folder_paths]
    num_folders = len(folder_paths)

    collected_lines = []
    collected_labels = []

    if num_folders == 1:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        file_paths = load_data(folder_paths[0], exclude_keywords=args.exclude)

        if file_paths:
            plot_data(file_paths, ax, collected_lines, collected_labels)
        else:
            ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', fontsize=18)
            ax.set_axis_off()

        ax.set_xlabel("Training Rounds", fontsize=30)
        ax.set_ylabel("Accuracy (%)", fontsize=30)

        # Legend
        unique = dict(zip(collected_labels, collected_lines))
        fig.legend(unique.values(), unique.keys(),
                   loc="upper center",
                   ncol=min(2, len(unique)),
                   fontsize=25,
                   frameon=False,
                   bbox_to_anchor=(0.5, 1.12))

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    else:
        n_cols = 2
        n_rows = math.ceil(num_folders / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5.5*n_rows), squeeze=False)

        for idx, (folder_path, title) in enumerate(zip(folder_paths, folder_titles)):
            row, col = divmod(idx, n_cols)
            ax = axs[row][col]
            file_paths = load_data(folder_path, exclude_keywords=args.exclude)

            if file_paths:
                plot_data(file_paths, ax, collected_lines, collected_labels)
            else:
                ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', fontsize=18)
                ax.set_axis_off()

            ax.set_title(title, fontsize=30)

            if row == n_rows - 1:
                ax.set_xlabel("Training Rounds", fontsize=30)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel("Accuracy (%)", fontsize=30)
            else:
                ax.set_yticklabels([])

        for empty_idx in range(num_folders, n_rows * n_cols):
            row, col = divmod(empty_idx, n_cols)
            fig.delaxes(axs[row][col])

        # Global legend
        unique = dict(zip(collected_labels, collected_lines))
        fig.legend(unique.values(), unique.keys(),
                   loc="upper center",
                   ncol=min(2, len(unique)),
                   fontsize=25,
                   frameon=False,
                   bbox_to_anchor=(0.5, 1.22))

        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.5, w_pad=3.0)

    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/{args.output_name}.pdf"
    plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
