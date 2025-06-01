import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
import re

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

def get_method_name(file_path):
    filename = os.path.basename(file_path)
    match = re.match(r"([A-Za-z0-9_]+) \d+%", filename)
    return match.group(1).lower() if match else None

def load_data(folder_path, exclude_keywords=None):
    exclude_keywords = exclude_keywords or ["intensity_matrix", "selected_clients"]
    files = os.listdir(folder_path)
    included_files = [
        os.path.join(folder_path, f) for f in files
        if f.endswith(".pkl") and not any(keyword in f for keyword in exclude_keywords)
    ]
    return included_files

def get_final_accuracy(file_path):
    try:
        with open(file_path, 'rb') as f:
            history = pickle.load(f)
        if hasattr(history, "metrics_centralized") and "accuracy" in history.metrics_centralized:
            return 100 * history.metrics_centralized["accuracy"][-1][1]
        else:
            return None
    except Exception as e:
        print(f"[WARN] Could not process {file_path}: {e}")
        return None

def plot_folder(ax, folder_root, exclude_keywords=None):
    from matplotlib.ticker import FuncFormatter
    import matplotlib.lines as mlines

    handles, labels = [], []
    ax2 = ax.twinx()
    dashed_line_for_legend = None
    emissions_tick_label_x = None
    all_budgets = set()

    method_data = {}
    carbon_data = {}
    legend_entries = {}

    folder_prefix = os.path.basename(folder_root)

    files = [
        os.path.join(folder_root, f)
        for f in os.listdir(folder_root)
        if f.endswith(".pkl") and not any(k in f for k in (exclude_keywords or []))
    ]

    for file_path in files:
        fname = os.path.basename(file_path)

        # Match only files like: plot_budget_OortCA 20%.pkl
        match = re.match(rf"{re.escape(folder_prefix)}_([A-Za-z0-9]+) (\d+)%\.pkl", fname)
        if not match:
            continue

        method = match.group(1).lower()
        budget = int(match.group(2))

        acc = get_final_accuracy(file_path)
        if acc is not None:
            method_data.setdefault(method, []).append((budget, acc))

        # Emissions filename stays the same
        carbon_file = os.path.join(folder_root, f"{method.capitalize()} {budget}%_carbon_used.pkl")
        if os.path.exists(carbon_file):
            try:
                with open(carbon_file, "rb") as f:
                    val = pickle.load(f)
                if isinstance(val, (int, float)):
                    carbon_data.setdefault(method, []).append((budget, val))
            except Exception as e:
                print(f"[WARN] Couldn't read {carbon_file}: {e}")

    for method, entries in method_data.items():
        filtered_entries = [(b, a) for (b, a) in entries if b != 100]
        if not filtered_entries:
            continue

        filtered_entries = sorted(filtered_entries)
        budgets, accuracies = zip(*filtered_entries)
        all_budgets.update(budgets)

        method_key = method.lower()
        color = custom_colors.get(method_key, None)
        label = label_mapping.get(method_key, method)

        # Always plot a solid line now (no 100% → no change to line type)
        line, = ax.plot(
            budgets, accuracies,
            marker='o', markersize=10,
            label=label,
            color=color or None,
            linewidth=2
        )
        legend_entries[label] = line

        # Add emissions bars
        if method in carbon_data:
            for b, val in carbon_data[method]:
                ax2.bar(x=b, height=val / 1000, width=5, color=color or 'gray', alpha=0.4)
            emission_label = f"{label} Emissions"
            bar_patch = mlines.Line2D([], [], color=color or 'gray', linewidth=10, alpha=0.4, label=emission_label)
            legend_entries[emission_label] = bar_patch

    try:
        emissions_path = os.path.join(folder_root, "Oort 100%_carbon_used.pkl")
        if os.path.exists(emissions_path):
            with open(emissions_path, "rb") as f:
                emissions_val = pickle.load(f)
            if isinstance(emissions_val, (int, float)):
                final_x = max(all_budgets) + 10
                ax2.bar(x=final_x, height=emissions_val / 1000, width=5, color='black', alpha=0.4)
                legend_entries["Oort Emissions"] = mlines.Line2D([], [], color='black', linewidth=10, alpha=0.6, label="Oort Emissions")
                emissions_tick_label_x = final_x
    except Exception as e:
        print(f"[WARN] Could not read OortCA 100%_carbon_used.pkl: {e}")

    for file_name in os.listdir(folder_root):
        if file_name.endswith("100%.pkl") and "carbon" not in file_name:
            full_path = os.path.join(folder_root, file_name)
            try:
                with open(full_path, 'rb') as f:
                    data = pickle.load(f)
                acc = 100 * max(v for _, v in data.metrics_centralized["accuracy"])
                ax.axhline(y=acc, color='black', linestyle='--', linewidth=1.5)
                ax.text(x=min(all_budgets) - 3, y=acc + 0.3, s="Oort", fontsize=30, va='bottom', ha='left')
                if dashed_line_for_legend is None:
                    dashed_line_for_legend = mlines.Line2D([], [], color='black', linestyle='--', label='Unconstrained Availability')
            except Exception as e:
                print(f"[WARN] Could not read accuracy from {file_name}: {e}")

    if dashed_line_for_legend:
        legend_entries['Unconstrained Availability'] = dashed_line_for_legend
    xticks = sorted(all_budgets)
    if emissions_tick_label_x is not None:
        xticks.append(emissions_tick_label_x)

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [f"{x}" if x != emissions_tick_label_x else "100" for x in xticks],
        fontsize=33
    )

    ax.tick_params(axis='both', labelsize=33, width=2, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Carbon Budget (%)", fontsize=35)
    #ax.set_ylabel("Accuracy (%)", fontsize=30)


    ax.set_ylim(52, 68)
    ax.set_yticks([55, 60, 65])

    #ax2.set_ylabel("CO$_2$eq/kWh", fontsize=30)
    ax2.set_ylim(0, 400)
    #ax2.set_yticks([0, 250, 500])

    ax2.tick_params(axis='y', labelsize=33)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Return correct label-handle pairing
    labels, handles = zip(*legend_entries.items())
    print("DEBUG plot_folder():", labels)  # ✅ DEBUG
    return list(handles), list(labels)

def main():
    import argparse
    import os
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(description="Plot final accuracy vs carbon budget for multiple folders.")
    parser.add_argument("folders", nargs="+", type=str, help="List of root folders to process.")
    parser.add_argument("--output_name", type=str, default="final_accuracy_vs_budget", help="Output file name (no extension).")
    parser.add_argument("--exclude", type=str, nargs="*", default=["intensity_matrix", "selected_clients"], help="Exclude files with these substrings.")
    args = parser.parse_args()

    num_folders = len(args.folders)
    fig, axes = plt.subplots(num_folders, 1, figsize=(11, 5 * num_folders), sharex=True)

    if num_folders == 1:
        axes = [axes]  # make iterable

    all_handles, all_labels = [], []
    import string

    subplot_labels = list(string.ascii_lowercase)

    for idx, (ax, folder) in enumerate(zip(axes, args.folders)):
        handles, labels = plot_folder(ax, folder, exclude_keywords=args.exclude)
        all_handles.extend(handles)
        all_labels.extend(labels)

        # Add subplot label (a), (b) in left margin (next to y-axis)
        fig.text(
            -0.15,  # X: very close to left edge
            ax.get_position().y1 - 0.1,  # Y: just above top of this subplot
            f"({subplot_labels[idx]})",
            fontsize=35,
            fontweight="bold",
            ha="left",
            va="top"
        )

    # Remove x-axis ticks (both major and minor) on top subplot
    if len(axes) > 1:
        axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # === Shared y-label ===
    fig.text(-0.05, 0.55, "Accuracy (%)", va='center', rotation='vertical', fontsize=35)
    fig.text(1.02, 0.55, "kgCO$_2$eq", va='center', rotation='vertical', fontsize=35)

    # === MANUAL LEGEND ORDER ===
    ordered_labels = [
        "OortCA",
        "OortCAWT",  "Unconstrained Availability", "OortCA Emissions", "OortCAWT Emissions",
        "Oort Emissions"
    ]

    # Build a lookup dictionary for fast access
    handles_dict = {label: handle for label, handle in zip(all_labels, all_handles)}

    # Filter out only those that are present
    ordered_handles = [handles_dict[l] for l in ordered_labels if l in handles_dict]
    ordered_labels = [l for l in ordered_labels if l in handles_dict]

    # Plot legend
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=2,
        fontsize=33,
        frameon=False,
        handlelength=2,
        columnspacing=2
    )

    for ax in axes[:-1]:
        ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/{args.output_name}.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
