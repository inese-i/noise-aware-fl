import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
from matplotlib.gridspec import GridSpec
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
from matplotlib.lines import Line2D


def load_flat_folder(folder_path):
    carbon_files = {}
    accuracy_files = {}

    for f in os.listdir(folder_path):
        if not f.endswith(".pkl"):
            continue

        filepath = os.path.join(folder_path, f)

        if f.endswith("_carbon_used.pkl"):
            method_name = f.replace("_carbon_used.pkl", "")
            carbon_files[method_name] = filepath

        elif any(suffix in f for suffix in ["_selected_clients", "_intensity_matrix", "_carbon_used"]):
            continue

        elif re.match(r"^.+?_(.+?)\.pkl$", f):
            method_name = re.match(r"^.+?_(.+?)\.pkl$", f).group(1)
            accuracy_files[method_name] = filepath

    return carbon_files, accuracy_files

def load_flat_folder_with_cumulative(folder_path):
    cumulative_files = {}
    accuracy_files = {}

    for f in os.listdir(folder_path):
        if not f.endswith(".pkl"):
            continue

        filepath = os.path.join(folder_path, f)

        # Match cumulative file, e.g., "OortCA_cumulative_usage.pkl"
        if f.endswith("_cumulative_usage.pkl"):
            method_name = f.replace("_cumulative_usage.pkl", "")
            cumulative_files[method_name] = filepath

        # Skip irrelevant files
        elif any(suffix in f for suffix in ["_selected_clients", "_intensity_matrix", "_carbon_used"]):
            continue

        # Match accuracy file, extract method name from suffix
        match = re.match(r".*_(\w+)\.pkl$", f)  # e.g., "test_OortCA.pkl" → "OortCA"
        if match:
            method_name = match.group(1)
            accuracy_files[method_name] = filepath

    return cumulative_files, accuracy_files


def load_line_files(folder_path):
    exclude_keywords = ["intensity_matrix", "selected_clients", "carbon_used"]
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pkl") and not any(k in f for k in exclude_keywords)
    ]

def load_bar_data(folder_path):
    carbon_files = {}
    accuracy_files = {}

    for f in os.listdir(folder_path):
        if not f.endswith(".pkl"):
            continue

        path = os.path.join(folder_path, f)

        if f.endswith("_carbon_used.pkl"):
            method = f.split("_")[-3] if "_carbon_used.pkl" in f else None
            if method:
                carbon_files[method] = path

        elif "intensity_matrix" in f or "selected_clients" in f or "carbon_used" in f:
            continue

        else:
            method = f.split("_")[-1].replace(".pkl", "")
            accuracy_files[method] = path

    return carbon_files, accuracy_files

def get_final_accuracy(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        acc_list = data.metrics_centralized.get("accuracy", [])
        return max(x[1] for x in acc_list) * 100 if acc_list else None
    except:
        return None

def get_total_carbon(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            val = pickle.load(f)
        return float(val)
    except:
        return None

def plot_accuracy_lines(ax, file_paths):
    colors = {
        "RandomWT": "#2c89d9",
        "Random": "black",
        "OortWT": '#7f7fff',
        "OortCA": '#7f7fff',
        "Oort": '#E69F00'
    }

    for path in sorted(file_paths):
        name = "OortCA" if "OortCA" in path else "Oort"
        try:
            with open(path, "rb") as f:
                hist = pickle.load(f)
            acc_list = hist.metrics_centralized.get("accuracy", [])
            rounds = [t for t, _ in acc_list[:100]]
            acc = [a * 100 for _, a in acc_list[:100]]
            ax.plot(rounds, acc, label=name, color=colors[name], linewidth=3)
        except Exception as e:
            print(f"[WARN] Could not load {path}: {e}")

    ax.set_ylim(30, 72)
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_yticks(np.arange(30, 72, 20))
    ax.tick_params(labelsize=26, width=2, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("Training Rounds", fontsize=32)
    ax.set_ylabel("Accuracy (%)", fontsize=32)
    ax.legend(fontsize=26, frameon=False, loc="lower right")


def get_emission_at_max_accuracy(accuracy_file, cumulative_file):
    try:
        with open(accuracy_file, 'rb') as f:
            history = pickle.load(f)
        acc_list = history.metrics_centralized.get("accuracy", [])
        if not acc_list:
            print(f"[DEBUG] No accuracy data in {accuracy_file}")
            return None

        max_acc_round, max_acc_val = max(acc_list, key=lambda x: x[1])
        print(f"[DEBUG] Max accuracy {max_acc_val:.4f} at round {max_acc_round} in {accuracy_file}")

        with open(cumulative_file, 'rb') as f:
            cumulative_emissions = pickle.load(f)

        if not isinstance(cumulative_emissions, list):
            print(f"[DEBUG] Cumulative file is not a list: {cumulative_file}")
            return None

        if max_acc_round >= len(cumulative_emissions):
            max_acc_round = len(cumulative_emissions) - 1
            print(f"[DEBUG] Adjusted max_acc_round to {max_acc_round}")

        emission = cumulative_emissions[max_acc_round]
        print(f"[DEBUG] Total emissions up to max accuracy round {max_acc_round}: {emission:.2f}")
        return emission

    except Exception as e:
        print(f"[ERROR] Failed to get emission at max accuracy: {e}")
        return None


def plot_combined(folders, split=False):
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

    custom_colors = {
        "RandomWT": "#2c89d9",
        "Random": "black",
        "OortWT": '#7f7fff',
        "OortCA": '#7f7fff',
        "Oort": '#E69F00'
    }

    for i, folder in enumerate(folders):
        folder_name = os.path.basename(folder.rstrip("/"))

        # Create a new figure for the accuracy plot (figure 1)
        fig1 = plt.figure(figsize=(14, 5), constrained_layout=True)
        ax1 = fig1.add_subplot(111)  # Accuracy plot

        # Create a new figure for the emissions/accuracy plot (figure 2)
        fig2 = plt.figure(figsize=(14, 6), constrained_layout=True)
        ax2 = fig2.add_subplot(111)  # Emissions/Accuracy plot
        ax2b = ax2.twinx()  # Create a second y-axis for emissions

        def load_flat_folder_with_cumulative(folder_path):
            cumulative_files, accuracy_files = {}, {}
            print(f"[DEBUG] Scanning folder: {folder_path}")

            for f in os.listdir(folder_path):
                if not f.endswith(".pkl"):
                    continue

                filepath = os.path.join(folder_path, f)

                if f.endswith("_cumulative_usage.pkl"):
                    method_name = f.replace("_cumulative_usage.pkl", "")
                    cumulative_files[method_name] = filepath
                    print(f"[DEBUG] Found cumulative file: {f} -> method: {method_name}")

                elif any(suffix in f for suffix in ["_selected_clients", "_intensity_matrix", "_carbon_used"]):
                    continue

                else:
                    match = re.match(r".*_(\w+)\.pkl$", f)
                    if match:
                        method_name = match.group(1)
                        accuracy_files[method_name] = filepath
                        print(f"[DEBUG] Found accuracy file: {f} -> method: {method_name}")

            print(f"[DEBUG] Total accuracy methods: {list(accuracy_files.keys())}")
            print(f"[DEBUG] Total cumulative methods: {list(cumulative_files.keys())}")
            return cumulative_files, accuracy_files

        def get_emission_at_max_accuracy(accuracy_file, cumulative_file):
            try:
                with open(accuracy_file, 'rb') as f:
                    history = pickle.load(f)
                acc_list = history.metrics_centralized.get("accuracy", [])
                print(f"[DEBUG] Accuracy entries in {accuracy_file}: {len(acc_list)}")
                if not acc_list:
                    return None

                max_acc_round, max_acc = max(acc_list, key=lambda x: x[1])
                print(f"[DEBUG] Max accuracy round: {max_acc_round}, value: {max_acc}")

                with open(cumulative_file, 'rb') as f:
                    cumulative_emissions = pickle.load(f)
                print(f"[DEBUG] Cumulative entries in {cumulative_file}: {len(cumulative_emissions)}")

                if max_acc_round >= len(cumulative_emissions):
                    max_acc_round = len(cumulative_emissions) - 1

                emission = cumulative_emissions[max_acc_round]
                print(f"[DEBUG] Emission at max accuracy round: {emission}")
                return emission
            except Exception as e:
                print(f"[ERROR] Failed to load emission or accuracy: {e}")
                return None

        def get_final_accuracy(file_path):
            try:
                with open(file_path, 'rb') as f:
                    history = pickle.load(f)
                if hasattr(history, "metrics_centralized") and "accuracy" in history.metrics_centralized:
                    acc_list = history.metrics_centralized["accuracy"]
                    return max([val[1] for val in acc_list]) if acc_list else None
                return None
            except:
                return None

        _, accuracy_files = load_flat_folder_with_cumulative(folder)

        # Plot accuracy on figure 1
        for method, path in accuracy_files.items():
            try:
                with open(path, "rb") as f:
                    hist = pickle.load(f)
                if not hasattr(hist, "metrics_centralized") or "accuracy" not in hist.metrics_centralized:
                    continue
                acc_list = hist.metrics_centralized["accuracy"]
                rounds = [t for t, _ in acc_list[:100]]
                acc = [a * 100 for _, a in acc_list[:100]]
                name = simplify_label(method)
                color = custom_colors.get(name, 'gray')
                linestyle = '--' if 'random' in name.lower() else '-'
                ax1.plot(rounds, acc, label=name, color=color, linewidth=2.2, linestyle=linestyle)
            except Exception as e:
                print(f"[WARN] Could not load {path}: {e}")

        # Final formatting for accuracy plot
        ax1.set_ylim(20, 72)
        ax1.set_xticks(np.arange(0, 101, 25))
        ax1.set_yticks(np.arange(30, 72, 20))
        ax1.tick_params(labelsize=33, width=2, length=6)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylabel("Accuracy (%)", fontsize=35, labelpad=10)
        ax1.set_xlabel("Training Rounds", fontsize=35)

        # Get handles and labels
        handles, labels = ax1.get_legend_handles_labels()

        # Sort alphabetically by label
        sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*sorted_handles_labels)

        # Add sorted legend
        ax1.legend(sorted_handles, sorted_labels, loc="upper center", ncol=2,
                   fontsize=30, bbox_to_anchor=(0.6, 0.4), frameon=False)

        # Plot emissions/accuracy on figure 2
        cumulative_files, accuracy_files = load_flat_folder_with_cumulative(folder)
        methods, emissions_to_max_acc, final_acc = [], [], []
        valid_methods = sorted(set(cumulative_files) & set(accuracy_files))
        for method in valid_methods:
            emission = get_emission_at_max_accuracy(accuracy_files[method], cumulative_files[method])
            acc = get_final_accuracy(accuracy_files[method])
            if emission is not None and acc is not None:
                label = simplify_label(method)
                methods.append(label)
                emissions_to_max_acc.append(emission / 1000)
                final_acc.append(acc * 100)

        methods, emissions_to_max_acc, final_acc = zip(*sorted(zip(methods, emissions_to_max_acc, final_acc)))
        x = np.arange(len(methods))
        width = 0.5  # Thicker bars
        bars1 = ax2b.bar(x, emissions_to_max_acc, width, color="gray", alpha=0.5)
        bars2 = ax2.plot(x, final_acc, marker='o', markersize=15, color="cornflowerblue", linewidth=2.5)[0]
        fig2.legend(
            [bars2, bars1[0]],
            ["Max Accuracy", "Emissions to Max Accuracy"],
            loc='upper center',
            bbox_to_anchor=(0.47, 1.27),
            ncol=2,
            fontsize=35,
            frameon=False
        )

        ax2.set_ylabel("Max Accuracy (%)", fontsize=37, labelpad = 20 )#color="cornflowerblue")
        ax2.tick_params(axis='y',  labelsize=35) #labelcolor="cornflowerblue"
        ax2b.set_ylabel("kgCO$_2$eq", fontsize=37, labelpad=20)
        ax2.set_ylim(20, 70)
        ax2.set_yticks([30, 50, 70])
        ax2b.set_ylim(0, 510)
        ax2b.set_yticks([0, 250, 500])

        # X-axis
        ax2.tick_params(axis='x', pad=20)  # Add space below labels
        ax2b.tick_params(axis='x', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontsize=35)  # Set method label font size

        # Y-axis
        ax2b.tick_params(axis='y', labelsize=35)  # Right y-axis (emissions)

        ax2.set_xlabel("Method", fontsize=37)

        # Title for the emissions/accuracy plot
        #fig2.text(0.5, 1.03, f"(b) Emissions to Max Accuracy for {folder_name}", ha='center', va='bottom', fontsize=32)

        # Save separate PDFs for each subplot if `split` is True
        # Save separate PDFs for each subplot if `split` is True
        if split:
            os.makedirs("plots", exist_ok=True)
            fig1.savefig(f"plots/{folder_name}_accuracy_subplot_{i + 1}.pdf", format='pdf', bbox_inches="tight")
            print(f"[INFO] Saved to plots/{folder_name}_accuracy_subplot_{i + 1}.pdf")

            fig2.savefig(f"plots/{folder_name}_emissions_subplot_{i + 1}.pdf", format='pdf', bbox_inches="tight")
            print(f"[INFO] Saved to plots/{folder_name}_emissions_subplot_{i + 1}.pdf")

            plt.close(fig1)
            plt.close(fig2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", help="Folders containing _Oort.pkl and _OortCA.pkl files")
    parser.add_argument("--split", action="store_true", help="Save each subplot as a separate PDF")
    args = parser.parse_args()

    plot_combined(args.folders, split=args.split)
