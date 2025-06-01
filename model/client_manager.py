import pickle
import random
from typing import Dict, List, Optional
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.common.logger import log
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import os

from electricity_maps.load_availability_and_emissions_data import (
    get_carbon_intensities_for_all_regions, get_next_100_hours_curtailment,
    plot_heatmap, combine_client_data
)

def save_matrix(matrix, name, cfg_name):
    folder_name, file_name = cfg_name.split('_', 1)
    os.makedirs(f"results/{folder_name}", exist_ok=True)
    file_path = f"results/{folder_name}/{file_name}_{name}.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(matrix, f)
    print(f"Matrix {name} saved to {file_path}")


class MyClientManager(SimpleClientManager):
    """Client manager with carbon intensity and availability classification."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.num_clients = self.cfg.num_clients
        self.next_sample = None
        self.client_data = {}
        self.next_client_index = 0
        self.classification_matrix = None
        self.round = 0
        self.selected_clients_matrix = np.zeros((self.cfg.num_rounds, self.cfg.num_clients), dtype=int)
        self.intensity_matrix = None
        self.load_data()


    def set_next_sample(self, next_sample: Optional[List[str]]):
        """Set the next sample of clients manually."""
        if next_sample is not None:
            self.next_sample = [self.clients[cid] for cid in next_sample]
        else:
            self.next_sample = None

    def sample(self, num_clients: int, min_num_clients: Optional[int] = None,
               criterion: Optional[Criterion] = None) -> List[ClientProxy]:
        """
        Sample a number of clients, optionally using pre-set next sample.
        """
        min_num_clients = min_num_clients or num_clients
        self.wait_for(min_num_clients)

        connected_cids = list(self.clients.keys())

        print(f"[DEBUG] Connected clients: {len(connected_cids)}")

        if num_clients > len(connected_cids):
            log(
                "INFO",
                "Sampling failed: available clients (%d) < requested clients (%d).",
                len(connected_cids),
                num_clients
            )
            return []

        if self.next_sample:
            sampled_cids = [client.cid for client in self.next_sample]  # Extract client IDs from next_sample
            sampled_clients = self.next_sample
        else:
            sampled_cids = random.sample(connected_cids, num_clients)
            sampled_clients = [self.clients[cid] for cid in sampled_cids]

        # Update the selection matrix for the current round
        for cid in sampled_cids:
            client_index = int(cid)
            self.selected_clients_matrix[self.round-1, client_index] = 1

        return sampled_clients

    def sample_from(self, num_clients: int, list_ids: List[str]) -> List[ClientProxy]:
        """Sample a number of clients from a specific list of IDs."""
        sampled_cids = list_ids if num_clients >= len(list_ids) else random.sample(list_ids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

    def load_data(self):
        if self.cfg.add_curtailment_data:
            self.load_with_curtailment()
        else:
            self.load_intensities_data()

    def load_intensities_data(self):
        carbon_intensities = get_carbon_intensities_for_all_regions(limit=self.cfg.num_clients)
        if not carbon_intensities:
            print("[WARNING] No carbon intensity probing available.")
            return

        sorted_intensities = dict(sorted(carbon_intensities.items()))
        self.client_data = {}
        for client_id, (region, intensity_values) in enumerate(sorted_intensities.items()):
            self.client_data[str(client_id)] = {
                "region": region,
                "intensity_data": intensity_values
            }

        self.fill_intensity_matrix()
        save_matrix(self.intensity_matrix, "intensity_matrix", self.cfg.name)
        print("[INFO] Carbon intensity data loaded and matrix filled.")

    def load_with_curtailment(self):
        curtailment_data = get_next_100_hours_curtailment(limit=self.cfg.num_clients)
        carbon_intensities = get_carbon_intensities_for_all_regions(limit=self.cfg.num_clients)

        if not curtailment_data or not carbon_intensities:
            print("[WARNING] Missing curtailment or carbon intensity probing.")
            return

        sorted_curtailment = dict(sorted(curtailment_data.items()))
        sorted_intensities = dict(sorted(carbon_intensities.items()))

        curtailment_threshold = 470
        combined_data = combine_client_data(
            sorted_curtailment,
            sorted_intensities,
            curtailment_threshold
        )

        self.client_data = {}
        for client_id, (region, values) in enumerate(combined_data.items()):
            self.client_data[str(client_id)] = {
                "region": region,
                "intensity_data": values
            }

        self.fill_intensity_matrix()
        self.fill_classification_matrix()
        save_matrix(self.intensity_matrix, "intensity_matrix", self.cfg.name)

    def fill_classification_matrix(self):
        if self.intensity_matrix is None:
            print("[WARNING] Intensity matrix is empty. Cannot fill classification matrix.")
            return

        num_clients, num_rounds = self.intensity_matrix.shape
        self.classification_matrix = np.zeros((num_clients, num_rounds), dtype=int)

        for round_idx in range(num_rounds):
            round_intensities = self.intensity_matrix[:, round_idx]
            mean_intensity = np.mean(round_intensities[round_intensities > 0])  # Exclude curtailment (0 values)

            for client_id in range(num_clients):
                intensity = round_intensities[client_id]

                if intensity == 0:
                    self.classification_matrix[client_id, round_idx] = 1  # Green (curtailment available)
                elif intensity < mean_intensity:
                    self.classification_matrix[client_id, round_idx] = 1  # Green (below mean intensity)
                elif intensity > 0:
                    self.classification_matrix[client_id, round_idx] = 2  # Brown (above mean intensity)

    def fill_intensity_matrix(self):
        """
        Fill the intensity matrix with combined probing.
        Values are 0 if curtailment energy is available, otherwise actual intensity values.
        """
        if not self.client_data:
            print("Client probing is empty. Please load the probing first.")
            return

        num_clients = len(self.client_data)
        num_rounds = self.cfg.num_rounds
        self.intensity_matrix = np.zeros((num_clients, num_rounds), dtype=float)

        for client_id, data in self.client_data.items():
            client_index = int(client_id)
            intensity_values = data.get("intensity_data", [])

            if len(intensity_values) < num_rounds:
                print(f"[WARNING] Client {client_id} has fewer intensity rounds than expected.")
                intensity_values = np.pad(
                    intensity_values,
                    (0, num_rounds - len(intensity_values)),
                    constant_values=0
                )

            self.intensity_matrix[client_index, :] = intensity_values[:num_rounds]

def plot_combined_intensity_heatmap_and_lines(intensity_matrix, intensity_data_dict, output_path="plots/intensity_combined_aligned_final.pdf"):
    intensity_matrix = intensity_matrix[:, :100]
    intensity_data_dict = {k: v[:100] for k, v in intensity_data_dict.items()}

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[40, 1], height_ratios=[1.5, 1], wspace=0.017, hspace=0.1)

    # ---- Top: Heatmap ----
    ax1 = fig.add_subplot(gs[0, 0])
    semi_transparent_green =  [0.85,0.92,1] #  [0.6, 0.98, 0.6, 0.6]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_to_orange_darker",
        [semi_transparent_green, "gray", "lightgray", "#E69F00"]
    )
    im = ax1.imshow(intensity_matrix, aspect="auto", cmap=cmap, interpolation="nearest", origin="upper")

    cax = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("gCO$_2$/kWh", fontsize=20)

    num_clients, num_rounds = intensity_matrix.shape
    ax1.set_ylabel("Client ID", fontsize=20)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.set_yticks([0, num_clients // 2, num_clients - 1])
    ax1.set_yticklabels([0, num_clients // 2, num_clients], fontsize=18)
    ax1.set_title("Assigned Carbon Intensities", fontsize=21)

    # ---- Bottom: Line Plot ----
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2_right = ax2.twinx()

    selected_regions = list(intensity_data_dict.keys())[:3]
    color_palette = ["green", "#E69F00", "gray"]

    for i, (region, y) in enumerate(intensity_data_dict.items()):
        x = np.arange(len(y))
        color = color_palette[i % len(color_palette)]
        ax2_right.plot(x, y, color=color, linewidth=1.5, label=region)
        ax2_right.annotate(
            f"ID = {i}",
            xy=(0, y[0]),
            xytext=(-10, 0),
            textcoords='offset points',
            fontsize=15,
            color=color,
            va='center',
            ha='right',
            clip_on=False
        )

    ax2_right.set_ylabel("gCO$_2$/kWh", fontsize=20)
    ax2_right.yaxis.set_label_coords(1.12, 0.5)  # Adjust x=1.12 until it aligns with colorbar label
    ax2_right.tick_params(axis='y', labelsize=18)
    ax2.set_yticks([])

    ax2.set_xlim(left=0)
    ax2_right.set_xlim(left=0)
    ax2.set_xlabel("Hour", fontsize=20)
    ax2.set_xticks(range(0, 101, 20))  # Show ticks every 20 hours
    ax2.set_xticklabels([str(i) for i in range(0, 101, 20)], fontsize=18)

    # Add region legend below the plot
    ax2_right.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=len(selected_regions),
        fontsize=18,
        frameon=False
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Final aligned plot saved to {output_path}")

def main():
    from electricity_maps.load_availability_and_emissions_data import get_carbon_intensities_for_all_regions

    # Example configuration
    class Config:
        num_rounds = 100
        num_clients = 30
        name = "Test_Run"
        add_curtailment_data = False

    cfg = Config()
    manager = MyClientManager(cfg=cfg)
    manager.load_data()

    if manager.intensity_matrix is None:
        print("[WARNING] Intensity matrix is empty.")
        return

    intensity_data_dict = get_carbon_intensities_for_all_regions(limit=3)
    plot_combined_intensity_heatmap_and_lines(
        intensity_matrix=manager.intensity_matrix,
        intensity_data_dict=intensity_data_dict,
        output_path=f"results/{cfg.name}_intensity_combined.pdf"
    )

if __name__ == "__main__":
    main()

