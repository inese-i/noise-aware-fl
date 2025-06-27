from abc import ABC, abstractmethod
from typing import List
import numpy as np
from omegaconf import DictConfig
from carbon_oort import CarbonAwareOortSelector
from client_manager import MyClientManager
from oort import OortSelector
from budget import CarbonBudget

class SelectionStrategy(ABC):
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.clients_measurements = None
        self.carbon_budget = None
        self.carbon_intensities_matrix = None
        self.metric_threshold = 0
        self.blacklist = set()

    @abstractmethod
    def set_next_sample(self, server_round: int, client_manager: MyClientManager):
        """Choose clients and set as next sample in Client Manager"""

    def get_probing_threshold(self):
        probing_metric = 'loss_utility' #'avg_grad'

        if probing_metric not in self.clients_measurements.dtype.names:
            raise ValueError(f"[ERROR] Probing metric '{probing_metric}' not found in clients_measurements!")

        metric_values = self.clients_measurements[probing_metric]

        if metric_values.size == 0:
            print(f"[WARNING] No values found for probing_metric '{probing_metric}'!")
            self.metric_threshold = 0
        else:
            # Calculate the gradient threshold
            max_grad_norm = np.max(metric_values)
            if self.cfg.threshold_coeff is None:
                raise ValueError("[ERROR] `threshold_coeff` is None — check your configuration.")
            self.metric_threshold = self.cfg.threshold_coeff * max_grad_norm
            print(f"[DEBUG] Maximum gradient norm ({probing_metric}): {max_grad_norm}")
            print(f"[DEBUG] Gradient threshold: {self.metric_threshold}")

    def update_blacklist(self):
        probing_metric = 'loss_utility' # 'avg_grad'

        if probing_metric not in self.clients_measurements.dtype.names:
            raise ValueError(f"[ERROR] Probing metric '{probing_metric}' not found in clients_measurements!")

        grad_values = self.clients_measurements[probing_metric]
        self.blacklist.clear()

        for id, grad in enumerate(grad_values):
            # For CIFAR-100: blacklist clients with HIGH gradient norms (noisy clients)
            if 'cifar100' in self.cfg.dataset:
                if grad > self.metric_threshold:
                    self.blacklist.add(str(id))
            else:
                # For other datasets: blacklist clients with LOW gradient norms
                if grad < self.metric_threshold:
                    self.blacklist.add(str(id))

        print(f"Updated blacklist: {self.blacklist}")

    def preview_and_initialize_budget(self, client_manager: MyClientManager):
        self.carbon_intensities_matrix = client_manager.intensity_matrix

        if self.cfg.carbon_budget:
            self.carbon_budget = CarbonBudget(
                carbon_intensities_matrix=self.carbon_intensities_matrix,
                total_budget=self.cfg.total_carbon_budget,
                num_rounds=self.cfg.num_rounds,
                cfg_name=self.cfg.name,
            )

class MySelectionStrategy(SelectionStrategy):
    def __init__(self, config: DictConfig, sample_distribution):

        super().__init__(config)
        self.sample_distribution = sample_distribution
        self.oort_selector = OortSelector(sample_seed=None)
        self.optimised_selector = CarbonAwareOortSelector(sample_seed=None)
        self.clients_per_round = config.num_clients_per_round

    def set_next_sample(self, server_round: int, client_manager: MyClientManager):
        all_clients = list(client_manager.all().keys())

        # call probing
        if server_round == 1 and self.cfg.probing_round:
            client_ids = all_clients
            print(f"Next sample (probing round): {client_ids}")
            client_manager.set_next_sample(client_ids)
            return

        if server_round == 2 and self.cfg.probing_round :
            self.get_probing_threshold()
            self.update_blacklist()

        print(f"[DEBUG] Current blacklist: {self.blacklist}")
        # Filter clients
        eligible_clients = [client_id for client_id in all_clients if str(client_id) not in self.blacklist]

        if self.cfg.carbon_budget:
            if self.cfg.selection_strategy == "oort":
                client_ids = self.select_with_oort(server_round, client_manager, eligible_clients)
                client_manager.set_next_sample(client_ids)
                self.carbon_budget.update_cumulative_usage()
            elif self.cfg.selection_strategy == "carbon_oort":
                client_ids = self.select_with_selector(server_round, client_manager, eligible_clients)
            else:
                client_ids = self.select_random_clients_with_budget(server_round, eligible_clients)
                self.carbon_budget.update_cumulative_usage()

            if server_round == self.cfg.num_rounds:
                self.carbon_budget.print_usage_report()

        else:
            if self.cfg.selection_strategy == "oort":
                client_ids = self.select_with_oort(server_round, client_manager, eligible_clients)
                client_manager.set_next_sample(client_ids)
                return
            else:
                client_manager.set_next_sample(None)
                print("select FedAvg")
            return

        print(f"Next sample: {client_ids}")
        client_manager.set_next_sample(client_ids)


    def select_random_clients_with_budget(self, server_round: int, eligible_clients) -> List[str]:
        return self.carbon_budget.select_random_clients(
            server_round=server_round - 1,
            eligible_clients=eligible_clients,
            num_clients=self.cfg.num_clients_per_round,
        )

    def select_with_selector(self, server_round: int, client_manager: MyClientManager, eligible_clients):
        clients = eligible_clients
        all_clients_ids = [client for client in clients]
        feasible_clients = all_clients_ids
        print(f"[DEBUG] Server round {server_round}: Eligible clients: {all_clients_ids}")

        # Register clients
        if len(self.optimised_selector.totalArms) == 0:
            for client in all_clients_ids:
                size = self.sample_distribution.get(int(client), 1)
                duration = 1
                carbon_intensity = self.carbon_intensities_matrix[
                    int(client), server_round - 1]  # Use matrix for intensity

                print(f"[DEBUG] Registering client {client} with size={size}, duration={duration}, "
                      f"carbon_intensity={carbon_intensity}")

                self.optimised_selector.register_client(clientId=client, size=size,
                                                        carbon_intensity=carbon_intensity)

        metric_values = self.clients_measurements['loss_utility']
        print(f"[DEBUG] Metric values for update: {metric_values}")

        for client in all_clients_ids:
            client_idx = int(client)
            reward = metric_values[client_idx] if client_idx < len(
                metric_values) else 0
            carbon_intensity = self.carbon_intensities_matrix[client_idx, server_round - 1]  # Use matrix for intensity

            if reward:
                print(f"[DEBUG] Updating client {client} utility with reward={reward}, "
                      f"time_stamp={server_round}, duration=1, carbon_intensity={carbon_intensity}")
                self.optimised_selector.update_client_util(clientId=client, reward=reward,
                                                           time_stamp=server_round,
                                                           carbon_intensity=carbon_intensity)
            else:
                print(f"[DEBUG] Updating client {client} duration only (reward not available).")

        remaining_budget = self.carbon_budget.get_budget_for_round(server_round - 1)
        print(f"[DEBUG] Remaining carbon budget for round {server_round}: {remaining_budget}")

        # Filter feasible clients based on the carbon budget
        budget_feasible_clients = [
            client for client in feasible_clients
            if self.carbon_intensities_matrix[int(client), server_round - 1] <= remaining_budget
        ]
        print(f"[DEBUG] Budget-feasible clients: {budget_feasible_clients}")

        remaining_budget = self.carbon_budget.get_budget_for_round(server_round - 1)
        print(f"[DEBUG] Remaining budget for round {server_round}: {remaining_budget}")

        selected_client_names = self.optimised_selector.select_participant(
            self.clients_per_round,
            feasible_clients=feasible_clients,
            remaining_budget=remaining_budget
        )

        # Calculate total carbon usage for the selected clients
        total_used_carbon = sum(
            self.carbon_intensities_matrix[int(client), server_round - 1]
            for client in selected_client_names
        )
        print(f"[DEBUG] Total carbon used by selected clients: {total_used_carbon}")
        self.carbon_budget.deduct_from_budget(server_round - 1, total_used_carbon)
        self.carbon_budget.update_unused_budget_for_round(server_round - 1)

        client_manager.set_next_sample(selected_client_names)
        print(f"[DEBUG] Updated next sample in client manager with: {selected_client_names}")
        return selected_client_names

    def select_with_oort(self, server_round: int, client_manager: MyClientManager, eligible_clients):
        all_clients_ids = eligible_clients
        feasible_clients = eligible_clients
        print(f"[DEBUG] Feasible clients: {feasible_clients}")

        # Register clients
        if len(self.oort_selector.totalArms) == 0:
            for client in all_clients_ids:
                size = self.sample_distribution.get(int(client), 1)
                timesteps_per_epoch = 1
                carbon_intensity = self.carbon_intensities_matrix[int(client), server_round - 1]
                self.oort_selector.register_client(clientId=client, size=size, duration=timesteps_per_epoch,
                                                   carbon_intensity=carbon_intensity)

        # Update utilities
        metric_values = self.clients_measurements['loss_utility']
        print(f"[DEBUG] Metric values for update: {metric_values}")
        for client in all_clients_ids:
            reward = metric_values[int(client)] if int(client) < len(metric_values) else 0
            if reward:
                carbon_intensity = self.carbon_intensities_matrix[int(client), server_round - 1]
                self.oort_selector.update_client_util(clientId=client, reward=reward,
                                                      time_stamp=server_round, duration=0.3,
                                                      carbon_intensity=carbon_intensity)

            else:
                self.oort_selector.update_duration(clientId=client, duration=0.3)

        # --- Carbon Budget Filtering ---
        remaining_budget = self.carbon_budget.get_budget_for_round(server_round - 1)
        budget_feasible_clients = [
            client for client in feasible_clients
            if self.carbon_intensities_matrix[int(client), server_round - 1] <= remaining_budget
        ]
        print(f"[DEBUG] Budget-feasible clients: {budget_feasible_clients}")
        print(f"[DEBUG] Remaining budget: {remaining_budget}")

        # Select participants using Oort within the budget-feasible set
        selected_client_names = self.oort_selector.select_participant(
            self.clients_per_round,
            feasible_clients=budget_feasible_clients,
            remaining_budget=remaining_budget
        )
        print(f"[DEBUG] Oort selected: {selected_client_names}")

        if self.cfg.carbon_budget:
            total_used_carbon = sum(
                self.carbon_intensities_matrix[int(client), server_round - 1]
                for client in selected_client_names
            )
            print(f"[DEBUG] Total carbon used by selected clients: {total_used_carbon}")

            self.carbon_budget.deduct_from_budget(server_round - 1, total_used_carbon)
            self.carbon_budget.update_unused_budget_for_round(server_round - 1)

        # Set selected clients for the next round
        client_manager.set_next_sample(selected_client_names)
        print(f"[DEBUG] Updated next sample in client manager with: {selected_client_names}")
        return selected_client_names

