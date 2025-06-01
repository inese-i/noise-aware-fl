import random
from typing import List
import os
import pickle

class CarbonBudget:
    def __init__(self, carbon_intensities_matrix, total_budget, num_rounds, cfg_name):
        self.carbon_intensities_matrix = carbon_intensities_matrix
        self.total_budget = total_budget
        self.num_rounds = num_rounds
        self.budget_queue = self._initialize_budget_queue()
        self.total_used_carbon = 0  # Track total carbon used
        self.cfg_name = cfg_name
        self.cumulative_usage = []  # Track cumulative usage up to each round

    def _initialize_budget_queue(self):
        per_round_budget = self.total_budget / self.num_rounds
        return [per_round_budget] * self.num_rounds

    def deduct_from_budget(self, round_idx, deduction):
        if 0 <= round_idx < self.num_rounds:
            self.budget_queue[round_idx] -= deduction
            self.total_used_carbon += deduction
            if self.budget_queue[round_idx] < 0:
                self.budget_queue[round_idx] = 0

    def update_unused_budget_for_round(self, round_idx):
        if round_idx < self.num_rounds - 1:
            unused_budget = self.budget_queue[round_idx]
            self.budget_queue[round_idx + 1] += unused_budget
            self.budget_queue[round_idx] = 0

    def get_budget_for_round(self, round_idx):
        if 0 <= round_idx < self.num_rounds:
            return self.budget_queue[round_idx]
        return None

    def get_total_remaining_budget(self):
        return sum(self.budget_queue)

    def print_total_used_carbon(self):
        print(f"[INFO] Total Carbon Used Across All Rounds: {self.total_used_carbon}")

    def calculate_usage_rate(self):
        usage_rate = (self.total_used_carbon / self.total_budget) * 100
        return usage_rate

    def print_usage_report(self):
        total_remaining = self.get_total_remaining_budget()
        usage_rate = self.calculate_usage_rate()
        print("[INFO] Carbon Budget Report:")
        print(f"  Total Budget: {self.total_budget}")
        print(f"  Total Used: {self.total_used_carbon}")
        print(f"  Total Remaining: {total_remaining}")
        print(f"  Usage Rate: {usage_rate:.2f}%")
        self.save_total_used_carbon(cfg_name=self.cfg_name)
        self.save_cumulative_usage()  # Save the cumulative usage at the end

    def update_cumulative_usage(self):
        """
        Update the list of cumulative carbon usage up to the current round.
        """
        self.cumulative_usage.append(self.total_used_carbon)

    def save_cumulative_usage(self):
        """
        Save cumulative usage data to a .pkl file.
        """
        folder_name, file_name = self.cfg_name.split('_', 1)
        os.makedirs(f"results/{folder_name}", exist_ok=True)
        file_path = f"results/{folder_name}/{file_name}_cumulative_usage.pkl"

        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.cumulative_usage, f)
        except Exception as e:
            print(f"[ERROR] Could not save cumulative usage: {e}")

    def select_random_clients(self, server_round: int, eligible_clients: List[str], num_clients: int) -> List[str]:
        selected_clients = []

        remaining_budget = self.get_budget_for_round(server_round)

        budget_eligible_clients = [
            client_id for client_id in eligible_clients
            if self.carbon_intensities_matrix[int(client_id), server_round] <= remaining_budget
        ]
        print(f"[DEBUG] Eligible clients: {eligible_clients}")
        print(f"[DEBUG] Budget-eligible clients: {budget_eligible_clients}")

        used_carbon = 0.0

        while len(selected_clients) < num_clients:
            if not budget_eligible_clients:
                print("[WARNING] No more budget-eligible clients available. Exiting loop.")
                break

            client_id = random.choice(budget_eligible_clients)
            intensity = self.carbon_intensities_matrix[int(client_id), server_round]

            if intensity <= remaining_budget:
                selected_clients.append(client_id)
                used_carbon += intensity
                remaining_budget -= intensity
                budget_eligible_clients.remove(client_id)
            else:
                budget_eligible_clients.remove(client_id)

        # Deduct carbon and update central budget tracker
        print(f"[DEBUG] Total carbon used this round: {used_carbon:.2f}")
        self.deduct_from_budget(server_round, used_carbon)
        self.update_unused_budget_for_round(server_round)

        print(f"[DEBUG] Final selected clients for round {server_round}: {selected_clients}")
        return selected_clients

    def save_total_used_carbon(self, cfg_name):
        folder_name, file_name = cfg_name.split('_', 1)
        os.makedirs(f"results/{folder_name}", exist_ok=True)
        file_path = f"results/{folder_name}/{file_name}_carbon_used.pkl"

        with open(file_path, 'wb') as f:
            pickle.dump(self.total_used_carbon, f)

        print(f"Carbon usage saved to {file_path}")
