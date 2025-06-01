import math
from collections import OrderedDict
from logging import DEBUG, INFO, WARNING, log
from random import Random
from typing import List
import numpy as np


class CarbonAwareOortSelector:
    """Oort Selector adapted for balancing utility and carbon emissions with budget constraints."""

    def __init__(self, sample_seed=233, alpha=1.0, beta=1.0, gamma=0.0):
        self.totalArms = OrderedDict()
        self.training_round = 0

        self.exploration = 0.9
        self.decay_factor = 0.95
        self.exploration_min = 0.2
        self.alpha = alpha
        self.beta = beta
        self.curtailment_bonus = gamma

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()

        self.pacer_step = 20
        self.pacer_delta = 5
        self.blacklist_rounds = -1
        self.blacklist_max_len = 0.3
        self.clip_bound = 0.98

        self.sample_window = 5.0
        self.exploreClients = []
        self.exploitClients = []
        self.successfulClients = set()

        np.random.seed(sample_seed)

    def register_client(self, clientId, size, carbon_intensity=0):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {
                'reward': size,
                'carbon_intensity': carbon_intensity,
                'time_stamp': self.training_round,
                'count': 0,
                'status': True
            }
            self.unexplored.add(clientId)

    def update_client_util(self, clientId, reward, time_stamp=0, carbon_intensity=0):
        if clientId in self.totalArms:
            self.totalArms[clientId]['reward'] = reward
            self.totalArms[clientId]['carbon_intensity'] = carbon_intensity
            self.totalArms[clientId]['time_stamp'] = time_stamp
            self.totalArms[clientId]['count'] += 1
            self.totalArms[clientId]['status'] = True

            self.unexplored.discard(clientId)
            self.successfulClients.add(clientId)

    def get_blacklist(self):
        blacklist = []
        if self.blacklist_rounds != -1:
            sorted_client_ids = sorted(self.totalArms, reverse=True,
                                       key=lambda k: self.totalArms[k]['carbon_intensity'])
            for clientId in sorted_client_ids:
                if self.totalArms[clientId]['carbon_intensity'] > self.clip_bound:
                    blacklist.append(clientId)
                else:
                    break

            predefined_max_len = int(self.blacklist_max_len * len(self.totalArms))
            if len(blacklist) > predefined_max_len:
                log(WARNING, "Selector: Exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]

        return set(blacklist)

    def getTopK(self, numOfSamples, feasible_clients, remaining_budget):
        self.pacer()

        print(f"[DEBUG] Remaining carbon budget: {remaining_budget:.4f}")

        # Collect and sanitize utilities and carbon intensities
        utilities = []
        carbon_costs = []

        for cid in feasible_clients:
            raw_util = self.totalArms[cid].get('reward', 0.0)
            raw_carbon = self.totalArms[cid].get('carbon_intensity', 0.0)

            if raw_util is None:
                print(f"[WARNING] Client {cid} has None utility, setting to 0.0")
                raw_util = 0.0
            if raw_carbon is None:
                print(f"[WARNING] Client {cid} has None carbon intensity, setting to 0.0")
                raw_carbon = 0.0

            utilities.append(raw_util)
            carbon_costs.append(raw_carbon)

        # --- Pre-filter top clients by utility-to-carbon ratio ---
        scored = [
            (i, utilities[i] / (carbon_costs[i] + 1e-6)) for i in range(len(feasible_clients))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scored[:100]]  # adjustable

        clients = [feasible_clients[i] for i in top_indices]
        utilities = [utilities[i] for i in top_indices]
        carbon_costs = [carbon_costs[i] for i in top_indices]

        # --- Optimized DP Backpack with backtracking ---
        def dp_knapsack(clients, utilities, carbon_costs, k, budget):
            n = len(clients)
            scale = 100  # use lower scale for speed
            max_budget = int(budget * scale)
            carbon_scaled = [int(c * scale) for c in carbon_costs]

            dp = [[-1.0] * (max_budget + 1) for _ in range(k + 1)]
            choice = [[-1] * (max_budget + 1) for _ in range(k + 1)]
            dp[0][0] = 0.0

            for i in range(n):
                util = utilities[i]
                cost = carbon_scaled[i]

                for j in range(k, 0, -1):
                    for b in range(max_budget, cost - 1, -1):
                        if dp[j - 1][b - cost] != -1.0:
                            new_util = dp[j - 1][b - cost] + util
                            if new_util > dp[j][b]:
                                dp[j][b] = new_util
                                choice[j][b] = i

            # Find best score
            best_clients = []
            max_util = -1.0
            best_j = best_b = 0
            for j in range(1, k + 1):
                for b in range(max_budget + 1):
                    if dp[j][b] > max_util:
                        max_util = dp[j][b]
                        best_j, best_b = j, b

            # Backtrack
            j, b = best_j, best_b
            while j > 0 and b >= 0:
                i = choice[j][b]
                if i == -1:
                    break
                best_clients.append(clients[i])
                b -= carbon_scaled[i]
                j -= 1

            return best_clients

        selected_clients = dp_knapsack(
            clients, utilities, carbon_costs, numOfSamples, remaining_budget
        )

        total_used = sum(self.totalArms[cid]['carbon_intensity'] or 0.0 for cid in selected_clients)

        print(f"[DEBUG] Final selected clients (total={len(selected_clients)}): {selected_clients}")
        print(f"[DEBUG] Total carbon used: {total_used:.4f} out of budget {remaining_budget:.4f}\n")

        return selected_clients

    def select_participant(self, num_of_clients: int, feasible_clients: List[str], remaining_budget):
        if len([c for c in self.totalArms.values() if c['count'] > 0]) < num_of_clients:
            self.rng.shuffle(feasible_clients)
            client_len = min(num_of_clients, len(feasible_clients))
            clients = feasible_clients[:client_len]
        else:
            clients = self.getTopK(num_of_clients, feasible_clients, remaining_budget)

        self.training_round += 1
        return clients

    def pacer(self):
        self.exploration = max(self.exploration * self.decay_factor, self.exploration_min)
        log(INFO, f"Selector: Exploration rate updated to {self.exploration:.2f}")

    def get_norm(self, values, clip_bound=0.95, thres=1e-4):
        if not values or len(values) == 0:
            print("[WARNING] Empty values list passed to get_norm.")
            return 0, 0, thres, 0, 0

        sorted_values = sorted(values)
        max_value = sorted_values[-1]
        min_value = sorted_values[0] * 0.999
        range_value = max(max_value - min_value, thres)
        avg_value = sum(values) / max(len(values), 1)
        clip_index = min(int(len(sorted_values) * clip_bound), len(sorted_values) - 1)
        clip_value = sorted_values[clip_index]

        return float(max_value), float(min_value), float(range_value), float(avg_value), float(clip_value)
