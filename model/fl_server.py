# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""
from __future__ import annotations
import os
import pickle
import timeit
from logging import DEBUG, INFO
from typing import Dict, Optional, Any
import numpy as np
import torch
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.history import History
from flwr.server.server import FitResultsAndFailures, EvaluateResultsAndFailures, fit_clients, evaluate_clients
from flwr.server.strategy import Strategy
from typing import List, Tuple
from flwr.common import Metrics
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from client_manager import MyClientManager, save_matrix
from fl_client import model_set_parameters, test
from flwr.common import Parameters, Scalar
from selection_strategy import SelectionStrategy

summary_writers = {}
def get_summary_writer(client_id: int):
    if client_id not in summary_writers:
        summary_writers[client_id] = SummaryWriter(log_dir=f"runs/client_{client_id}")
    return summary_writers[client_id]

class MyServer(Server):
    """Flower server."""
    def __init__(self, *,
                 strategy: Strategy,
                 selection_strategy: SelectionStrategy,
                 writer: SummaryWriter,
                 testloader: Any,
                 model: torch.nn.Module,
                 device: torch.device,
                 config: DictConfig,
                 ) -> None:
        super(MyServer, self).__init__(client_manager=MyClientManager(config), strategy=strategy)
        self.cfg = config
        self.testloader = testloader
        self.device = device
        self.model = model
        self.writer = writer
        self.selected_clients_matrix = np.zeros((self.cfg.num_rounds, self.cfg.num_clients), dtype=int)
        self.clients_measurements = np.zeros(
            self.cfg.num_clients,
            dtype=[('avg_grad', 'f4'),
                   ('selection', 'f4'),
                   ('loss_utility', 'f4')]
        )
        self.selection_count = np.zeros(self.cfg.num_clients)
        # Share matrices with the selection strategy
        if isinstance(selection_strategy, tuple):
            self.selection_strategy = selection_strategy[0]
        else:
            self.selection_strategy = selection_strategy
        self._client_manager: MyClientManager
        self.selection_strategy.preview_and_initialize_budget(self._client_manager)
        self.selection_strategy.clients_measurements = self.clients_measurements


    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds with time tracking."""
        history = History()
        self._client_manager: MyClientManager

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)

        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            round_start_time = timeit.default_timer()
            self._client_manager: MyClientManager
            self._client_manager.set_next_sample(None)
            self._client_manager.round = current_round
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

            # Round completed
            round_end_time = timeit.default_timer()
            round_duration = round_end_time - round_start_time
            log(INFO, "Round %s finished in %s seconds", current_round, round_duration)
            self.writer.add_scalar("round_time", round_duration, current_round)

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s seconds", elapsed)

        # save results
        save_matrix(self.selected_clients_matrix, "selected_clients", self.cfg.name)
        folder_name, file_name = self.cfg.name.split('_', 1)
        os.makedirs(f"results/{folder_name}", exist_ok=True)
        file_path = f"results/{folder_name}/{folder_name}_{file_name}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(history, f)
        return history

    def fit_round(
            self,
            server_round: int,
            timeout: Optional[float]
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Use the selection strategy to compute the next sample of clients
        self._client_manager: MyClientManager
        self.selection_strategy.set_next_sample(server_round, self._client_manager)

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None

        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(DEBUG, "fit_round %s received %s results and %s failures", server_round, len(results), len(failures))

        client_gradients = []  # Will store tuples like (client_id, avg_grad)

        for client_proxy, result in results:
            cid = int(client_proxy.cid)
            writer = get_summary_writer(client_id=cid)
            self.selected_clients_matrix[server_round - 1, cid] = 1

            self.clients_measurements[cid]['selection'] = 1
            self.selection_count[cid] += 1

            if result.metrics.get("avg_grad") is not None:
                avg_grad = result.metrics["avg_grad"]
                self.clients_measurements[cid]['avg_grad'] = avg_grad
                print(f"[DEBUG] Client {cid} - Gradient Norm (avg_grad): {avg_grad}")
                client_gradients.append((cid, avg_grad))

            if result.metrics.get("loss_utility") is not None:
                loss_utility = result.metrics["loss_utility"]
                self.clients_measurements[cid]['loss_utility'] = loss_utility
                print(f"[DEBUG] Client {cid} - Loss Utility: {loss_utility}")

        if client_gradients:  # Check if any gradients are available
            client_gradients_sorted = sorted(client_gradients, key=lambda x: x[1])
            print("\nSorted Clients by Gradient Norm:")
            for cid, avg_grad in client_gradients_sorted:
                print(f"Client ID: {cid}, Average Gradient Norm: {avg_grad}")
        else:
            print("\nNo Gradient Norms Available for Sorting.")

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)
        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def evaluate_round(
            self,
            server_round: int,
            timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients"""

        self._client_manager: MyClientManager
        self._client_manager.set_next_sample(None)

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        if server_round == self.cfg.num_rounds:
            accuracies = {}
            for client_proxy, result in results:
                cid = int(client_proxy.cid)  # Convert client ID to integer
                accuracy = result.metrics.get("eval_accuracy")  # Extract accuracy
                accuracies[cid] = accuracy

        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result

        # Optionally log aggregated metrics
        if metrics_aggregated.get("avg_eval_accuracy") is not None:
            log(
                INFO,
                "evaluate_round %s: avg_eval_accuracy=%s",
                server_round,
                metrics_aggregated["avg_eval_accuracy"],
            )

        self._client_manager: MyClientManager
        self._client_manager.set_next_sample(None)

        return loss_aggregated, metrics_aggregated, (results, failures)


def get_on_fit_config_fn(config: DictConfig):
    """Return function that prepares config fort training with fit() to send to clients."""
    def on_fit_config_fn(server_round: int):
        return {
            "lr": config.config_fit.lr,
            "local_epochs": config.config_fit.local_epochs,
            "weight_decay": config.config_fit.weight_decay,
            "server_round": server_round,
            "start_round": config.experiments.start_round,
            "end_round": config.experiments.until_round,
            "probing_round": config.probing_round,
        }
    return on_fit_config_fn

def get_on_evaluate_config_fn (config: DictConfig):
    """Function used to configure validation on client."""
    def eval_config_fn(server_round: int):
        return {
            "server_round": server_round,
            "start_round": config.experiments.start_round,
            "end_round": config.experiments.until_round,
        }
    return eval_config_fn

def get_evaluate_fn(model, device, testloader, tensor_writer):
    """Return an evaluation function for server-side evaluation."""
    def evaluate_fn(server_round: int, parameters, config):
        model_set_parameters(model, parameters)
        loss, accuracy = test(model, device, testloader)
        tensor_writer.add_scalar("server accuracy on testset", accuracy, server_round)
        return loss, {"accuracy": accuracy}
    return evaluate_fn

def get_fit_metrics_aggregation_fn(tensor_writer, trainset_size, config):
    def fit_metrics_aggregation_fn(
        fit_metrics: list[tuple[int, dict[str, bool | bytes | float | int | str]]],
    ) -> dict[str, bool | bytes | float | int | str]:
        server_round = fit_metrics[0][1].get('server_round', None)
        averaged_loss = sum(item[0] * item[1]['loss'] / trainset_size for item in fit_metrics)
        tensor_writer.add_scalar("averaged_loss", averaged_loss, server_round)

        return {"averaged_loss": averaged_loss}

    return fit_metrics_aggregation_fn

def get_server_summary_writer():
    log_dir = f"runs/server"
    return SummaryWriter(log_dir)
