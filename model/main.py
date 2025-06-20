import argparse
import os
import numpy as np
import traceback
import torch
import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from datasets import get_dataloaders
from model import Net, CIFAR100Net
from results.plot_distributions import (
    get_class_distribution, plot_class_distribution_all_clients,
)
from fl_client import generate_client_fn
from fl_server import (
    MyServer,
    get_on_fit_config_fn,
    get_on_evaluate_config_fn,
    get_fit_metrics_aggregation_fn,
    get_server_summary_writer,
    get_evaluate_fn,
)
from selection_strategy import MySelectionStrategy

def get_model(dataset_name=None):
    np.random.seed(42)
    if dataset_name is not None:
        if 'cifar100' in dataset_name:
            return CIFAR100Net()
        if 'cifar10' in dataset_name:
            return Net()

    return Net()  # fallback if dataset_name is None or not recognized

@hydra.main(config_path="conf", config_name="base", version_base=None)
def run_config(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Select model based on dataset
    model = get_model(cfg.dataset)
    print(f"[DEBUG] Model created: {type(model)} for dataset: {cfg.dataset}")

    trainloaders, trainloaders2, valloaders, testloader = get_dataloaders(
        cfg.dataset, cfg.num_clients, cfg.batch_size, cfg.alpha,
        data_damage=cfg.experiments.data_damage,
        noise_std=cfg.experiments.noise_std
    )

    # Get num_classes from DATASET_LOADERS
    from datasets import DATASET_LOADERS
    base_name = cfg.dataset.split('_')[0]
    num_classes = DATASET_LOADERS[base_name]['num_classes']

    # Sanity check: ensure all labels are in the correct range for this dataset
    for cid, train_loader in enumerate(trainloaders):
        for _, targets in train_loader:
            for label in targets:
                assert 0 <= label.item() < num_classes, f"Label {label.item()} out of bounds for dataset with {num_classes} classes (client {cid})"
    
    # Sanity check: ensure all labels are in the correct range for the test set as well
    for _, targets in testloader:
        for label in targets:
            assert 0 <= label.item() < num_classes, f"[TEST] Label {label.item()} out of bounds for dataset with {num_classes} classes"

    class_distributions = {}
    for cid, train_loader in enumerate(trainloaders):
        class_distributions[cid] = get_class_distribution(train_loader, num_classes=num_classes)

    #plot_class_distribution_all_clients(class_distributions, cfg.name)#

    client_num_samples = {cid: len(train_loader.dataset) for cid, train_loader in enumerate(trainloaders)}
    num_defect_clients = cfg.experiments.num_defect_clients if cfg.experiments.num_defect_clients is not None else 0
    num_total_samples = sum(len(loader.dataset) for loader in trainloaders)

    client_fn = generate_client_fn(
        model, trainloaders, valloaders, device, trainloaders2, num_defect_clients)

    server_writer = get_server_summary_writer()
    fit_metrics_aggregation_fn = get_fit_metrics_aggregation_fn(
        server_writer, num_total_samples, cfg
    )
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.num_clients_per_round / cfg.num_clients,
        min_fit_clients=1,
        fraction_evaluate=0,
        on_fit_config_fn=get_on_fit_config_fn(cfg),
        on_evaluate_config_fn=get_on_evaluate_config_fn(cfg),
        evaluate_fn=get_evaluate_fn(model, device, testloader, tensor_writer=server_writer),
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    client_resources = {"num_gpus": 1} if device.type == "cuda" else None

    selection_strategy = MySelectionStrategy(cfg, sample_distribution=client_num_samples)

    server = MyServer(
        strategy=strategy,selection_strategy=selection_strategy , writer=server_writer, model=model, device=device,
        testloader=testloader, config=cfg)

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            server=server,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_resources=client_resources
        )
    except Exception as e:
        print("An error occurred during the Flower simulation:")
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run configurations with Hydra")
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Specify a single configuration file to run (e.g., conf/base.yaml).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all configurations in the default conf directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run all configurations in the specified folder (relative to conf/).",
    )
    args = parser.parse_args()

    # Default configuration directory
    conf_dir = "conf"

    # Option 1: Run all configurations in the default conf directory
    if args.run_all:
        config_names = [f.replace(".yaml", "") for f in os.listdir(conf_dir) if f.endswith(".yaml")]
        for config_name in config_names:
            GlobalHydra.instance().clear()
            hydra.initialize(config_path=conf_dir)
            cfg = hydra.compose(config_name=config_name)
            print(f"Running training for config: {config_name}")
            run_config(cfg)

    # Option 2: Run all configurations in a specified folder within the conf directory
    elif args.name:
        folder_path = os.path.join(conf_dir, args.name)
        if not os.path.isdir(folder_path):
            print(f"[ERROR] The specified folder does not exist: {folder_path}")
            exit(1)

        config_names = [f.replace(".yaml", "") for f in os.listdir(folder_path) if f.endswith(".yaml")]
        for config_name in config_names:
            GlobalHydra.instance().clear()
            hydra.initialize(config_path=folder_path)
            cfg = hydra.compose(config_name=config_name)
            print(f"Running training for config: {config_name}")
            run_config(cfg)

    # Option 3: Run a specific configuration file
    elif args.config_name:
        config_name = os.path.splitext(os.path.basename(args.config_name))[0]
        GlobalHydra.instance().clear()
        hydra.initialize(config_path=conf_dir)
        cfg = hydra.compose(config_name=config_name)
        print(f"Running training for config: {config_name}")
        run_config(cfg)

    # Option 4: Run the default (base) configuration
    else:
        print("Running default configuration: base")
        run_config()  # `base.yaml` is used by default
