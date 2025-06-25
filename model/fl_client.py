from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import numpy as np
import flwr as fl
import torch.nn.functional as F
import torch
from datasets import preview_cifar10, preview_tiny_imagenet

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, device, trainloader, valloader, cid, trainloader2=None):
        super().__init__()
        self.net = net
        self.client_id = int(cid)
        self.trainloader = trainloader
        self.trainloader2 = trainloader2
        self.valloader = valloader
        self.device = device
        self.optimizer = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        model_set_parameters(self.net, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round"))
        lr = config.get("lr", 0.001)
        
        
        if server_round == 1:
            local_epochs = 2
        else:
            local_epochs = config.get("local_epochs", 1) if config.get("local_epochs") is not None else 1
            
        weight_decay = config.get("weight_decay", 0) if config.get("weight_decay") is not None else 0

        if self.trainloader2 and config.get("start_round") <= server_round <= config.get("end_round"):
            self.trainloader = self.trainloader2


        # if server_round == 1 and self.client_id == 1:
        #     preview_cifar10(self.trainloader)

        # if server_round == 1 and self.client_id == 1:
        #     preview_mnist(self.trainloader)

        # if server_round == 1 and self.client_id == 1:
        #     preview_tiny_imagenet(self.trainloader)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        if config.get("probing_round") and config.get("server_round")==1:
            avg_loss, avg_grad, loss_utility = train_with_fgn(
                model=self.net, device=self.device, train_loader=self.trainloader,
                optimizer=self.optimizer, local_epochs=local_epochs
            )

            return self.get_parameters(config), len(self.trainloader.dataset), {
                'loss': avg_loss, 'avg_grad': avg_grad, 'eval': 0, 'loss_utility': loss_utility,
            }

        avg_loss, loss_utility = train(self.net, self.device, self.trainloader, self.optimizer, local_epochs)

        return self.get_parameters(config), len(self.trainloader.dataset), {'loss': avg_loss,
                                                                            'loss_utility': loss_utility,
                                                                            'eval': 0}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        avg_loss, accuracy = test(self.net, self.device, self.valloader)
        num_samples = len(self.valloader.dataset)
        metrics = {"eval_accuracy": accuracy}
        print(f"{self.client_id} ID - eval acc : {accuracy}")
        return avg_loss, num_samples, metrics


def generate_client_fn(net, trainloaders, valloaders, device, trainloaders2=None, damaged_clients=0):
    def client_fn(cid: str) -> FlowerClient:
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        if trainloaders2 and int(cid) < damaged_clients:
            return FlowerClient(net, device, trainloader, valloader, cid, trainloader2=trainloaders2[int(cid)])
        return FlowerClient(net, device, trainloader, valloader, cid)
    return client_fn

def train_with_fgn(model, device, train_loader, optimizer, local_epochs=1):

    model.train()
    total_loss, total_samples, total_grad_norm_sq = 0.0, 0, 0.0
    sample_loss = []

    for _ in range(local_epochs):
        running_loss, epoch_samples, grad_norm = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            sample_loss.append(loss)
            loss.backward()

            running_loss += loss.item() * target.size(0)
            epoch_samples += target.size(0)
            optimizer.step()
            # L2 norm squared (per batch)
            temp_norm = sum(
                param.grad.detach().data.norm(2).item() ** 2 for param in model.parameters() if param.grad is not None
            )
            grad_norm += temp_norm

        total_loss += running_loss
        total_samples += epoch_samples
        # Utility/norm per epoch, all samples
        grad_norm = epoch_samples * (grad_norm / epoch_samples) ** 0.5  # Equivalent to |B_i| * sqrt(avg grad norm)
        total_grad_norm_sq += grad_norm

    avg_loss = total_loss / total_samples
    avg_grad_norm = total_grad_norm_sq / local_epochs

    return avg_loss, avg_grad_norm, statistical_utility(sample_loss)

def train(model, device, train_loader, optimizer, local_epochs=1):
    model.train()
    total_loss, total_samples = 0.0, 0
    sample_loss = []

    for _ in range(local_epochs):
        running_loss, epoch_samples = 0.0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            sample_loss.append(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * target.size(0)
            epoch_samples += target.size(0)

        total_loss += running_loss
        total_samples += epoch_samples

    avg_loss = total_loss / total_samples

    return avg_loss, statistical_utility(sample_loss)

def statistical_utility(sample: list):
    """Statistical utility as defined in Oort"""
    squared_sum = sum([torch.square(l) for l in sample]).item()
    return len(sample) * np.sqrt(1 / len(sample) * squared_sum)

def test(net, device, testloader):
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        avg_loss = 0.0  # Default value if the dataset is empty
    else:
        avg_loss = loss / len(testloader.dataset)

    if total == 0:
        accuracy = 0.0  # Default value if no predictions were made
    else:
        accuracy = correct / total
    return avg_loss, accuracy

def model_set_parameters(net, parameters):
    state_dict = OrderedDict()
    for k, v in zip(net.state_dict().keys(), parameters):
        if v is None or (isinstance(v, list) and not v):
            print(f"Parameter for {k} is empty or None")
        else:
            v = torch.tensor(v, dtype=torch.int64 if 'num_batches_tracked' in k else None)
            state_dict[k] = v
    try:
        net.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading state dictionary: {e}")



