import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

def load_cifar10(train_transform, test_transform):
    trainset = CIFAR10("./dataset", train=True, download=True, transform=train_transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=test_transform)
    return trainset, testset

def distribute_indices(class_indices, num_clients, strategy='iid', dirichlet_alpha=10):
    client_indices = [[] for _ in range(num_clients)]
    if strategy == 'iid':
        # Equal samples per class per client
        images_per_class_per_client = min(len(indices) for indices in class_indices.values()) // num_clients
        for class_idx in class_indices:
            np.random.shuffle(class_indices[class_idx])
            for i in range(num_clients):
                start_idx = i * images_per_class_per_client
                end_idx = start_idx + images_per_class_per_client
                client_indices[i].extend(class_indices[class_idx][start_idx:end_idx])

    elif strategy == 'dirichlet':
        # Dirichlet-based allocation with fixed seed
        np.random.seed(42)
        for class_idx in range(10):
            class_data = np.array(class_indices[class_idx])
            proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
            proportions = proportions / proportions.sum()
            client_class_counts = (proportions * len(class_data)).astype(int)
            np.random.shuffle(class_data)
            idx = 0
            for client_id, count in enumerate(client_class_counts):
                client_indices[client_id].extend(class_data[idx:idx + count])
                idx += count
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return client_indices

def create_dataloaders(trainset, testset, client_indices, batch_size):
    trainloaders = []
    valloaders = []
    valset = Subset(trainset, range(len(trainset)))

    for indices in client_indices:
        train_subset = Subset(trainset, indices)
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        trainloaders.append(trainloader)

        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        valloaders.append(valloader)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloaders, valloaders, testloader

def get_dataloaders(dataset, num_clients, batch_size, dirichlet_alpha=10, data_damage=None, noise_std=1, config=None):
    print(f"Loading data for dataset {dataset}, num_clients={num_clients}, batch_size={batch_size}")

    train_transform_original = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_transform_damaged = None
    # Define data damage transformations
    if data_damage == 'noise':
        def add_noise(img):
            img = transforms.ToTensor()(img)
            noise = torch.randn(img.size()) * noise_std
            img = img + noise
            return img.clamp(0, 1)

        train_transform_damaged = transforms.Compose([
            transforms.Lambda(add_noise),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    else:
        train_transform_damaged = None

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load original CIFAR-10 dataset
    trainset_original, testset = load_cifar10(train_transform_original, test_transform)
    trainset_damaged = None

    # Apply transformations for damaged dataset
    if train_transform_damaged:
        trainset_damaged, _ = load_cifar10(train_transform_damaged, test_transform)

    # Create class indices for distribution
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(trainset_original):
        class_indices[label].append(idx)

    # Distribute indices based on the chosen strategy
    if dataset == 'cifar10_iid':
        client_indices = distribute_indices(class_indices, num_clients, strategy='iid')
    elif dataset == 'cifar10_dirichlet':
        client_indices = distribute_indices(class_indices, num_clients, strategy='dirichlet',
                                            dirichlet_alpha=dirichlet_alpha)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")

    # Create dataloaders for original and damaged datasets
    trainloaders, valloaders, testloader = create_dataloaders(trainset_original, testset, client_indices, batch_size)

    if trainset_damaged:
        trainloaders_damaged, _, _ = create_dataloaders(trainset_damaged, testset, client_indices, batch_size)
        return trainloaders, trainloaders_damaged, valloaders, testloader
    else:
        return trainloaders, None, valloaders, testloader


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        img, label = self.dataset[index]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.dataset)


# Function to denormalize images for visualization
def denormalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean
    return image.clamp(0, 1)

def preview_cifar10(trainloader):
    images, labels = next(iter(trainloader))
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axs = plt.subplots(5, 4, figsize=(12, 15))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            img = denormalize(images[i])
            img = transforms.ToPILImage()(img)
            ax.imshow(img)
            ax.set_title(CLASSES[labels[i]])
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    dataset = 'cifar10'
    num_clients = 5
    batch_size = 32
    dirichlet_alpha = 0.5
    data_damage = 'noise'
    noise_std = 0.1

    trainloaders, trainloaders2, valloaders, testloader = get_dataloaders(
        dataset=dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        dirichlet_alpha=dirichlet_alpha,
        data_damage=data_damage,
        noise_std=noise_std
    )
    preview_cifar10(trainloaders[0])
    if trainloaders2 is not None:
        print("Previewing images from the damaged probing loader (first client)")
        preview_cifar10(trainloaders2[0])


