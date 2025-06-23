import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import matplotlib.pyplot as plt
import torchvision

# Dataset loader registry for easy extensibility
DATASET_LOADERS = {
    'cifar10': {
        'loader': lambda train_transform, test_transform: CIFAR10("./dataset", train=True, download=True, transform=train_transform),
        'test_loader': lambda test_transform: CIFAR10("./dataset", train=False, download=True, transform=test_transform),
        'num_classes': 10,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    },
    'cifar100': {
        'loader': lambda train_transform, test_transform: CIFAR100("./dataset", train=True, download=True, transform=train_transform),
        'test_loader': lambda test_transform: CIFAR100("./dataset", train=False, download=True, transform=test_transform),
        'num_classes': 100,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    },
    # Add new datasets here
}

def load_cifar10(train_transform, test_transform):
    trainset = CIFAR10("./dataset", train=True, download=True, transform=train_transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=test_transform)
    return trainset, testset

def load_cifar100(train_transform, test_transform):
    trainset = CIFAR100("./dataset", train=True, download=True, transform=train_transform)
    testset = CIFAR100("./dataset", train=False, download=True, transform=test_transform)
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
        for class_idx in class_indices:
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

    # Parse base dataset name (e.g., 'cifar10' from 'cifar10_iid')
    base_name = dataset.split('_')[0]
    if base_name not in DATASET_LOADERS:
        raise NotImplementedError(f"Dataset '{base_name}' not implemented")
    ds_info = DATASET_LOADERS[base_name]
    mean, std = ds_info['normalize']

    # Set up transforms
    train_transform_original = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_transform_damaged = None
    if data_damage == 'noise':
        def add_noise(img):
            img = transforms.ToTensor()(img)
            noise = torch.randn(img.size()) * noise_std
            img = img + noise
            return img.clamp(0, 1)
        train_transform_damaged = transforms.Compose([
            transforms.Lambda(add_noise),
            transforms.Normalize(mean, std),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    trainset_original = ds_info['loader'](train_transform_original, test_transform)
    testset = ds_info['test_loader'](test_transform)
    trainset_damaged = None
    if train_transform_damaged:
        trainset_damaged = ds_info['loader'](train_transform_damaged, test_transform)
    num_classes = ds_info['num_classes']

    # Create class indices for distribution
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset_original):
        class_indices[label].append(idx)

    # Distribute indices based on the chosen strategy
    if dataset.endswith('iid'):
        client_indices = distribute_indices(class_indices, num_clients, strategy='iid')
    elif dataset.endswith('dirichlet'):
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
    # Determine number of channels from image or mean
    if image.shape[0] == 1 or len(mean) == 1:
        mean = torch.tensor(mean).view(1, 1, 1)
        std = torch.tensor(std).view(1, 1, 1)
    else:
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

def preview_cifar100(trainloader, trainloader2):
    """Preview CIFAR-100 dataset samples from two dataloaders."""
    dataiter = iter(trainloader)
    dataiter2 = iter(trainloader2)
    images, labels = next(dataiter)
    images2, labels2 = next(dataiter2)

    # Display images from the first dataloader
    fig, axs = plt.subplots(5, 4, figsize=(12, 15))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            img = denormalize(images[i])
            img = transforms.ToPILImage()(img)
            ax.imshow(img)
            ax.set_title(f"Client {labels[i]}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Display images from the second dataloader
    fig, axs = plt.subplots(5, 4, figsize=(12, 15))
    for i, ax in enumerate(axs.flat):
        if i < len(images2):
            img = denormalize(images2[i])
            img = transforms.ToPILImage()(img)
            ax.imshow(img)
            ax.set_title(f"Client {labels2[i]}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Commenting out CIFAR-10 preview
    # dataset = 'cifar10_dirichlet'
    # num_clients = 5
    # batch_size = 32
    # dirichlet_alpha = 0.5
    # data_damage = "noise"  # Apply noise damage
    # noise_std = 10  # Increase noise standard deviation for visibility

    # trainloaders, trainloaders2, valloaders, testloader = get_dataloaders(
    #     dataset=dataset,
    #     num_clients=num_clients,
    #     batch_size=batch_size,
    #     dirichlet_alpha=dirichlet_alpha,
    #     data_damage=data_damage,
    #     noise_std=noise_std
    # )

    # print("Previewing noised CIFAR-10 images...")
    # preview_cifar10(trainloaders2[0])

    # Adding CIFAR-100 preview with noise from two dataloaders
    dataset = 'cifar100_dirichlet'
    num_clients = 5
    batch_size = 32
    dirichlet_alpha = 0.5
    data_damage = "noise"  # Apply noise damage
    noise_std = 0.5  # Increase noise standard deviation for visibility

    trainloaders, trainloaders2, valloaders, testloader = get_dataloaders(
        dataset=dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        dirichlet_alpha=dirichlet_alpha,
        data_damage=data_damage,
        noise_std=noise_std
    )

    print("Previewing noised CIFAR-100 images from original loader...")
    preview_cifar100(trainloaders[0], trainloaders2[0])

if __name__ == "__main__":
    main()


