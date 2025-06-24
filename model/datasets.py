import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import matplotlib.pyplot as plt
import torchvision
import urllib.request
import zipfile
import os
from torch.utils.data import SubsetRandomSampler, random_split
import shutil
import random  # Ensure the random module is imported

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
    'tinyimagenet': {
        'loader': lambda train_transform, test_transform: ImageFolder("./dataset/tiny-imagenet-200/train", transform=train_transform),
        'test_loader': lambda test_transform: ImageFolder("./dataset/tiny-imagenet-200/val", transform=test_transform),
        'num_classes': 200,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
    np.random.seed(42)  # Set a fixed random seed for reproducibility

    if strategy == 'iid':
        # Equal samples per class per client
        images_per_class_per_client = min(len(indices) for indices in class_indices.values()) // num_clients
        for class_idx in sorted(class_indices.keys()):  # Sort class indices for consistency
            class_indices[class_idx].sort()  # Sort data indices for deterministic order
            for i in range(num_clients):
                start_idx = i * images_per_class_per_client
                end_idx = start_idx + images_per_class_per_client
                client_indices[i].extend(class_indices[class_idx][start_idx:end_idx])

    elif strategy == 'dirichlet':
        # Dirichlet-based allocation with fixed seed
        for class_idx in sorted(class_indices.keys()):  # Sort class indices for consistency
            class_data = np.array(sorted(class_indices[class_idx]))  # Sort data indices for deterministic order
            proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
            proportions = proportions / proportions.sum()
            client_class_counts = (proportions * len(class_data)).astype(int)
            idx = 0
            for client_id, count in enumerate(client_class_counts):
                client_indices[client_id].extend(class_data[idx:idx + count])
                idx += count
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Debug: Print the number of samples per client
    for i, indices in enumerate(client_indices):
        print(f"[DEBUG] Client {i} has {len(indices)} samples.")

    return client_indices

def create_dataloaders(trainset, testset, client_indices, batch_size):
    trainloaders = []
    valloaders = []
    valset = Subset(trainset, range(len(trainset)))

    for indices in client_indices:
        train_subset = Subset(trainset, indices)
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)  # Enable shuffling after splitting
        trainloaders.append(trainloader)

        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)  # Keep validation set deterministic
        valloaders.append(valloader)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)  # Keep test set deterministic
    return trainloaders, valloaders, testloader

def get_dataloaders(dataset, num_clients, batch_size, dirichlet_alpha=10, data_damage=None, noise_std=1, config=None):
    print(f"Loading data for dataset {dataset}, num_clients={num_clients}, batch_size={batch_size}")

    # Parse base dataset name (e.g., 'cifar10' from 'cifar10_iid')
    base_name = dataset.split('_')[0]
    if base_name not in DATASET_LOADERS:
        raise NotImplementedError(f"Dataset '{base_name}' not implemented")
    ds_info = DATASET_LOADERS[base_name]
    mean, std = ds_info['normalize']

    # Ensure Tiny ImageNet is downloaded if needed
    if base_name == 'tinyimagenet':
        train_dir = './dataset/tiny-imagenet-200/train'
        if not os.path.exists(train_dir):
            print("[INFO] Tiny ImageNet not found. Downloading and extracting...")
            download_tinyimagenet()

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
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))  # Adjusted to show only the first 10 samples
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

def preview_tiny_imagenet(dataloader):
    """Preview Tiny ImageNet dataset samples."""
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Display images
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("Tiny ImageNet Preview")
    plt.show()

def download_tinyimagenet(data_dir="./dataset"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = f"{data_dir}/tiny-imagenet-200.zip"
    extract_path = f"{data_dir}/tiny-imagenet-200"

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Ensure the extraction directory exists before extracting
    os.makedirs(extract_path, exist_ok=True)

    # Download the dataset
    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    # Extract the dataset
    if not os.path.exists(extract_path) or not os.listdir(extract_path):
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    # Reorganize the dataset
    reorganize_tinyimagenet(extract_path)

def reorganize_tinyimagenet(data_dir="./dataset/tiny-imagenet"):
    """Reorganize Tiny ImageNet dataset to match ImageFolder structure."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Debug: Check if train directory exists
    if not os.path.exists(train_dir):
        print(f"[DEBUG] Train directory not found at {train_dir}. Please check the dataset.")
        raise FileNotFoundError(f"Train directory not found at {train_dir}")
    else:
        print(f"[DEBUG] Train directory found at {train_dir}.")

    # Reorganize validation set
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(val_images_dir) and os.path.exists(val_annotations_file):
        print("Reorganizing validation set...")
        with open(val_annotations_file, "r") as f:
            annotations = f.readlines()

        for line in annotations:
            parts = line.strip().split("\t")
            image_name, class_name = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            shutil.move(os.path.join(val_images_dir, image_name), os.path.join(class_dir, image_name))

        # Remove the now-empty images directory and annotations file
        shutil.rmtree(val_images_dir)
        os.remove(val_annotations_file)
        print("Validation set reorganized.")

def load_tiny_imagenet(num_clients: int, batch_size: int, beta: float):
    """Load Tiny ImageNet dataset with optional IID or non-IID partitioning."""
    if not os.path.exists('data/tiny-imagenet-200/'):
        download_and_unzip_tiny_imagenet()

    # Data Transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Data and Creating Train/Test Split
    trainset = torchvision.datasets.ImageFolder(
        root='data/tiny-imagenet-200/train',
        transform=train_transform
    )
    testset = torchvision.datasets.ImageFolder(
        root='data/tiny-imagenet-200/val',
        transform=test_transform
    )

    trainloaders = []
    if 0.0 < beta < 1.0:
        client_to_data_ids = _get_niid_client_data_ids(trainset, num_clients, beta)
        for client_id in client_to_data_ids:
            tmp_client_img_ids = client_to_data_ids[client_id]
            tmp_train_sampler = SubsetRandomSampler(tmp_client_img_ids)
            _append_to_dataloaders(trainset, batch_size, trainloaders, tmp_train_sampler)
    else:
        partition_size = len(trainset) // num_clients
        lengths = [partition_size] * num_clients
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        for dataset in datasets:
            _append_to_dataloaders(dataset, batch_size, trainloaders)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_classes = len(np.unique(testset.targets))
    return trainloaders, testloader, num_classes

# Helper functions for non-IID partitioning and dataloader creation
def _get_niid_client_data_ids(trainset, num_clients, beta):
    """Generate non-IID client data indices using Dirichlet distribution."""
    class_indices = {i: [] for i in range(len(trainset.classes))}
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)

    client_to_data_ids = {i: [] for i in range(num_clients)}
    for class_id, indices in class_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([beta] * num_clients)
        proportions = (proportions * len(indices)).astype(int)
        start_idx = 0
        for client_id, count in enumerate(proportions):
            client_to_data_ids[client_id].extend(indices[start_idx:start_idx + count])
            start_idx += count

    return client_to_data_ids

def _append_to_dataloaders(dataset, batch_size, dataloaders, sampler=None):
    """Append a dataloader to the list of dataloaders."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    dataloaders.append(dataloader)

def download_and_unzip_tiny_imagenet(data_dir="data"):
    """Download and unzip the Tiny ImageNet dataset."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = f"{data_dir}/tiny-imagenet-200.zip"
    extract_path = f"{data_dir}/tiny-imagenet"

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download the dataset
    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    # Extract the dataset
    if not os.path.exists(extract_path):
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    # Reorganize the dataset
    reorganize_tinyimagenet(extract_path)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    dataset = 'tinyimagenet_dirichlet'
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

    #print("Previewing noised CIFAR-100 images from original loader...")
    #preview_cifar100(trainloaders[0], trainloaders2[0])

    print("Previewing Tiny ImageNet...")
    preview_tiny_imagenet(trainloaders2[0])

if __name__ == "__main__":
    set_random_seed(42)
    main()

