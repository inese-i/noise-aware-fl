import logging
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_class_distribution(train_loader):
    """Returns class distribution for a single client"""
    class_counts = {i: 0 for i in range(10)}

    for data, target in train_loader:
        for label in target:
            class_counts[label.item()] += 1
    return class_counts

def plot_class_distribution_all_clients(class_distributions, name):

    num_clients = len(class_distributions)
    client_ids = list(class_distributions.keys())
    data = np.array([list(class_distributions[cid].values()) for cid in client_ids])
    colors = sns.color_palette("tab10", 10)
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(num_clients)
    for i, class_color in enumerate(colors):
        ax.barh(client_ids, data[:, i], left=bottom, color=class_color, label=f'Class {i}')
        bottom += data[:, i]

    ax.invert_yaxis()
    ax.set_xlabel('Number of Samples', fontsize=30)
    ax.set_ylabel('Client ID', fontsize=30)

    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',  prop={'size': 25})
    plt.tight_layout()
    plt.savefig(f'results/{name}_class_dist.pdf', format='pdf', dpi=300)
