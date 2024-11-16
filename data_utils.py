import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Subset

def build_mnist(n_clients, alpha, batch_size, seed):
    """
    Builds the MNIST dataset for either centralized or federated learning scenarios.
    
    Args:
        n_clients (int): The number of clients for federated learning. If 1, centralized training is performed.
        alpha (float): The parameter for controlling the Dirichlet distribution used in partitioning the dataset 
                       among clients
        batch_size (int): The size of the batches to be loaded by the DataLoader.
        seed (int): torch random seed 
        
    Returns:
        If `n_clients == 1`:
            Tuple[DataLoader, DataLoader]: Returns a tuple containing the training and testing DataLoader 
                                           for centralized training.
        If `n_clients > 1`:
            Tuple[List[DataLoader], DataLoader]: Returns a tuple containing a list of DataLoaders for each client 
                                                 (training data) and a single DataLoader for testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    N = len(trainset)
    Y = np.array(trainset.targets)
    n_classes = 10
    
    # Centralized training case 
    if n_clients == 1: 
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
        return trainloader, testloader

    clients = partition_dataset(trainset, Y, n_classes, n_clients, alpha, seed)
    clientloaders = [DataLoader(client, batch_size=batch_size, shuffle=True) for client in clients]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return clientloaders, testloader

def partition_dataset(dataset, Y, n_classes, n_clients, alpha, seed):
    """
    Partitions a dataset into subsets for multiple clients, supporting both IID and non-IID cases.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to be partitioned.
        Y (np.array): The target labels of the dataset, used to group examples by class.
        n_classes (int): The number of unique classes in the dataset (e.g., 10 for MNIST).
        n_clients (int): The number of clients.
        alpha (float): The parameter controlling the distribution of data across clients. 
                       If `alpha == -1`, the dataset is partitioned IID (Independent and Identically Distributed).
                       If `alpha > 0`, the dataset is partitioned non-IID using a Dirichlet distribution
        seed (int): torch random seed.
    
    Returns:
        List[torch.utils.data.Subset]: A list of `torch.utils.data.Subset` objects, where each subset represents the 
                                       data assigned to a particular client.
    """
    clients = []

    # TODO: Fill out the IID Case for problem 1B
    # IID Case
    if alpha == -1:
        # Shuffle the dataset indices
        indices = np.random.permutation(len(dataset))
        # Split the indices evenly among clients
        split_indices = np.array_split(indices, n_clients)
        # Create a Subset for each client
        clients = [Subset(dataset, split_idx) for split_idx in split_indices]
    # TODO: Fill out the NIID Case for problem 2A
    # NIID Case
    else:
        # Group indices by class
        class_indices = [np.where(Y == i)[0] for i in range(n_classes)]
        client_indices = [[] for _ in range(n_clients)]

        # For each class, partition the data among clients using a Dirichlet distribution
        for class_idx in class_indices:
            # Get the proportion for each client using Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * n_clients)
            # Scale proportions to match the number of samples in the class
            proportions = (len(class_idx) * proportions).astype(int)

            # Shuffle class indices
            np.random.shuffle(class_idx)

            # Split the class indices among clients based on the proportions
            start_idx = 0
            for client_idx, num_samples in enumerate(proportions):
                client_indices[client_idx].extend(class_idx[start_idx:start_idx + num_samples])
                start_idx += num_samples

        # Create Subset for each client
        clients = [Subset(dataset, np.array(indices)) for indices in client_indices]

    return clients