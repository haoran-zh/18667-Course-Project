import torch 
import numpy as np
from tqdm import tqdm
import copy 
from sampling_algorithms import dataset_size_sampling 

def train(model, trainloader, opt, max_iters): 
    """
    Trains a given model for a specified number of iterations using data from a DataLoader.
    
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): A DataLoader that provides batches of training data (inputs and targets).
        opt (torch.optim.Optimizer): The optimizer used to update the model's parameters during training.
        max_iters (int): The maximum number of iterations (batches) to train the model for.
    
    Returns:
        None
    """
    model.train()

    for i, (inputs, targets) in enumerate(trainloader): 
        if i == max_iters: 
            break  

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
        opt.zero_grad()
        loss.backward()    
        opt.step() 

def prox_train(init_model, model, trainloader, opt, mu, max_iters): 
    """
    Trains a given model for a specified number of iterations using data from a DataLoader and an additional proximal term.
    
    Args:
        init_model (torch.nn.Module): The neural network weights prior to training 
        model (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): A DataLoader that provides batches of training data (inputs and targets).
        opt (torch.optim.Optimizer): The optimizer used to update the model's parameters during training.
        mu (float): The proximal coefficient to be used in training 
        max_iters (int): The maximum number of iterations (batches) to train the model for.
    
    Returns:
        None
    """
    # TODO: Fill out the proximal training code for problem 2C 
    model.train()

    for i, (inputs, targets) in enumerate(trainloader):
        if i == max_iters:
            break

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            # Add proximal term to the loss
            proximal_term = 0.0
            for param, init_param in zip(model.parameters(), init_model.parameters()):
                proximal_term += torch.sum((param - init_param) ** 2)
            loss += (mu / 2) * proximal_term

        opt.zero_grad()
        loss.backward()
        opt.step()

def eval(model, testloader): 
    """
    Evaluates the performance of a model on a test dataset and computes the accuracy.
    
    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        testloader (torch.utils.data.DataLoader): A DataLoader that provides batches of test data (inputs and targets).
    
    Returns:
        float: The accuracy of the model on the test dataset, calculated as the proportion of correctly classified examples.
    """
    model.eval()
    total_err = 0
    total_counts = 0

    for inputs, targets in testloader: 
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        
        err = err = (targets != outputs.argmax(1)).sum()
        counts = torch.ones(len(inputs))

        total_err += err 
        total_counts += counts.sum()

    accuracy = (1 - total_err/total_counts).item()
    return accuracy

def central_train(model, trainloader, testloader, opt, eval_rounds, max_iters): 
    """
    Trains a model using centralized training and evaluates its accuracy after each round of training.
    
    Args:
        model (torch.nn.Module): The neural network model to be trained and evaluated.
        trainloader (torch.utils.data.DataLoader): A DataLoader that provides batches of training data (inputs and targets).
        testloader (torch.utils.data.DataLoader): A DataLoader that provides batches of test data for evaluation.
        opt (torch.optim.Optimizer): The optimizer used to update the model's parameters during training.
        eval_rounds (int): The number of evaluation rounds, i.e., how many times to alternate between training and evaluation.
        max_iters (int): The maximum number of iterations (batches) to train the model for in each evaluation round.
    
    Returns:
        List[float]: A list of accuracy values recorded after each evaluation round. The first value in the list should be the untrained model's initial performance 
    """
    accuracies = [] 
    # TODO: Fill out the code for centralized training for problem 1A

    # Evaluate and store initial accuracy before training
    initial_accuracy = eval(model, testloader)
    accuracies.append(initial_accuracy)

    for round_num in range(eval_rounds):
        model.train()
        iteration_count = 0

        for inputs, targets in trainloader:
            if iteration_count >= max_iters:
                break

            # Forward pass
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            opt.step()

            iteration_count += 1

        # Evaluate model performance and record accuracy
        accuracy = eval(model, testloader)
        accuracies.append(accuracy)
        print(f"Round {round_num + 1}/{eval_rounds} - Accuracy: {accuracy:.4f}")

    return accuracies

def agg_models(server_model, client_models): 
    """
    Aggregates the parameters of multiple client models by averaging them and updating the server model with the averaged parameters.
    
    Args:
        server_model (torch.nn.Module): The server model that will be updated with the averaged parameters from the client models.
        client_models (List[torch.nn.Module]): A list of client models whose parameters will be averaged to update the server model.
    
    Returns:
        None
    """
    # TODO: Fill out the code for model aggregation for problem 1B
    with torch.no_grad():
        for param in server_model.parameters():
            param.data.zero_()

        # Sum the parameters from each client model
        for client_model in client_models:
            for server_param, client_param in zip(server_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data

        # Average the parameters
        for param in server_model.parameters():
            param.data /= len(client_models)

def diff_models(init_model, final_model):
    """
    Calculate the difference between two PyTorch models and return a new model containing the differences.
    
    Args:
        init_model (torch.nn.Module): The initial model (before training).
        final_model (torch.nn.Module): The final model (after training).
    
    Returns:
        diff_model (torch.nn.Module): A new model containing the difference between init_model - final_model.
    """
    # TODO: May be helpful for problem 3D
    # compute the difference between the parameters of the two models
    # result, zeroslike
    weight_diff = set_model_as_zeros(init_model)
    for weight_diff_param, init_param, final_param in zip(weight_diff.parameters(), init_model.parameters(), final_model.parameters()):
        weight_diff_param.data = init_param.data - final_param.data
    return weight_diff


def gradient_norm(model):
    # compute the norm of the gradient of the model (this is l2 norm)
    grad_norm = 0.0
    for param in model.parameters():
        grad_norm += torch.sum(param.data ** 2)
    grad_norm = grad_norm ** 0.5
    return grad_norm

def sum_models(model1, model2):
    """
    Creates a new model where each parameter is the sum of the corresponding parameters
    in model1 and model2.
    
    Args:
        model1 (torch.nn.Module): The first PyTorch model.
        model2 (torch.nn.Module): The second PyTorch model.
    
    Returns:
        sum_model (torch.nn.Module): A new model where each parameter is the sum of the corresponding
                                     parameters from model1 and model2.
    """
    # TODO: May be helpful for problem 3D
    weight_sum = set_model_as_zeros(model1)
    for weight_sum_param, init_param, final_param in zip(weight_sum.parameters(), model1.parameters(), model2.parameters()):
        weight_sum_param.data = init_param.data + final_param.data
    return weight_sum

def model_normalize(model, d):
    weight_divide = set_model_as_zeros(model)
    for weight_divide_param, model_param in zip(weight_divide.parameters(), model.parameters()):
        weight_divide_param.data = model_param.data / d
    return weight_divide


def rescale_model(model, k):
    weight_divide = set_model_as_zeros(model)
    for weight_divide_param, model_param in zip(weight_divide.parameters(), model.parameters()):
        weight_divide_param.data = model_param.data * k
    return weight_divide

def update_model(init_model, diff_model): 
    """
    In-place update of the initial model with the diff model, where
    init_model is updated as init_model - diff_model.
    
    Args:
        init_model (torch.nn.Module): The initial model to be updated.
        diff_model (torch.nn.Module): The model containing parameter differences.
    """
    # TODO: May be helpful for problem 3D
    # define own functions
    pass

def fl_train(server_model, clients, comm_rounds, lr, momentum, local_iters, testloader, client_frac=1.0, mu=0.0):
    """
    Trains a model using Federated Learning (FL) by conducting multiple communication rounds between a central server and clients.

    Args:
        server_model (torch.nn.Module): The central model hosted on the server, which will be updated based on client models.
        clients (List[torch.utils.data.DataLoader]): A list of DataLoaders, each providing training data for a client.
        comm_rounds (int): The number of communication rounds between the server and the clients.
        lr (float): The learning rate for local client training.
        momentum (float): The momentum for the SGD optimizer used in local client training.
        local_iters (Union[int, List[int]]): The number of local training iterations for each client. If an integer, all clients 
                                             use the same number of iterations. If a list, specifies iterations per client.
        testloader (torch.utils.data.DataLoader): The DataLoader for evaluating the global model's accuracy after each round.
        client_frac (float, optional): The fraction of clients participating in each communication round. Defaults to 1.0 (i.e., full client participation).
        mu (float, optional): The coefficient for the proximal term in the FedProx algorithm. Defaults to 0.0, indicating no proximal term (i.e., standard FL).

    Returns:
        List[float]: A list of accuracy values recorded after each communication round. The first value in the list should be the server model's initial performance 

    """
    accuracies = []
    # TODO: Fill out the code for FL training. This method will updated in many of the HW problems so please try to keep the code as clean as possible 
    # A few helpful hints - 
    # - At the start of each communication round, clients should get a COPY of the server model weights. You do not want to be training the original server model weights at each client
    # - Clients should have their own optimizers, it might be helpful to create an SGD optimizer for the clients that are sampled in the round once they are initialized as a copy of the server model weights
    # - Make sure to aggregate the updated client model weights into the new server model at the end of each communication round 
    # Evaluate and store initial accuracy before training
    initial_accuracy = eval(server_model, testloader)
    accuracies.append(initial_accuracy)
    print(f"Initial Accuracy: {initial_accuracy:.4f}")

    for round_num in range(comm_rounds):
        selected_clients = np.random.choice(len(clients), int(client_frac * len(clients)), replace=False)
        # selected_clients should be all
        client_models = []

        # Train each selected client model
        for client_idx in selected_clients:
            client_model = copy.deepcopy(server_model)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=momentum)

            # Perform local training
            trainloader = clients[client_idx]
            # if local_iters is a list, then use local_iters[client_idx] as the number of iterations
            if isinstance(local_iters, list):
                client_iters = local_iters[client_idx]
            else:
                client_iters = local_iters
            if mu > 0.0:
                prox_train(server_model, client_model, trainloader, optimizer, mu, max_iters=client_iters)
            else:
                train(client_model, trainloader, optimizer, max_iters=client_iters)

            client_models.append(client_model)

            # Aggregate client models to update the server model
        agg_models(server_model, client_models)

        # Evaluate model performance and record accuracy
        accuracy = eval(server_model, testloader)
        accuracies.append(accuracy)
        print(f"Round {round_num + 1}/{comm_rounds} - Accuracy: {accuracy:.4f}")

    return accuracies


def set_model_as_zeros(model_like):
    """
    Set the parameters of a PyTorch model to zeros.

    Args:
        model (torch.nn.Module): The PyTorch model to be modified.
    """
    zeros_model = copy.deepcopy(model_like)
    for param in zeros_model.parameters():
        param.data.zero_()
    return zeros_model


def FedVARP_agg(server_model, model_diffs, y_updates, selected_clients):
    # server_model_new = server_model - Delta
    # Delta = sum(y_updates)/len(y_updates) + (model_diffs - y_updates) / len(model_diffs)
    # compute sum of y_updates
    sum_y_updates = set_model_as_zeros(server_model)
    for y_update in y_updates:
        sum_y_updates = sum_models(sum_y_updates, y_update)
    # divide by len(y_updates) total number of clients
    sum_y_updates = model_normalize(sum_y_updates, len(y_updates))

    # compute model_diffs - y_updates, only for active clients
    cnt = 0
    sum_Delta = set_model_as_zeros(server_model)
    for client_idx in selected_clients:
        model_diff = model_diffs[cnt]
        # y_update = y_updates[client_idx]
        # diff = diff_models(model_diff, y_update)
        # sum_Delta = sum_models(sum_Delta, diff)
        sum_Delta = sum_models(sum_Delta, model_diff)
        cnt += 1
    # divide by len(model_diffs) number of active clients
    sum_Delta = model_normalize(sum_Delta, len(selected_clients))
    # Delta = sum_y_updates + sum_Delta
    Delta = sum_models(sum_y_updates, sum_Delta)
    # server_model_new = server_model - Delta

    server_model = diff_models(server_model, Delta)
    return server_model




def fl_train_FedVARP(server_model, clients, comm_rounds, lr, momentum, local_iters, testloader, client_frac=1.0, mu=0.0):
    """
    Trains a model using Federated Learning (FL) by conducting multiple communication rounds between a central server and clients.

    Args:
        server_model (torch.nn.Module): The central model hosted on the server, which will be updated based on client models.
        clients (List[torch.utils.data.DataLoader]): A list of DataLoaders, each providing training data for a client.
        comm_rounds (int): The number of communication rounds between the server and the clients.
        lr (float): The learning rate for local client training.
        momentum (float): The momentum for the SGD optimizer used in local client training.
        local_iters (Union[int, List[int]]): The number of local training iterations for each client. If an integer, all clients
                                             use the same number of iterations. If a list, specifies iterations per client.
        testloader (torch.utils.data.DataLoader): The DataLoader for evaluating the global model's accuracy after each round.
        client_frac (float, optional): The fraction of clients participating in each communication round. Defaults to 1.0 (i.e., full client participation).
        mu (float, optional): The coefficient for the proximal term in the FedProx algorithm. Defaults to 0.0, indicating no proximal term (i.e., standard FL).

    Returns:
        List[float]: A list of accuracy values recorded after each communication round. The first value in the list should be the server model's initial performance

    """
    accuracies = []
    # TODO: Fill out the code for FL training. This method will updated in many of the HW problems so please try to keep the code as clean as possible
    # A few helpful hints -
    # - At the start of each communication round, clients should get a COPY of the server model weights. You do not want to be training the original server model weights at each client
    # - Clients should have their own optimizers, it might be helpful to create an SGD optimizer for the clients that are sampled in the round once they are initialized as a copy of the server model weights
    # - Make sure to aggregate the updated client model weights into the new server model at the end of each communication round
    # Evaluate and store initial accuracy before training
    initial_accuracy = eval(server_model, testloader)
    accuracies.append(initial_accuracy)
    print(f"Initial Accuracy: {initial_accuracy:.4f}")

    y_updates = [set_model_as_zeros(server_model) for _ in range(len(clients))]  # stale updates, y


    for round_num in range(comm_rounds):
        selected_clients = np.random.choice(len(clients), int(client_frac * len(clients)), replace=False)
        # selected_clients should be all
        model_diffs = []  # Delta

        # Train each selected client model
        for client_idx in selected_clients:
            client_model = copy.deepcopy(server_model)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=momentum)

            # Perform local training
            trainloader = clients[client_idx]
            # if local_iters is a list, then use local_iters[client_idx] as the number of iterations
            if isinstance(local_iters, list):
                client_iters = local_iters[client_idx]
            else:
                client_iters = local_iters
            if mu > 0.0:
                prox_train(server_model, client_model, trainloader, optimizer, mu, max_iters=client_iters)
            else:
                train(client_model, trainloader, optimizer, max_iters=client_iters)
            model_diffs.append(diff_models(server_model, client_model))

            # Aggregate client models to update the server model
        # FedVARP aggregation
        server_model = FedVARP_agg(server_model, model_diffs, y_updates, selected_clients)
        # update y_updates for active clients
        cnt = 0
        for client_idx in selected_clients:
            y_updates[client_idx] = model_diffs[cnt]
            cnt += 1

        # Evaluate model performance and record accuracy
        accuracy = eval(server_model, testloader)
        accuracies.append(accuracy)
        print(f"Round {round_num + 1}/{comm_rounds} - Accuracy: {accuracy:.4f}")

    return accuracies


def agg_prob(server_model, model_diffs, selected_clients, p_i, data_ratio):
    # server_model_new = server_model - Delta
    # Delta = sum(y_updates)/len(y_updates) + (model_diffs - y_updates) / len(model_diffs)
    # compute sum of y_updates
    sum_Delta = set_model_as_zeros(server_model)
    for client_idx in selected_clients:
        model_diff = model_diffs[client_idx]
        model_diff = rescale_model(model_diff, data_ratio[client_idx] / p_i[client_idx] * 0.9) # global learning rate: 0.9
        sum_Delta = sum_models(sum_Delta, model_diff)
    # Delta = sum_y_updates + sum_Delta
    # server_model_new = server_model - Delta
    server_model = diff_models(server_model, sum_Delta)
    return server_model


def fl_train_vr(server_model, clients, comm_rounds, lr, momentum, local_iters, testloader, client_frac=1.0, mu=0.0):
    """
    Trains a model using Federated Learning (FL) by conducting multiple communication rounds between a central server and clients.

    Args:
        server_model (torch.nn.Module): The central model hosted on the server, which will be updated based on client models.
        clients (List[torch.utils.data.DataLoader]): A list of DataLoaders, each providing training data for a client.
        comm_rounds (int): The number of communication rounds between the server and the clients.
        lr (float): The learning rate for local client training.
        momentum (float): The momentum for the SGD optimizer used in local client training.
        local_iters (Union[int, List[int]]): The number of local training iterations for each client. If an integer, all clients
                                             use the same number of iterations. If a list, specifies iterations per client.
        testloader (torch.utils.data.DataLoader): The DataLoader for evaluating the global model's accuracy after each round.
        client_frac (float, optional): The fraction of clients participating in each communication round. Defaults to 1.0 (i.e., full client participation).
        mu (float, optional): The coefficient for the proximal term in the FedProx algorithm. Defaults to 0.0, indicating no proximal term (i.e., standard FL).

    Returns:
        List[float]: A list of accuracy values recorded after each communication round. The first value in the list should be the server model's initial performance

    """
    from sampling_algorithms import variance_reduced_sampling
    accuracies = []
    initial_accuracy = eval(server_model, testloader)
    accuracies.append(initial_accuracy)

    # record datasize ratio for each client
    all_clients = np.arange(len(clients))
    num_data = []
    for client_idx in all_clients:
        num_data.append(len(clients[client_idx].dataset))
    num_data = np.array(num_data)
    data_ratio = num_data / num_data.sum()

    N = len(clients)
    m = client_frac * N


    print(f"Initial Accuracy: {initial_accuracy:.4f}")

    for round_num in range(comm_rounds):
        # selected_clients should be all
        client_models = []
        # Train each selected client model
        initial_model = copy.deepcopy(server_model)
        for client_idx in all_clients:
            client_model = copy.deepcopy(server_model)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=momentum)
            # Perform local training
            trainloader = clients[client_idx]
            # how many datapoints in trainloader
            num_data = len(trainloader.dataset)
            client_iters = local_iters
            train(client_model, trainloader, optimizer, max_iters=client_iters)

            client_models.append(client_model)
        # compute weight different (gradient)
        model_diffs = []
        diff_norms = []
        for client_idx in all_clients:
            diff = diff_models(initial_model, client_models[client_idx])
            norm = gradient_norm(diff)
            model_diffs.append(diff)
            diff_norms.append(norm * data_ratio[client_idx])
        # decide the sampling result
        selected_clients, p_i = variance_reduced_sampling(N, m, diff_norms)

        server_model = agg_prob(server_model, model_diffs, selected_clients, p_i, data_ratio)

        # Evaluate model performance and record accuracy
        accuracy = eval(server_model, testloader)
        accuracies.append(accuracy)
        print(f"Round {round_num + 1}/{comm_rounds} - Accuracy: {accuracy:.4f}")

    return accuracies


def fl_train_ds(server_model, clients, comm_rounds, lr, momentum, local_iters, testloader, client_frac=1.0, mu=0.0):
    
    accuracies = []
    initial_accuracy = eval(server_model, testloader)
    accuracies.append(initial_accuracy)

    N = len(clients)
    all_clients = np.arange(N)
    num_data = np.array([len(clients[client_idx].dataset) for client_idx in all_clients])
    data_ratio = num_data / num_data.sum()

    m = int(client_frac * N)  

    print(f"Initial Accuracy: {initial_accuracy:.4f}")

    for round_num in range(comm_rounds):
        selected_clients, p_i = dataset_size_sampling(N, m, num_data) 

        client_models = {}
        initial_model = copy.deepcopy(server_model)
        for client_idx in selected_clients:  
            client_model = copy.deepcopy(server_model)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=momentum)
            trainloader = clients[client_idx]
            client_iters = local_iters  
            train(client_model, trainloader, optimizer, max_iters=client_iters)

            client_models[client_idx] = client_model

        model_diffs = [None] * N
        for client_idx in selected_clients:
            client_model = client_models[client_idx]
            diff = diff_models(initial_model, client_model)
            model_diffs[client_idx] = diff

        server_model = agg_prob(server_model, model_diffs, selected_clients, p_i, data_ratio)

        accuracy = eval(server_model, testloader)
        accuracies.append(accuracy)
        print(f"Round {round_num + 1}/{comm_rounds} - Accuracy: {accuracy:.4f}")

    return accuracies