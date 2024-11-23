import torch
import numpy as np
import copy

# start your sampling algorithms in this file

def variance_reduced_sampling(n, m, model_diffs):
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.


    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(model_diffs)

    l = int(n - m + 1)
    best_l = l
    if m == 0:  # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(model_diffs[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / model_diffs[sorted_indices[l - 1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_i = np.zeros(n)
    sum_upto_l = sum(model_diffs[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            p_i[sorted_indices[i]] = 1.0
        else:
            p_i[sorted_indices[i]] = (m + best_l - n) * model_diffs[sorted_indices[i]] / sum_upto_l

    allocation_result = np.zeros(n, dtype=int)
    for idx in range(n):
        p = p_i[idx]
        # binomial sampling, decide 0 or 1
        allocation_result[idx] = np.random.choice([0, 1], p=[1.0 - p, p])
    # convert allocation_result to list of indices (selected_clients)
    selected_clients = []
    for idx in range(n):
        if allocation_result[idx] == 1:
            selected_clients.append(idx)
    return selected_clients, p_i

def dataset_size_sampling(n, m, dataset_sizes):

    total_size = np.sum(dataset_sizes)


    p_i = dataset_sizes / total_size

    selected_clients = np.random.choice(n, size=m, replace=False, p=p_i)

    return selected_clients, p_i