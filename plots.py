import argparse
from argparse import RawTextHelpFormatter
import torch.optim as optim
import random 
import matplotlib.pyplot as plt
import sys

from models import ConvNet
from data_utils import build_mnist
from train_utils import  central_train, fl_train, fl_train_FedVARP, fl_train_vr, fl_train_ds,fl_train_random, fl_train_full
import pickle


def random_sampling(args):
    clients, testloader = build_mnist(args.num_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    all_accuracies = []
    client_frac = 0.1

    model = ConvNet()
    accuracies = fl_train(model, clients, args.comm_rounds, args.lr, args.momentum, args.local_iters,
                                  testloader, client_frac=client_frac)
    all_accuracies.append(accuracies)

    # Plot the accuracies
    plt.figure(figsize=(10, 6))

    # Loop over each accuracy list and corresponding n_clients value
    for i, accuracies in enumerate(all_accuracies):
        plt.plot(accuracies, label=f'client_frac={client_frac}')

    # Add labels and title
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'random sampling')
    plt.legend()
    plt.grid(True)
    # Show the plots
    plt.savefig(f'plots/random.png')

    # save accuracies
    with open(f'plots/random.pkl', 'wb') as f:
        pickle.dump(all_accuracies, f)



def variance_reduced_sampling_train(args):
    clients, testloader = build_mnist(args.num_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    all_accuracies = []
    client_frac = 0.1

    model = ConvNet()
    accuracies = fl_train_vr(model, clients, args.comm_rounds, args.lr, args.momentum, args.local_iters,
                          testloader, client_frac=client_frac)
    all_accuracies.append(accuracies)

    # Plot the accuracies
    plt.figure(figsize=(10, 6))

    # Loop over each accuracy list and corresponding n_clients value
    for i, accuracies in enumerate(all_accuracies):
        plt.plot(accuracies, label=f'client_frac={client_frac}')

    # Add labels and title
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'random sampling')
    plt.legend()
    plt.grid(True)
    # Show the plots
    plt.savefig(f'plots/vr_sampling.png')

    # save accuracies
    with open(f'plots/vr_sampling.pkl', 'wb') as f:
        pickle.dump(all_accuracies, f)
        
def dataset_size_sampling_train(args):
    clients, testloader = build_mnist(args.num_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    all_accuracies = []
    client_frac = 0.1  

    model = ConvNet()
   
    accuracies = fl_train_ds(model, clients, args.comm_rounds, args.lr, args.momentum, args.local_iters,
                             testloader, client_frac=client_frac)  
    all_accuracies.append(accuracies)

  
    plt.figure(figsize=(10, 6))

    for i, accuracies in enumerate(all_accuracies):
        plt.plot(accuracies, label=f'client_frac={client_frac}')

    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Dataset Size Sampling') 
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/ds_sampling.png') 

    with open('plots/ds_sampling.pkl', 'wb') as f: 
        pickle.dump(all_accuracies, f)


def random_sampling_train(args):
    clients, testloader = build_mnist(args.num_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    all_accuracies = []
    client_frac = 0.1  

    model = ConvNet()

    accuracies = fl_train_random(model, clients, args.comm_rounds, args.lr, args.momentum, args.local_iters,
                                 testloader, client_frac=client_frac)
    all_accuracies.append(accuracies)

    plt.figure(figsize=(10, 6))

    for i, accuracies in enumerate(all_accuracies):
        plt.plot(accuracies, label=f'client_frac={client_frac}')

    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Random Sampling')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/random_sampling.png')

    with open('plots/random_sampling.pkl', 'wb') as f:
        pickle.dump(all_accuracies, f)
        
def full_participation_train(args):
    clients, testloader = build_mnist(args.num_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    all_accuracies = []

    model = ConvNet()

    accuracies = fl_train_full(model, clients, args.comm_rounds, args.lr, args.momentum, args.local_iters,
                               testloader)
    all_accuracies.append(accuracies)

    plt.figure(figsize=(10, 6))

    plt.plot(accuracies, label='Full Participation')

    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Full Client Participation')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/full_participation.png')

    with open('plots/full_participation.pkl', 'wb') as f:
        pickle.dump(all_accuracies, f)
        
        
def other_sampling(args):
    clients, testloader = build_mnist(args.num_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    all_accuracies = []
    client_frac = 0.1

    model = ConvNet()
    accuracies = fl_train(model, clients, args.comm_rounds, args.lr, args.momentum, args.local_iters,
                          testloader, client_frac=client_frac)
    all_accuracies.append(accuracies)

    # Plot the accuracies
    plt.figure(figsize=(10, 6))

    # Loop over each accuracy list and corresponding n_clients value
    for i, accuracies in enumerate(all_accuracies):
        plt.plot(accuracies, label=f'client_frac={client_frac}')

    # Add labels and title
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'random sampling')
    plt.legend()
    plt.grid(True)
    # Show the plots
    plt.savefig(f'plots/random.png')

    # save accuracies
    with open(f'plots/random.pkl', 'wb') as f:
        pickle.dump(all_accuracies, f)




def main(args):
    if args.samplingAlgo is None or args.samplingAlgo not in globals().keys():
        print("Please run a specific/valid question to test your code")
        sys.exit()
    
    function_name = args.samplingAlgo
    globals()[function_name](args)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='HW 3 FL Implementation', formatter_class=RawTextHelpFormatter)

    # Question Number 
    parser.add_argument('--samplingAlgo', type=str,
        help='Which sampling algorithm')

    # System params 
    parser.add_argument('--seed', type=int, default=123,
        help='Torch random seed')
    
    # Dataset params
    parser.add_argument('--num-clients', type=int, default=10,
        help='Number of client devices to use in FL training')
    parser.add_argument('--iid-alpha', type=float, default=-1,
        help='Level of heterogeneity to introduce across client devices')
    parser.add_argument('--batch-size', type=int, default=32, 
        help='Batch size for local client training')
    
    # Server training params 
    parser.add_argument('--comm-rounds', type=int, default=30,
        help='Number of communication rounds')
    parser.add_argument('--clients-frac', type=int, default=1.0,
        help='Fraction of clients to use in each communication round')
    
    # Client training params 
    parser.add_argument('--lr', type=float, default=1e-3,
        help='Learning rate at client for local updates')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='Momentum')
    parser.add_argument('--local-iters', type=int, default=10,
        help='Number of local iterations to use for training')

    
    args = parser.parse_args()

    main(args)