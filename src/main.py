import argparse
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import numpy as np
import sys
from sklearn import svm

from utils import load_dataset, pos_graphs_pool, print_dataset_stat
from GNN import GmapAD_GCN, GmapAD_GAT, train_gnn
from evolution import evolution_svm
import os
import random
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='OHSU', help="['KKI', 'OHSU', 'MUTAG', 'Mutangenicity', 'PROTEINS', 'AIDS', 'NCI1', 'IMDB-BINARY', 'REDDIT-BINARY')")
    parser.add_argument('--ds_rate', type=float, default=0.1, help='Dataset downsampling rate for Graph classification datasets.')
    parser.add_argument('--ds_cl', type=int, default=0, help='The default downsampled class.')

    # GNN related parameters
    parser.add_argument('--gnn_layer', type=str, default='GCN', help="['GCN','GAT']")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', default=0.005, help='Learning rate of the optimiser.')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay of the optimiser.')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--tol', type=int, default=300)
    parser.add_argument('--early_stop', type=bool, default=True, help="Early stop when training GNN")

    # Node pooling related parameters
    parser.add_argument('--n_p_g', type=str, default='positive', help="['positive', 'negative']")
    parser.add_argument('--n_p_stg', type=str, default='mean', help="['mean','max', 'min']")
    
    # Evolving related parameters
    parser.add_argument('--w_stg', type=str, default='one-hot', help="['one-hot']")
    parser.add_argument('--clf', type=str, default='svm', help="['svm', 'others']")
    parser.add_argument('--mut_rate', type=float, default=0.5, help="['svm','nb', 'others']")
    parser.add_argument('--cros_rate', type=float, default=0.9, help="['svm','nb', 'others']")
    parser.add_argument('--evo_gen', type=int, default=2000, help="number of evolution generations")
    parser.add_argument('--cand_size', type=int, default=30, help="candidates in each generation")

    # Model hyperparameters
    parser.add_argument('--gnn_dim', type=int, default=128)
    parser.add_argument('--fcn_dim', type=int, default=32)
    parser.add_argument('--gce_q', default=0.7, help='gce q')
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=64, help="number of the most informative nodes, this parameter also decides the finally graph embedding dimension.")

    # For GAT only, num of attention heads
    parser.add_argument('--gat_heads', default=8, help='GAT heads')

    # Test round
    parser.add_argument('--round', type=int, default=1, help='test round')

    args = parser.parse_args()
    return args

def downsample(ds_rate, ds_cl, graphs):
    if args.dataset not in ['KKI', 'OHSU']:
        ds_rate = args.ds_rate
        ds_cl = args.ds_cl
        ds_graphs = []
        all_graphs = []
        num_nodes = 0
        for graph in graphs:
            num_nodes += graph.num_nodes
            if graph.y == ds_cl:
                ds_graphs.append(graph)
            all_graphs.append(graph)
        ds_graphs = ds_graphs[int(len(ds_graphs)*ds_rate):]
        [all_graphs.remove(graph) for graph in ds_graphs]
        return all_graphs
    else:
        return graphs

if __name__ == "__main__":

    args = arg_parser()
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Training device: {device}")
    print(f"loading dataset {args.dataset}")
    print(f"Testing Round: {args.round}")

    graph_path = f"./data/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt"
    train_path = f"./data/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt"
    val_path = f"./data/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt"
    test_path = f"./data/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt"
    
    if not os.path.exists(graph_path) or not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
        graphs = load_dataset(args.dataset, args)
        if args.dataset in ['KKI', 'OHSU']:
            random.shuffle(graphs)
        else:
            graphs = graphs.shuffle()
        torch.save(graphs, f"../data/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt")
        train_ratio = args.train_ratio
        val_ratio = args.test_ratio
        train_graphs = graphs[:int(len(graphs)*train_ratio)]
        val_graphs = graphs[int(len(graphs)*train_ratio): int(len(graphs)*(train_ratio+val_ratio))]
        test_graphs = graphs[int(len(graphs)*(train_ratio+val_ratio)):]

        # Downsampling
        train_graphs = downsample(args.ds_rate, args.ds_cl, train_graphs)
        val_graphs = downsample(args.ds_rate, args.ds_cl, val_graphs)
        test_graphs = downsample(args.ds_rate, args.ds_cl, test_graphs)
        
        # Save downsampled datasets
        torch.save(train_graphs, f"./data/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt")
        torch.save(val_graphs, f"./data/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt")
        torch.save(test_graphs, f"./data/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt")
    else:
        print("load from pre-splitted data.")
        graphs = torch.load(f"./data/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt")
        train_graphs = torch.load(f"./data/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt")
        val_graphs = torch.load(f"./data/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt")
        test_graphs = torch.load(f"./data/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt")

    print_dataset_stat(args, graphs)

    if args.gnn_layer == "GCN":
        model = GmapAD_GCN(num_nodes=graphs[0].x.shape[0], input_dim=graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)
    else:
        model = GmapAD_GAT(num_nodes=graphs[0].x.shape[0], input_dim=graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2, num_heads=args.gat_heads)
    
    model = model.to(device)
    print(f"Start training model {args.gnn_layer}")
    train_gnn(model, train_graphs, val_graphs, test_graphs, args)

    # Get the candidate pool, grpah reprsentations
    pos_graphs = []
    neg_graphs = []

    for graph in train_graphs:
        if graph.y == 1:
            pos_graphs.append(graph)
        else:
            neg_graphs.append(graph)

    node_pool = pos_graphs_pool(pos_graphs, model, args)
    node_pool = node_pool.cpu()
    print(f"Generating Node pool size: {node_pool.size()}")

    if args.clf == "svm":
        clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
        print(f"Test on {args.dataset}, using SVM, graph pool is {args.n_p_g}, node pool stg is {args.n_p_stg}")
        clf, x_train_pred, Y_train, x_val_pred, Y_val, x_test_pred, Y_test = evolution_svm(clf, model, node_pool, args, train_graphs, val_graphs, test_graphs)
 
