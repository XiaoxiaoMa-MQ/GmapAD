from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, degree
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import numpy as np
import os

def gen_graphs(records):
    node_name = []
    ## Get all node names and assign a global ID to each node
    for i in range(len(records)):
        if records[i] != "\n":
            if 'n' in records[i]:
                n_name = records[i].split(' ')[2].strip('\n')
                if n_name not in node_name:
                    node_name.append(n_name)

    node_index = dict()
    for g_id, name in enumerate(node_name):
        node_index.update({
            name : g_id })

    graph_names = []
    graphs = dict()
    graph = dict()
    nodes = dict()
    edges = dict()
    graph_num = 1
    edge_id = 1

    for i in range(len(records)):
        if records[i] != "\n":
            indicator = records[i].split(' ')[0].strip('\n')
            
            # Graph ID
            if 'g' == indicator:
                graph_name = records[i].split(' ')[2].strip('\n')
                graph_names.append(graph_name)
            
            # Node name and ID
            if 'n' == indicator:
                node_id_local = records[i].split(' ')[1].strip('\n')
                n_name = records[i].split(' ')[2].strip('\n')
                nodes.update({
                        node_id_local : n_name
                    })
            
            if 'e' == indicator:
                st_node_id_local = records[i].split(' ')[1].strip('\n')
                ed_node_id_local = records[i].split(' ')[2].strip('\n')
                w = records[i].split(' ')[3].strip('\n')
                edge = [st_node_id_local, ed_node_id_local, w]
                edges.update({
                    edge_id : edge
                })
                edge_id = edge_id + 1
            
            if 'x' == indicator:
                label = records[i].split(' ')[1].strip('\n')
                if label == '1':
                    g_label = 1
                else:
                    g_label = 0
        else:
            graph.update({
                "nodes" : nodes,
                "edges" : edges,
                "label" : g_label
            })
            graphs.update({
                graph_name : graph
            })
            graph = dict()
            nodes = dict()
            edges = dict()
            edge_id = 0
            graph_num = graph_num + 1

    return graphs, node_index

def load_dataset(dataset, args):

    root_path = "../data/"

    if dataset in ['KKI', 'OHSU']:
        with open(root_path+"BrainNetwork/{}.nel".format(dataset)) as f:
            data = f.readlines()
        f.close()
        graphs, node_index =  gen_graphs(data)
        # create a graph data for each graph in graphs
        dataset = []
        for graph in graphs:
            nodes = graphs[graph]['nodes']
            edges = graphs[graph]['edges']
            label = graphs[graph]['label']
            label = np.array([label])
            label = torch.tensor(label,dtype=torch.long)
            st_nodes = []
            ed_nodes = []
            w= []
            for edge in edges:
                node_s = edges[edge][0]
                node_e = edges[edge][1]
                st_nodes.append(node_index[nodes[edges[edge][0]]])
                ed_nodes.append(node_index[nodes[edges[edge][1]]])
                w.append(float(edges[edge][2]))
            selflinks = list(range(0,len(node_index)))
            st_nodes = st_nodes + selflinks
            ed_nodes = ed_nodes + selflinks

            st_nodes = torch.tensor(st_nodes, dtype=torch.long)
            ed_nodes = torch.tensor(ed_nodes, dtype=torch.long)
            w = torch.LongTensor(w)
            edge_index = torch.cat((st_nodes, ed_nodes)).reshape(2,st_nodes.shape[0])
            w = w.reshape(w.shape[0],1)
            
            spm = to_scipy_sparse_matrix(edge_index)
            spm = spm.toarray()
            spm.astype(np.float64)
            spm = torch.tensor(spm, dtype=torch.float)
            graph = Data(x=spm, edge_index=edge_index, y=label)
            dataset.append(graph)
        
        dataset = build_x(dataset)
        return dataset
    else:
        data = TUDataset(root=f'{root_path}/TUDataset', name=f'{dataset}')
        return data

def build_x(graphs):
    num_nodes = graphs[0].num_nodes
    node_tags = []
    for i in range(len(graphs)):
        node_tags.append(degree(graphs[i].edge_index[0]).tolist())

    # Extracting unique tag labels
    tagset = set([])
    for tag in node_tags:
        tagset = tagset.union(set(tag))
    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for graph in graphs:
        graph.x = torch.zeros(num_nodes, len(tagset))

    for i in range(len(graphs)):
        graphs[i].x[range(len(node_tags[i])), [tag2index[tag] for tag in node_tags[i]]] = 1
    return graphs

def base_map(g_reps, pool_candidate):
    '''
    input: list of graph representations, node pool candidates
    output: each graph's representation on candidate pool
    '''
    rep = torch.stack(g_reps)
    return torch.cdist(rep, pool_candidate, p=1)

def pos_graphs_pool(graphs, model, args):
    '''
    This function returns the node pool, G0 ... GN are all from postive graphs in the trainning set.
    '''
    if args.dataset in ['KKI', 'OHSU']:
        n_reps = []
        g_reps = []
        for graph in graphs:
            graph.to(args.device)
            _, n_rep, g_rep = model(graph)
            n_reps.append(n_rep.detach())
            g_reps.append(g_rep.detach())
        n_reps = torch.stack(n_reps)
        g_reps = torch.stack(g_reps)
        
        rep_stg= args.n_p_stg
        # return the node pool according to the pool genenrating strategy
        if rep_stg == "mean":
            node_pool = torch.mean(n_reps, 0)
        if rep_stg == "sum":
            node_pool = torch.sum(n_reps, 0)
        if rep_stg == "max":
            node_pool = torch.max(n_reps, 0).values
        if rep_stg == "min":
            node_pool = torch.min(n_reps, 0).values
        if rep_stg == "concat":
            node_pool = n_reps[0]
            for n_rep in n_reps[1:]:
                node_pool = torch.cat((node_pool, n_rep), dim=0)
            node_pool = top_k_nodes(node_pool, g_reps, args)
    else:
        n_reps = []
        g_reps = []
        for graph in graphs:
            graph.to(args.device)
            _, n_rep, g_rep = model(graph)
            n_reps.append(n_rep.detach())
            g_reps.append(g_rep.detach())
        node_pool = n_reps[0]
        for n_rep in n_reps[1:]:
            node_pool = torch.cat((node_pool, n_rep), dim=0)
        g_reps = torch.stack(g_reps)
        g_reps = g_reps.squeeze()
        node_pool = top_k_nodes(node_pool, g_reps, args)
    return node_pool

def top_k_nodes(node_pool, g_reps, args):
    node_pool = F.normalize(node_pool, p=2, dim=-1)
    g_reps = F.normalize(g_reps, p=2, dim=-1)
    cos_sim = torch.matmul(g_reps, node_pool.T)
    node_sims = cos_sim.sum(dim=0)
    top_k_nodes = torch.topk(node_sims, args.topk, -1, True).indices
    node_pool = node_pool.index_select(0, top_k_nodes)

    return node_pool

def class_wise_loss(pred, y):
    criterion = torch.nn.CrossEntropyLoss()
    mask_0 = [False for _ in range(y.shape[0])]
    mask_1 = [False for _ in range(y.shape[0])]
    for i, l in enumerate(y):
        if l == 1:
            mask_1[i] = True
            mask_0[i] = False
        else:
            mask_1[i] = False
            mask_0[i] = True

    loss = criterion(pred[mask_0], y[mask_0]) + criterion(pred[mask_1], y[mask_1])  # Compute the loss.
    return loss

def create_dirs(args):
    # check all necessary directories
    dirs = [
        f"../data/{args.dataset}/",
        f"../data/{args.dataset}/{args.gnn_layer}/",
        f"../data/{args.dataset}/{args.gnn_layer}/SVM"
    ]
    for d in dirs:
        if not os.path.exists(d):
            print(f"Directory {d} not exist, creating...")
            os.mkdir(d)

def print_dataset_stat(args, graphs):
    a_graphs, a_nodes, a_edges, t_nodes, t_edges = 0, 0, 0, 0, 0
    for graph in graphs:
        t_nodes += graph.num_nodes
        t_edges += graph.num_edges
        if graph.y == args.ds_cl:
            a_graphs += 1
            a_nodes += graph.num_nodes
            a_edges += graph.num_edges
    
    print(f"Graph Statistics {args.dataset} : ")
    print("{:<8} | {:<10} | {:<10} | {:<10} ".format("Class", "#Graphs", "Avg. V", "Avg. E" ))
    print("{:<8} | {:<10} | {:<10} | {:<10} ".format("G_0", a_graphs, a_nodes/a_graphs, a_edges/a_graphs ))
    print("{:<8} | {:<10} | {:<10} | {:<10} ".format("G_1", len(graphs) - a_graphs,(t_nodes - a_nodes) /(len(graphs) - a_graphs),  (t_edges - a_edges) /(len(graphs) - a_graphs) ))
