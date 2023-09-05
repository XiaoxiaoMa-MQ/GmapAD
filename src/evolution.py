import torch
import torch.nn.functional as F
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, hinge_loss
from utils import base_map
from tqdm import tqdm
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np

def initialize_w(size, num_nodes, stg="one-hot"):

    W = []
    if stg == "one-hot":
        w = torch.ones(num_nodes, 1, requires_grad=False)
        W.append(w)
        for i in range(size-1):
            # n = range(0, num_nodes)
            n = range(1, num_nodes)
            k = random.randint(0, num_nodes-1)
            ones = random.sample(n, k)
            w = torch.zeros(num_nodes, 1, requires_grad=False)
            w[ones] = 1
            W.append(w)
        W = torch.stack(W).view(size, num_nodes)

    return W

def mutation_cross_w(W, args):

    mut_rate = args.mut_rate 
    cros_rate = args.cros_rate
    
    new_cands = []
    
    # Get new candidates under strategy 'one-hot'
    if args.w_stg == "one-hot":
        for i, candidate in enumerate(W):
            # Generate the mutated candidates
            r = random.sample(range(len(W)),5)
            while i in r:
                r = random.sample(range(len(W)),5)
            mutated_cand = W[r[0]] + mut_rate * (W[r[1]] + W[r[2]]) + mut_rate * (W[r[3]] + W[r[4]])
            zero = torch.zeros_like(mutated_cand)
            one = torch.ones_like(mutated_cand)
            mutated_cand = torch.where(mutated_cand>=2, one, mutated_cand)
            mutated_cand = torch.where(mutated_cand<2, zero, mutated_cand)

            # Cross-over the mutated candidates
            cros_cand = mutated_cand
            for j, vij in enumerate(mutated_cand):
                pos = torch.rand(1)
                if pos < cros_rate or (i==j):
                    break
                else:
                    cros_cand[j] = candidate[j]
            new_cands.append(cros_cand)
        new_cands = torch.stack(new_cands).view(W.shape)
    
    return new_cands

def gen_can_pool(W, node_pool):
    pool_candidates = []
    for i, w in enumerate(W):
        w = w.view(-1,1)
        candidate = torch.mul(node_pool, w)
        pool_candidates.append(candidate)

    return pool_candidates

def evo_classify(clf, X_train, Y_train, X_val, Y_val, X_test, Y_test):

    clf.fit(X_train, Y_train)
    x_train_pred = clf.predict(X_train)
    svm_loss = hinge_loss(Y_train, x_train_pred)
    return svm_loss

def evolution_svm(clf, model, node_pool, args, train_graphs, val_graphs, test_graphs):
    
    cand_size = args.cand_size
    evo_epochs = args.evo_gen

    # Get the training, val, and test graphs
    g_train_reps = []
    g_train_lbs = []

    g_val_reps = []
    g_val_lbs = []

    g_test_reps = []
    g_test_lbs = []

    best_results = []

    for graph in train_graphs:
        graph.to(args.device)
        _, _, g_rep = model(graph)
        g_train_reps.append(g_rep.view(-1).detach().cpu())
        g_train_lbs.append(graph.y.detach().cpu())

    for graph in val_graphs:
        graph.to(args.device)
        _, _, g_rep = model(graph)
        g_val_reps.append(g_rep.view(-1).detach().cpu())
        g_val_lbs.append(graph.y.detach().cpu())
    
    for graph in test_graphs:
        graph.to(args.device)
        _, _, g_rep = model(graph)
        g_test_reps.append(g_rep.view(-1).detach().cpu())
        g_test_lbs.append(graph.y.detach().cpu())

    # Initialize W and candidate pool
    num_nodes = node_pool.shape[0]
    old_W = initialize_w(cand_size, num_nodes, stg=args.w_stg)
    ini_cands = gen_can_pool(old_W, node_pool)


    Y_train = torch.stack(g_train_lbs).cpu().view(-1).numpy()
    Y_val = torch.stack(g_val_lbs).cpu().view(-1).numpy()
    Y_test = torch.stack(g_test_lbs).cpu().view(-1).numpy()

    best_svm_losses = []
    svm_curve = []

    for i, cand in enumerate(ini_cands):
        h_gs = base_map(g_train_reps, cand)
        X_train = h_gs.cpu().numpy()
        
        h_gs = base_map(g_val_reps, cand)
        X_val = h_gs.cpu().numpy()

        h_gs = base_map(g_test_reps, cand)
        X_test = h_gs.cpu().numpy()
        
        svm_loss = evo_classify(clf, X_train, Y_train, X_val, Y_val, X_test, Y_test)
        best_svm_losses.append(svm_loss)

    evo_tolerance = 100
    cur_tolerance = 0
    global_min_loss = 100
    for epoch in range(evo_epochs):
        #print(f"evolving at Generation: {epoch:3d}")
        new_W = mutation_cross_w(old_W, args)
        new_cands = gen_can_pool(new_W, node_pool)

        for i, cand in enumerate(new_cands):
            h_gs = base_map(g_train_reps, cand)
            X_train = h_gs.cpu().numpy()

            h_gs = base_map(g_val_reps, cand)
            X_val = h_gs.cpu().numpy()

            h_gs = base_map(g_test_reps, cand)
            X_test = h_gs.cpu().numpy()
            
            svm_loss = evo_classify(clf, X_train, Y_train, X_val, Y_val, X_test, Y_test)

            if best_svm_losses[i] < svm_loss:
                 new_W[i] = old_W[i]
            else:
                best_svm_losses[i] = svm_loss
            
        old_W = new_W
        min_loss = min(best_svm_losses)
        if min_loss < global_min_loss:
            global_min_loss= min_loss
        else:
            cur_tolerance = cur_tolerance + 1
        if cur_tolerance > evo_tolerance:
            break
        
        svm_curve.append(global_min_loss)
        
    min_svm_loss = min(best_svm_losses)

    best_w = old_W[best_svm_losses.index(min_svm_loss)]
    best_w = best_w.view(-1,1)
    best_cands = torch.mul(node_pool, best_w)
    h_gs = base_map(g_train_reps, best_cands)
    X_train = h_gs.cpu().numpy()

    h_gs = base_map(g_val_reps, best_cands)
    X_val = h_gs.cpu().numpy()

    h_gs = base_map(g_test_reps, best_cands)
    X_test = h_gs.cpu().numpy()
    clf.fit(X_train, Y_train)
    x_train_pred = clf.predict(X_train)
    x_val_pred = clf.predict(X_val)
    x_test_pred = clf.predict(X_test)

    return clf, x_train_pred, Y_train, x_val_pred, Y_val, x_test_pred, Y_test