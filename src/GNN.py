from torch.nn import Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from utils import class_wise_loss

class GmapAD_GAT(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_channels, num_classes, num_heads=8):
        super(GmapAD_GAT, self).__init__()
        torch.manual_seed(7)
        self.name = "GmapAD_GAT"
        self.conv1 = GATConv(input_dim, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, int(hidden_channels/2), heads=num_heads)
        self.fc = Linear(int(hidden_channels/2)*num_heads, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        n_reps = self.conv2(x, edge_index)
        n_reps = F.leaky_relu(n_reps)
        n_reps = self.dropout(n_reps)

        # 2. Readout layers
        g_rep = n_reps.mean(dim=0, keepdim=True)
        # 3. Apply a final classifier
        x = F.dropout(g_rep, p=0.5, training=self.training)
        x = F.softmax(self.fc(x), dim=1)
        # the return values: predicted label, node representations, graph representaion.
        return x, n_reps, g_rep

class GmapAD_GCN(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_channels, num_classes):
        super(GmapAD_GCN, self).__init__()
        torch.manual_seed(7)
        self.name = "GmapAD_GCN"
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels/2))
        self.fc = Linear(int(hidden_channels/2), num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        n_reps = self.conv2(x, edge_index)
        n_reps = n_reps.relu()

        # 2. Readout layer
        g_rep = n_reps.mean(dim=0, keepdim=True)
        # 3. Apply a final classifier
        x = F.dropout(g_rep, p=0.5, training=self.training)
        x = F.softmax(self.fc(x), dim=1)
        return x, n_reps, g_rep

def train_gnn(model, train_graphs, val_graphs, test_graphs, args):

    tolerance = 0
    last_loss = 100
    device = args.device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, args.epochs):
        loss = train(model, train_graphs, optimizer, criterion, device)
        '''
        Early Stop GNN learning Process
        '''
        if loss >= last_loss and args.early_stop:
            tolerance += 1
            if tolerance >= args.tol:
                print(f'Early Stop at Epoch: {epoch:03d}') #, Train Acc: {train_acc:.5f}, Train Pre: {train_pre:.5f}, Val Acc: {val_acc:.5f}, Val Pre: {val_pre:.5f}, Loss: {loss:.4f}')
                return
        else:
            tolerance = 0
        last_loss = loss

    print(f"GNN Fully trained \n")

def train(model, dataset, optimizer, criterion, device):
    model.train()
    pred = []
    y = []
    for graph in dataset:  # Iterate in batches over the training dataset.
        graph = graph.to(device)
        out, n_rep, g_rep = model(graph)  # Perform a single forward pass.
        pred.append(out)
        y.append(graph.y)
         
    pred = torch.stack(pred, dim=0).squeeze()
    y = torch.stack(y, dim=1).squeeze()
    loss = class_wise_loss(pred, y)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()  # Clear gradients.
    return loss
