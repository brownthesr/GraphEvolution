from torch.utils.data import Dataset
from utils import generate_ssbm, generate_dcbm
import torch
from torch_geometric.data import Data
from Graph_transformer.data import GraphDataset
from torch_geometric.loader import DataLoader
class SBMGraphDataset(Dataset):
    def __init__(self, num_nodes, num_features, num_graphs = 320):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_graphs = num_graphs

    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self,idx):
        graph, communities,edge_list =generate_dcbm(self.num_nodes,3,.7,.05,5,1.3)
        # graph, communities = generate_ssbm(self.num_nodes,self.num_features,.7,.1)

        # perm = np.random.permutation(self.num_nodes)
        # communities = communities[perm]
        # graph = graph[perm,:][:,perm]
        graph = torch.tensor(graph).long()
        one_hots = torch.eye(self.num_features)
        features = one_hots[torch.tensor(communities).long()]
        return graph,features, edge_list
class GraphTDataset(Dataset):
    def __init__(self, num_nodes, num_features, num_graphs = 320):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_graphs = num_graphs

    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self,idx):
        graph, communities,edge_list =generate_dcbm(self.num_nodes,3,.7,.05,5,1.3)
        graph = torch.tensor(graph).long()
        one_hots = torch.eye(self.num_features)
        features = one_hots[torch.tensor(communities).long()]
        data = Data(x=features,edge_index=edge_list,y=torch.randn(10))
        data.adj = graph
        return data
    