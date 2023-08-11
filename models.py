import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import math
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        # d_model needs to be even
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        # print(pe.shape)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_model)))
                # print(pos,i+1)
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
                requires_grad=False).cpu()
        return x

class GCNTime(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super(GCNTime, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=1)
        self.encoders.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
        self.pos_encoder = PositionalEncoder(hidden_channels)
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=1)
            self.encoders.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
            
        # self.convs.append(GCNConv(hidden_channels, num_classes))
    def generate_square_subsequent_mask(self,sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for i in range(self.num_layers):
            # Convolutions should have input of shape [T, N, F] and [T, 2, E] where E is the number of edges and N is the number of nodes
            # and T is the number of time steps and F is the number of features
            # in short these guys need to be a list of graphs varying in time
            # grouped by graph time
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            # make sure to set batch first = True   
            # Inputs of shape [T, N, F] where T is the sequence length, N is the number of nodes, F is the number of features
            # These guys are shaped the same way that the other guys are however they perfom things across time
            x = self.pos_encoder(x)
            x = self.encoders[i](x,mask = self.generate_square_subsequent_mask(x.shape[0]))
        # the output should be of the shape [T, N, F] where T is the sequence length, N is the number of nodes, F is the number of features
        return x


model = GCNTime(5, 5, 6, 2)
test = torch.rand(40,10, 5)
edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0],
                        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0, 9]])
edge_attr = torch.rand(20, 5)
data = Data(x=test, edge_index=edges, edge_attr=edge_attr)
print(data)
print(model(data).shape)
# print(model.parameters)
