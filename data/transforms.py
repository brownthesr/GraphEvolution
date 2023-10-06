import torch
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import numpy as np
from torch_geometric.utils import from_networkx
NEW_NODE = torch.Tensor([0]).long()
EOS_TOKEN = torch.Tensor([1]).long()
SOS_TOKEN = torch.Tensor([2]).long()
def one_hot_transform(data):
    # Assuming 'x' is a vector of integers representing classes
    num_classes = int(data.state.max().item()) + 1  # Get the number of classes
    one_hot = torch.zeros((data.state.size(0), num_classes))
    one_hot.scatter_(1, data.state.long().unsqueeze(-1), 1)
    data.x = one_hot
    return data

def pyg_transform(graph):
    """Convert a NetworkX graph to a PyTorch Geometric Data object."""
    # Convert NetworkX graph to a PyG Data object
    data = from_networkx(graph)
    
    # Use the node 'state' attribute as the node feature for the PyG Data object
    data.x = torch.tensor([[graph.nodes[node]['state']] for node in graph.nodes()], dtype=torch.float)
    
    return data

def spectral_transform(data, k=19):
    """
    A transform that computes the Laplacian spectral coordinates and the first k eigenvalues.
    
    Parameters:
    - data: PyTorch Geometric Data object.
    - k: Number of eigenvalues to be computed.
    
    Returns:
    - Modified Data object with additional attributes.
    """
    # Convert edge_index to an adjacency matrix
    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src, dest = edge_index[:, i]
        adjacency_matrix[src, dest] = 1
        adjacency_matrix[dest, src] = 1  # Since it's an undirected graph

    # Compute the Laplacian matrix
    L = laplacian(adjacency_matrix, normed=False)
    
    # Compute the eigendecomposition of the Laplacian
    eigenvalues, eigenvectors = eigh(L)
    
    # Store the first k eigenvalues
    data.eigenvalues = torch.tensor(eigenvalues[1:k+1], dtype=torch.float)
    # Store the components of the first k eigenvectors as spectral coordinates for each node
    data.spectral_coords = torch.tensor(eigenvectors[:, 1:k+1], dtype=torch.float)  # excluding the 0 eigenvalue

    data.x = torch.hstack((data.x,data.spectral_coords))
    
    return data

def get_sequence(data):
        """
        Converts a graph (represented as an edge list) into a sequence.

        Parameters
        ----------
        edge_list: list of tuple
            List of edges where each edge is represented as a tuple (source_node, target_node)
        x: torch tensor of shape (num_nodes, feature_dim)
            The features of our nodes
        latent_embeddings: torch tensor of shape (feature_dim,)
            The outputs from the encoder

        Returns
        -------
        list: Sequence representation of our graphs
        list: Masks indicating where the actual embeddings are
        list: Masks indicating where the integer tokens are
        """
        x=  data.x
        edge_list = data.edge_index.T
        num_nodes = x.shape[0]

        # Start with SOS token
        sequences = [SOS_TOKEN]
        # Track used unique identifiers to ensure no repetition
        used_ids = set([SOS_TOKEN.item(), EOS_TOKEN.item(), NEW_NODE.item()])
        node_to_id = {}
        for i in range(num_nodes):
            # Add NEW_NODE token
            sequences.append(NEW_NODE)
            
            # Add unique identifier for the node
            while True:
                unique_id = torch.randint(5, 43, (1,)).long()
                if unique_id.item() not in used_ids:
                    used_ids.add(unique_id.item())
                    node_to_id[i] = unique_id
                    break
            sequences.append(unique_id)
            
            # Add the actual embedding encoded by x
            sequences.append(x[i])
            
            # Add unique identifiers for neighboring nodes (that comes before it in the sequence)
            for edge in edge_list:
                if edge[1] == i and edge[0] < i:
                    sequences.append(node_to_id[edge[0].item()])
        
        # Add EOS token
        sequences.append(EOS_TOKEN)
        # Create masks
        continuous_target_mask = [isinstance(item, torch.Tensor) and len(item) > 1 for item in sequences]
        discrete_mask = [not val for val in continuous_target_mask]
        
        discrete_data = None
        continuous_data = None
        for i,val in enumerate(discrete_mask):
            if val:
                if discrete_data is not None:
                    discrete_data = torch.hstack((discrete_data,sequences[i]))
                else:
                    discrete_data = sequences[i]
            else:
                if continuous_data is not None:
                    continuous_data = torch.vstack((continuous_data,sequences[i]))
                else:
                    continuous_data = sequences[i]
        data.discrete_data=discrete_data
        data.continuous_data=continuous_data
        data.continuous_mask=continuous_target_mask
        data.discrete_mask=discrete_mask
        return data

def transformer_batch(databatch, discrete_emb, contin_emb):
    items = databatch.to_data_list()
    sequences = []
    max_len = 0

    for item in items:
        # Reconstruct the sequence
        sequence = []
        i, j = 0, 0
        cont = contin_emb(item.continuous_data)
        discrete = discrete_emb(item.discrete_data.unsqueeze(-1))

        # Interleave these two according to the mask
        for switch in item.continuous_mask:
            if switch:
                sequence.append(cont[i])
                i += 1
            else:
                sequence.append(discrete[j].squeeze(0))
                j += 1

        sequences.append(torch.stack(sequence))
        max_len = max(max_len, len(sequence))

    # Pad sequences and create padding mask
    padded_sequences = []
    padding_masks = []
    for sequence in sequences:
        pad_size = max_len - sequence.size(0)
        padded_sequence = torch.cat([sequence, torch.zeros(pad_size, *sequence.size()[1:])], dim=0)
        padded_sequences.append(padded_sequence)
        mask = torch.cat([torch.ones(sequence.size(0)), torch.zeros(pad_size)], dim=0)
        padding_masks.append(mask)

    b_seq = torch.stack(padded_sequences)
    padding_mask = torch.stack(padding_masks)

    return b_seq, padding_mask


#TODO
# Ensure that running these types of batchs through a GCN works

# maybe try adding subgraph counts to features
# Try using a diffusion map on the trained autoencoder
#   plot the first 3-4 eigenfunctions