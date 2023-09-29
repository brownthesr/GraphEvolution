import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import math

NO_EDGE = torch.Tensor([0]).long()
NEW_EDGE = torch.Tensor([1]).long()
NEW_NODE = torch.Tensor([2]).long()
EOS_TOKEN = torch.Tensor([3]).long()
SOS_TOKEN = torch.Tensor([4]).long()

def draw_graph(adj,communities,epoch):
    """
    Draws the graph structure of a network

    This assigns colors according to the communities. Warning -
    does not work well for graphs of over 500 nodes. Additionally
    parameters only color nodes up to 4 different groups, additional
    coloring can be added. Isolated nodes are removed to enhance
    visibility.

    Parameters
    ----------
    adj : numpy array of shape (num_nodes, num_nodes)
        The adjacency matrix
    communities : list of size (num_nodes)
        The community assignments
    """ 
    graph = nx.from_numpy_array(adj)
    isos = list(nx.isolates(graph))
    mask = np.ones(len(communities), dtype=bool)
    mask[isos] = False
    #print(nx.number_of_isolates(g)) # used for debugging
    communities = communities[mask]
    graph.remove_nodes_from(isos)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    adj = nx.to_numpy_array(graph)
    colors = ["yellow"]*len(adj)
    colors = np.array(colors)
    colors[np.where(communities == 0)] = "green"
    colors[np.where(communities == 1)] = "blue"
    colors[np.where(communities == 2)] = "red"
    colors[np.where(communities == 3)] = "yellow"
    degrees = [graph.degree(n) for n in graph.nodes()]
    nx.draw(graph,node_color=colors,with_labels=True, node_size=[d*50 for d in degrees])
    plt.savefig(f"epoch:{epoch} graph generation")
    plt.clf()

def positional_embedding(positions, d_model):
    """
    Generates positional embeddings for given positions.

    :param positions: Number of positions (sequence length).
    :param d_model: Dimensionality of the embeddings.
    :return: Positional embeddings of shape (positions, d_model).
    """

    # Generate position indices (shape: positions, 1)
    position_indices = torch.arange(positions).unsqueeze(1).float()

    # Calculate div term (shape: 1, d_model // 2)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    # Calculate sine and cosine values (shape: positions, d_model // 2)
    sin_term = torch.sin(position_indices * div_term)
    cos_term = torch.cos(position_indices * div_term)

    # Concatenate sine and cosine terms (shape: positions, d_model)
    position_embedding = torch.zeros(positions, d_model)
    position_embedding[:, 0::2] = sin_term
    position_embedding[:, 1::2] = cos_term

    return position_embedding

def get_sequence(A,x,emb):
    sequence = torch.Tensor(emb(SOS_TOKEN))
    for i,collumn in enumerate(A.T):
        sequence = torch.vstack([sequence,emb(NEW_NODE)])
        sequence = torch.vstack([sequence,x[i]])
        for j in range(i):
            sequence = torch.vstack([sequence,emb(collumn[j]+1)])
    sequence = torch.vstack([sequence,emb(EOS_TOKEN)])
    return torch.Tensor(sequence)

def generate_ssbm(num_nodes,num_classes,p_intra,p_inter,community = None):
    """Generates a SSBM.

    This function generates a Symmetric Stochastic Block model. It does
    this by creating blocks for in and out of class probability. Then
    it draws from a uniform distribution and uses the p_intra and p_inter
    probabilities to assign edges between specific nodes

    Args:
        num_nodes (int): The number of nodes.
        num_classes (int): The number of classes.
        p_intra (float): The probability of having
            in class connections.
        p_inter (float): The probability of having
            edges between classes
        community (list): Optional, may specify how
            the nodes are divided into communities.
            Automatically assigns communities if none
            are provided.

    Returns:
        Graph (list): An adjacency matrix representing the
            edges in the generated graph.
        Communities (list): The node assignment to communities.
    """
    if community is None:
        # assign a community to each node
        community = np.repeat(list(range(num_classes)),np.ceil(num_nodes/num_classes))

        #np.repeat(list to iterate over, how many times to repeat an item)

        #make sure community has size n
        community = community[0:num_nodes]
        # just in case repeat repeated too many

    communities = community.copy()

    # make it a collumn vector
    community = np.expand_dims(community,1)

    # generate a boolean matrix indicating whether
    # two nodes share a community
    # this is a smart way to generate a section graph
    intra = community == community.T
    inter = community != community.T# we can also use np.logical not

    random = np.random.random((num_nodes,num_nodes))
    tri = np.tri(num_nodes,k=-1).astype(bool)

    intergraph = (random < p_intra) * intra * tri
    # this creates a matrix that only has trues where
    # random< p_intra, they are in intra, and along half the matrix
    # (if it were the whole matrix it would be double the edges we want)
    intragraph = (random < p_inter) * inter * tri# same thing here
    graph = np.logical_or(intergraph,intragraph)
    graph = graph*1# this converts it to a int tensor
    graph += graph.T
    return graph,communities

def generate_power_distr(num_nodes, gamma):
    """Pulls node degrees from a powerlaw distribution

    Args:
        num_nodes (int): Number of nodes
        gamma (float): the degree of our powerlaw distribution

    Returns:
        list(int): a list of the drawn degrees
        list(float): a list of the probability distribution
    """
    degrees = np.arange(num_nodes)+1
    probs = 1/(degrees**gamma)
    probs = probs/probs.sum()
    degrees = np.random.choice(degrees,num_nodes,p=probs)
    return degrees,probs

def generate_dcbm(num_nodes,num_classes,p_intra,p_inter,avg_degree,gamma,community = None):
    """This Generates a Degree Corrected Stochastic Block Model

    Args:
        num_nodes (int): number of nodes
        num_classes (int): number of classes
        p_intra (float): probability of intra-class connection
        p_inter (float): probability of inter class dimensions
        avg_degree (int): average degree of network
        gamma (float): power law degree
        community (list, optional): A list of the community structure. Defaults to None.

    Returns:
        list: the adjacency matrix with the corrosponding communities
    """
    if community is None:
        # assign a community to each node
        community = np.repeat(list(range(num_classes)),np.ceil(num_nodes/num_classes))

        #np.repeat(list to iterate over, how many times to repeat an item)

        #make sure community has size n
        community = community[0:num_nodes]
        # just in case repeat repeated too many

    communities = community.copy()

    # make it a collumn vector
    community = np.expand_dims(community,1)

    # generate a boolean matrix indicating whether
    # two nodes share a community
    # this is a smart way to generate a section graph
    intra = community == community.T
    inter = community != community.T# we can also use np.logical not

    random = np.random.random((num_nodes,num_nodes))
    tri = np.tri(num_nodes,k=-1).astype(bool)

    degrees,degree_distr = generate_power_distr(num_nodes,gamma)
    fatness = avg_degree/(degree_distr@(np.arange(num_nodes)+1))
    p_inter = p_inter/avg_degree*degrees*fatness
    p_intra = p_intra/avg_degree*degrees*fatness

    intergraph = (random < p_intra) * intra * tri
    # this creates a matrix that only has trues where
    # random< p_intra, they are in intra, and along half the matrix
    # (if it were the whole matrix it would be double the edges we want)
    intragraph = (random < p_inter) * inter * tri# same thing here
    graph = np.logical_or(intergraph,intragraph)
    graph = graph*1# this converts it to a int tensor
    graph += graph.T
    graph += np.eye(num_nodes).astype(int)
    edge_list = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes) if graph[i, j]]
    return graph,communities,edge_list 
