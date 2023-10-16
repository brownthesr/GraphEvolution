import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
from tqdm import tqdm
import random
import networkx as nx
from torch_geometric.nn import GCN
import sys
import matplotlib.pyplot as plt
from dataset import SISDataset
import os
from transforms import pyg_transform,spectral_transform,one_hot_transform,get_sequence,batch_to_sequence

def draw_and_save_graphs(sequence, frame_dir, sequence_idx):
    # Get positions for nodes in the first graph and use them for all graphs in the sequence
    pos = nx.spring_layout(sequence[0])
    
    for idx, graph in enumerate(sequence):
        plt.figure(figsize=(8, 6))
        
        # Draw susceptible nodes
        susceptible_nodes = [node for node, data in graph.nodes(data=True) if data['state'] == 0]
        nx.draw_networkx_nodes(graph, pos, nodelist=susceptible_nodes, node_color='blue', node_size=200)
        
        # Draw infected nodes
        infected_nodes = [node for node, data in graph.nodes(data=True) if data['state'] == 1]
        nx.draw_networkx_nodes(graph, pos, nodelist=infected_nodes, node_color='red', node_size=200)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos)
        
        # Save the frame
        frame_path = os.path.join(frame_dir, f'sequence_{sequence_idx}_frame_{idx}.png')
        plt.savefig(frame_path)
        plt.close()

def sis_dynamics_with_dynamic_edges(graph, beta, gamma, initial_infections, p_disconnect,time_steps=40):
    """
    Simulate the SIS model on a graph with changing edge structures.
    
    Parameters:
    - graph: initial graph with all nodes susceptible.
    - beta: infection probability.
    - gamma: recovery probability.
    - initial_infections: number of initially infected nodes.
    - p_disconnect: probability of a node disconnecting from an infected neighbor.
    - p_connect: probability of a node trying to connect to another susceptible node (this will now be unused).
    
    Returns:
    - A sequence of 20 graph states.
    """
    
    # Initialize states: 0 for Susceptible, 1 for Infected
    for node in graph.nodes():
        graph.nodes[node]['state'] = 0
    
    # Randomly select nodes for initial infection
    initial_infected_nodes = random.sample(list(graph.nodes()), initial_infections)
    for node in initial_infected_nodes:
        graph.nodes[node]['state'] = 1
    
    sequence = []
    
    for _ in range(time_steps):
        # Create a copy of the current graph to represent the next state
        next_graph = graph.copy()
        
        # Disconnect is more likely if there is a higher proportion of infected individuals
        rho = len([node for node in graph.nodes() if graph.nodes[node]['state'] == 1])/len(graph.nodes())
        for node in graph.nodes():
            if graph.nodes[node]['state'] == 0:  # If node is susceptible
                neighbors = list(graph.neighbors(node))
                infected_neighbors = [neighbor for neighbor in neighbors if graph.nodes[neighbor]['state'] == 1]
                # Disconnect from infected neighbors with a certain probability
                for infected_neighbor in infected_neighbors:
                    if random.random() < p_disconnect*rho:
                        next_graph.remove_edge(node, infected_neighbor)
                        
                        # Connect to a susceptible node once it disconnects from an infected node
                        susceptible_nodes = [n for n in graph.nodes() if graph.nodes[n]['state'] == 0 and n != node and not graph.has_edge(node, n)]
                        if susceptible_nodes:  # Check if there are any susceptible nodes to connect to
                            new_friend = random.choice(susceptible_nodes)
                            next_graph.add_edge(node, new_friend)
                    # For every infected neighbor you could get sick
                    if random.random() < beta:
                        next_graph.nodes[node]['state'] =1

                # # If the node has infected neighbors, it can get infected with probability beta
                # if len(infected_neighbors) > 0 and random.random() < beta:
                #     next_graph.nodes[node]['state'] = 1
                
            else:  # If node is infected
                # Node can recover with probability gamma
                if random.random() < gamma:
                    next_graph.nodes[node]['state'] = 0
        
        # Append the graph to the sequence
        sequence.append(next_graph)
        
        # Update the graph to the next state for the next iteration
        graph = next_graph
    
    return sequence

import sys

def main(array_id):
    # Calculate the range based on array_id
    start = (array_id - 1) * 1_000  # Assuming each job processes 1000 sequences
    end = array_id * 1_000
    infection_prob = 0.1
    recovery_prob = 0.1
    initial_infections = 5
    prob_disconnect = .4
    
    sequences_to_save = []
    sequences_to_save = []
    for _ in tqdm(range(start, end)):
        n = random.randint(20, 30)
        G = nx.erdos_renyi_graph(n, 0.4)
        sequence = sis_dynamics_with_dynamic_edges(G, infection_prob, recovery_prob, initial_infections, prob_disconnect)
        pyg_sequence = [pyg_transform(graph) for graph in sequence] # list of data objects
        pyg_sequence = [get_sequence(spectral_transform(one_hot_transform(graph))) for graph in pyg_sequence]# 
        sequences_to_save.append(pyg_sequence)

    # Save to a unique file based on array_id
    torch.save(sequences_to_save, f'small_dataset/sis_sequences_{array_id}.pt')

if __name__ == "__main__":
    array_id = int(sys.argv[1])  # Get SLURM array job index from command line
    main(array_id)

