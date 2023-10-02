import random
import networkx as nx
import matplotlib.pyplot as plt
import os
def sis_dynamics_without_tg(graph, beta, gamma, initial_infections):
    """
    Simulate the SIS model on a graph without converting to Torch Geometric data structures.
    
    Parameters:
    - graph: initial graph with all nodes susceptible.
    - beta: infection probability.
    - gamma: recovery probability.
    - initial_infections: number of initially infected nodes.
    
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
    
    for _ in range(20):
        # Create a copy of the current graph to represent the next state
        next_graph = graph.copy()
        
        for node in graph.nodes():
            if graph.nodes[node]['state'] == 0:  # If node is susceptible
                neighbors = list(graph.neighbors(node))
                infected_neighbors = sum(1 for neighbor in neighbors if graph.nodes[neighbor]['state'] == 1)
                
                # If the node has infected neighbors, it can get infected with probability beta
                if infected_neighbors > 0 and random.random() < beta:
                    next_graph.nodes[node]['state'] = 1
            else:  # If node is infected
                # Node can recover with probability gamma
                if random.random() < gamma:
                    next_graph.nodes[node]['state'] = 0
        
        # Append the graph to the sequence
        sequence.append(next_graph)
        
        # Update the graph to the next state for the next iteration
        graph = next_graph
    
    return sequence


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
# Redefining the parameters
beta = 0.3
gamma = 0.1
initial_infections = 3

# Generate 5 sequences again with varying node sizes between 15-30
sequences_without_tg = []
for _ in range(5):
    node_count = random.randint(15, 30)
    graph = nx.erdos_renyi_graph(node_count, 0.5)
    sequence = sis_dynamics_without_tg(graph, beta, gamma, initial_infections)
    sequences_without_tg.append(sequence)

# frame_dir = "./sis_frames"
# if not os.path.exists(frame_dir):
#     os.makedirs(frame_dir)

# draw_and_save_graphs(sequences_without_tg[0], frame_dir, 0)

# import imageio

# # List frames in order
# frame_files = sorted(os.listdir(frame_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
# frame_paths = [os.path.join(frame_dir, fname) for fname in frame_files]

# # Create a video from frames
# video_path = "./video.mp4"
# imageio.mimwrite(video_path, [imageio.imread(frame) for frame in frame_paths], fps=2)