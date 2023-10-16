import numpy as np
import imageio
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
from gen_dataset import sis_dynamics_with_dynamic_edges


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

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

def draw_and_save_graphs_moving_linear(sequence, frame_dir, sequence_idx, num_interpolated_frames=10):
    # Get initial positions for nodes in the first graph
    pos = nx.spring_layout(sequence[0], k=.15)
    velocities = {node: np.array([0, 0]) for node in sequence[0].nodes()}  # Initialize velocities

    frame_count = 0
    interpolated_pos = None
    for idx in range(len(sequence) - 1):
        current_graph = sequence[idx]
        next_graph = sequence[idx + 1]

        # Compute new positions based on the spring layout for the next graph, starting from the old positions
        new_pos = nx.spring_layout(next_graph, pos=pos, k=1)

        # Generate interpolated frames
        for i in range(num_interpolated_frames):
            t = (i + 1) / num_interpolated_frames

            # Interpolate positions with momentum and friction
            interpolated_pos = {}
            for node in current_graph.nodes():
                interpolated_pos[node] = np.array(pos[node]) + t * (np.array(new_pos[node]) - np.array(pos[node]))

            plt.figure(figsize=(8, 6))
            if i < 5:
                susceptible_nodes = [node for node, data in current_graph.nodes(data=True) if data['state'] == 0]
                nx.draw_networkx_nodes(current_graph, interpolated_pos, nodelist=susceptible_nodes, node_color='blue', node_size=200)
                infected_nodes = [node for node, data in current_graph.nodes(data=True) if data['state'] == 1]
                nx.draw_networkx_nodes(current_graph, interpolated_pos, nodelist=infected_nodes, node_color='red', node_size=200)
            else:
                susceptible_nodes = [node for node, data in next_graph.nodes(data=True) if data['state'] == 0]
                nx.draw_networkx_nodes(next_graph, interpolated_pos, nodelist=susceptible_nodes, node_color='blue', node_size=200)
                infected_nodes = [node for node, data in next_graph.nodes(data=True) if data['state'] == 1]
                nx.draw_networkx_nodes(next_graph, interpolated_pos, nodelist=infected_nodes, node_color='red', node_size=200)

            nx.draw_networkx_edges(next_graph, interpolated_pos)
            frame_path = os.path.join(frame_dir, f'sequence_{sequence_idx}_frame_{frame_count}.png')
            plt.savefig(frame_path)
            plt.close()
            frame_count += 1

        # Update the positions for the next iteration
        pos = interpolated_pos.copy()

def draw_and_save_graphs_moving_with_velocity(sequence, frame_dir, sequence_idx, alpha=1.0, num_interpolated_frames=10, friction=0.8):
    # Get initial positions for nodes in the first graph
    pos = nx.spring_layout(sequence[0], k=.15)
    velocities = {node: np.array([0, 0]) for node in sequence[0].nodes()}  # Initialize velocities

    frame_count = 0
    for idx in range(len(sequence) - 1):
        current_graph = sequence[idx]
        next_graph = sequence[idx + 1]

        # Compute new positions based on the spring layout for the next graph, starting from the old positions
        new_pos = nx.spring_layout(next_graph, pos=pos, k=1)

        # Generate interpolated frames
        for i in range(num_interpolated_frames):
            t = (i + 1) / num_interpolated_frames
            interpolation_factor = alpha * t

            interpolated_pos = {}
            for node in current_graph.nodes():
                # Calculate acceleration (difference between current and target positions)
                acceleration = np.array(new_pos[node]) - np.array(pos[node])
                
                # Update velocity based on acceleration and apply friction
                velocities[node] = velocities[node] * friction + acceleration * interpolation_factor
                
                # Update node position based on velocity
                interpolated_pos[node] = np.array(pos[node]) + velocities[node]

            plt.figure(figsize=(8, 6))
            susceptible_nodes = [node for node, data in next_graph.nodes(data=True) if data['state'] == 0]
            nx.draw_networkx_nodes(next_graph, interpolated_pos, nodelist=susceptible_nodes, node_color='blue', node_size=200)
            infected_nodes = [node for node, data in next_graph.nodes(data=True) if data['state'] == 1]
            nx.draw_networkx_nodes(next_graph, interpolated_pos, nodelist=infected_nodes, node_color='red', node_size=200)
            nx.draw_networkx_edges(next_graph, interpolated_pos)
            frame_path = os.path.join(frame_dir, f'sequence_{sequence_idx}_frame_{frame_count}.png')
            plt.savefig(frame_path)
            plt.close()
            frame_count += 1

            # Update the positions for the next iteration
            pos[node] = interpolated_pos[node]




if __name__ == "__main__":
    infection_prob = 0.1
    recovery_prob = 0.1
    initial_infections = 5
    prob_disconnect = .65

    # Generate 5 sequences again with varying node sizes between 20-30
    node_count = random.randint(70, 80)
    graph = nx.barabasi_albert_graph(node_count, 2).to_undirected()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    sequence = sis_dynamics_with_dynamic_edges(graph, infection_prob, recovery_prob, initial_infections,prob_disconnect,time_steps=160)
    

    frame_dir = "./sis_frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    draw_and_save_graphs_moving_linear(sequence, frame_dir, 0)


    # List frames in order
    frame_files = sorted(os.listdir(frame_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    frame_paths = [os.path.join(frame_dir, fname) for fname in frame_files]

    # Create a video from frames
    video_path = "./video_moving.mp4"
    imageio.mimwrite(video_path, [imageio.imread(frame) for frame in frame_paths], fps=20)