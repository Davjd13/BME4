import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from ts2vg import HorizontalVG
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def MLPNN(audio_files):
    hvg = HorizontalVG(weighted="abs_slope")
    all_layer_gexf = "/home/davjd313/MultilayerNetwork (BME_4)/Result/MLP_all_layer.gexf"
    all_layer_edgelist = "/home/davjd313/MultilayerNetwork (BME_4)/Result/MLP_all_layer.edgelist"
    intra_layer_edgelist = "/home/davjd313/MultilayerNetwork (BME_4)/Result/MLP_intra_layer.edgelist"
    inter_layer_edgelist = "/home/davjd313/MultilayerNetwork (BME_4)/Result/MLP_inter_layer.edgelist"

    # # Clear the output folder
    # if os.path.exists(all_layer_gexf):
    #     for file in os.listdir(all_layer_gexf):
    #         file_path = os.path.join(all_layer_gexf, file)
    #         try:
    #             if os.path.isfile(file_path):
    #                 os.unlink(file_path)  # Remove file
    #             elif os.path.isdir(file_path):
    #                 os.rmdir(file_path)  # Remove empty directory
    #         except Exception as e:
    #             print(f"Error deleting {file_path}: {e}")
    # else:
    #     os.makedirs(all_layer_gexf) 

    if os.path.exists(all_layer_edgelist):
        for file in os.listdir(all_layer_edgelist):
            file_path = os.path.join(all_layer_edgelist, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directory
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(all_layer_edgelist)

    if os.path.exists(intra_layer_edgelist):
        for file in os.listdir(intra_layer_edgelist):
            file_path = os.path.join(intra_layer_edgelist, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directory
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(intra_layer_edgelist)

    if os.path.exists(inter_layer_edgelist):
        for file in os.listdir(inter_layer_edgelist):
            file_path = os.path.join(inter_layer_edgelist, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directory
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(inter_layer_edgelist)

    # Loop through each audio file
    for audio_file in audio_files:
        print("Processing", audio_file)
        data = pd.read_csv(os.path.join(audio_folder, audio_file))
        data = data.drop(data.columns[0], axis=1)

        # Loop over each value of n
        for n in [256, 512, 1024]:
            A = {}
            layers = []
            columns_to_use = [0, 1, 2, 3]
            selected_columns = data.columns[columns_to_use]
            # Generate intra-layer adjacency matrices
            for idx, col in enumerate(selected_columns):
                layer = data[col][:n]
                hvg.build(layer)
                G_i = hvg.as_networkx()
                layers.append(layer)
                
                # Convert intra-layer graph to adjacency matrix and store
                A[f'A_{idx}_{idx}'] = nx.to_numpy_array(G_i, nodelist=range(n))

                # Save intra-layer graph as .edgelist file
                intra_layer_path_edgelist = os.path.join(
                    intra_layer_edgelist,
                    f'MLP_{audio_file.split(".")[0]}_{n}_{col}.weighted.edgelist'
                )
                nx.write_weighted_edgelist(G_i, intra_layer_path_edgelist)

            # Generate inter-layer adjacency matrices
            for i, layer_i in enumerate(layers):
                layer_i_scaled = MinMaxScaler().fit_transform(layer_i.values.reshape(-1, 1)).flatten()
                for j, layer_j in enumerate(layers):
                    if i != j:
                        layer_j_scaled = MinMaxScaler().fit_transform(layer_j.values.reshape(-1, 1)).flatten()
                        max_ij = np.maximum(layer_i_scaled, layer_j_scaled)
                        A_ij = np.zeros((n, n), dtype=float)
                        for x_idx, x in enumerate(layer_i):
                            for y_idx, y in enumerate(layer_j):
                                if x_idx != y_idx:
                                    if abs(x_idx - y_idx) == 1:
                                        A_ij[x_idx, y_idx] = abs(float((y - x) / (y_idx - x_idx)))
                                    else:
                                        z_between = max_ij[min(x_idx, y_idx) + 1: max(x_idx, y_idx)]
                                        if all(z < min(layer_i_scaled[x_idx], layer_j_scaled[y_idx]) for z in z_between):
                                            A_ij[x_idx, y_idx] = abs(float((y - x) / (y_idx - x_idx)))
                        A[f'A_{i}_{j}'] = A_ij

            # Process inter-layer matrices
            inter_layer_keys = [key for key in A.keys() if '_' in key and key.split('_')[1] != key.split('_')[2]]

            while inter_layer_keys:
                # Take the first key (e.g., 'A_1_2')
                key_ij = inter_layer_keys.pop(0)
                i, j = map(int, key_ij[2:].split('_'))
                
                # Find the reverse key (e.g., 'A_2_1')
                key_ji = f'A_{j}_{i}'
                if key_ji in inter_layer_keys:
                    inter_layer_keys.remove(key_ji)  # Remove it from processing list
                
                # Extract the matrices
                A_ij = A[key_ij]
                A_ji = A[key_ji]
                
                # Create the inter-layer matrix
                A_inter_ij = np.block([
                    [np.zeros((n, n)), A_ij],
                    [A_ji, np.zeros((n, n))]
                ])
                
                # Save the inter-layer matrix as a graph
                G_inter_ij = nx.from_numpy_array(A_inter_ij, create_using=nx.Graph)
                inter_layer_path_edgelist = os.path.join(
                    inter_layer_edgelist,
                    f'MLP_{audio_file.split(".")[0]}_{n}_{selected_columns[i]}_{selected_columns[j]}.weighted.edgelist'
                )
                nx.write_weighted_edgelist(G_inter_ij, inter_layer_path_edgelist)

            # Create supra-adjacency matrix
            num_layers = len(layers)
            supra_adj_matrix = np.zeros((num_layers * n, num_layers * n), dtype=float)

            for (key, matrix) in A.items():
                i, j = map(int, key[2:].split('_'))
                row_offset = i * n
                col_offset = j * n
                supra_adj_matrix[row_offset:row_offset + n, col_offset:col_offset + n] = matrix

            # Generate graph from supra-adjacency matrix
            G = nx.from_numpy_array(supra_adj_matrix, create_using=nx.Graph)

            # Save graph as .edgelist file
            columns_str = "_".join(selected_columns)
            all_layer_path_edgelist = os.path.join(all_layer_edgelist, f'MLP_{audio_file.split(".")[0]}_{n}_{columns_str}.weighted.edgelist')
            nx.write_weighted_edgelist(G, all_layer_path_edgelist)

            # # Assign layer attributes to nodes
            # for node in G.nodes():
            #     layer = "Layer1" if node < n else "Layer2"
            #     G.nodes[node]["layer"] = layer

            # # Assign edge type attributes
            # for u, v, d in G.edges(data=True):
            #     G.edges[u, v]["connection_type"] = "intra-layer" if (u < n and v < n) or (u >= n and v >= n) else "inter-layer"

            # # Save graph as .gexf file 
            # output_path_gexf = os.path.join(all_layer_gexf, f'MLP_{audio_file.split(".")[0]}_{n}_{columns_str}.gexf')
            # nx.write_gexf(G, output_path_gexf)

audio_folder = "/home/davjd313/MultilayerNetwork (BME_4)/Dataset/Audio"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.csv')]

MLPNN(audio_files)

# # Save supra-adjacency matrix to file
# output_file = os.path.join(test_folder, f"{audio_file.split('.')[0]}_supra_adj_matrix_{n}.txt")
# np.savetxt(output_file, supra_adj_matrix, fmt="%.2f")
# print(f"Supra-adjacency matrix for n={n} saved to {output_file}")

# # Visualization of the multilayer network
# pos = {}
# for node in G.nodes:
#     layer = node // n  # Determine the layer based on node index
#     idx = node % n     # Determine the position within the layer
#     pos[node] = (layer * 2, idx)

# plt.figure(figsize=(10, 8))
# nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", edge_color="gray", font_size=10, font_weight="bold")

# # Adjust edge label format to 2 decimal places
# edge_labels = {edge: f"{weight:.2f}" for edge, weight in nx.get_edge_attributes(G, 'weight').items()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# plt.title(f"Multilayer Network Visualization for n={n}")
# plt.show()