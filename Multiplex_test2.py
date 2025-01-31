import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import MLE_functions_v2
from ts2vg import HorizontalVG
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def Multiplex(audio_files):
    hvg1 = HorizontalVG(weighted="abs_slope")
    hvg2 = HorizontalVG(weighted="abs_slope")
    columns = ['P4', 'Cz', 'F8', 'T7']
    column_pairs = [(columns[i], columns[j]) 
                   for i in range(len(columns)) 
                   for j in range(i+1, len(columns))]
    
    results = []

    output_path = "/home/davjd313/MultilayerNetwork (BME_4)/Result/Multiplex_mutual_info.csv"

    for audio_file in audio_files:
        print("Processing", audio_file)
        data = pd.read_csv(os.path.join(audio_folder, audio_file))
        data = data.drop(data.columns[0], axis=1)
        
        for n in [1024, 512, 256]:
            mutual_info_values = []
            
            for col1, col2 in column_pairs:
                layer1 = data[col1][:n]
                layer2 = data[col2][:n]
                hvg1.build(layer1)
                hvg2.build(layer2)
                
                G_layer1 = hvg1.as_networkx()
                G_layer2 = hvg2.as_networkx()

                # Get degree lists
                degree_list_layer1 = MLE_functions_v2.degree_list(G_layer1)
                degree_list_layer2 = MLE_functions_v2.degree_list(G_layer2)

                # Get degree distributions
                empirical_layer1 = MLE_functions_v2.empirical(degree_list_layer1)
                empirical_layer2 = MLE_functions_v2.empirical(degree_list_layer2)

                # Initialize mutual information for this pair
                I = 0

                # Iterate over degrees in layer1 and layer2
                for i, x in enumerate(empirical_layer1[0]):  # Degrees in layer1
                    for j, y in enumerate(empirical_layer2[0]):  # Degrees in layer2
                        P_x = empirical_layer1[2][i]  # Marginal probability P(x)
                        P_y = empirical_layer2[2][j]  # Marginal probability P(y)

                        # Count joint occurrences for degrees x and y
                        N_x_y = sum(1 for k_x, k_y in zip(degree_list_layer1, degree_list_layer2) if k_x == x and k_y == y)
                        P_x_y = N_x_y / len(degree_list_layer1)  # Joint probability

                        # Update mutual information
                        if P_x_y > 0:  # Avoid log(0)
                            I += P_x_y * np.log2(P_x_y / (P_x * P_y))

                mutual_info_values.append(I)
            # Calculate the average mutual information for this region
            average_mutual_info = np.mean(mutual_info_values)
            # Create filename in the correct format
            base_filename = audio_file.replace('.csv', '')
            file_name = f"Multiplex_{base_filename}_{n}_P4_Cz_F8_T7.weighted.edgelist"
            
            # Add row to results
            row = {
                'file': file_name,
                'I_P4_Cz': round(mutual_info_values[0], 2),
                'I_P4_F8': round(mutual_info_values[1], 2),
                'I_P4_T7': round(mutual_info_values[2], 2),
                'I_Cz_F8': round(mutual_info_values[3], 2),
                'I_Cz_T7': round(mutual_info_values[4], 2),
                'I_F8_T7': round(mutual_info_values[5], 2),
                'I_avg': round(average_mutual_info, 2)
            }
            results.append(row)
    
    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print("CSV file created successfully!")

audio_folder = "/home/davjd313/MultilayerNetwork (BME_4)/Dataset/Audio"
audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.csv')])

Multiplex(audio_files)