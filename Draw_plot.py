import pandas as pd
import matplotlib.pyplot as plt
import ts_to_vg

# Load the dataset
file_path = "/home/davjd313/MultilayerNetwork (BME_4)/Dataset/Audio/s01_ex01_s01.csv"  # Replace with the correct file path
df = pd.read_csv(file_path)

# Drop the unwanted column (if present)
if 'file' in df.columns:
    df.drop('file', axis=1, inplace=True)

# Choose the column to plot (e.g., 'P4', 'Cz', 'F8', or 'T7')
dat1 = df['Cz'].iloc[15:20].to_numpy()
dat2 = df['T7'].iloc[15:20].to_numpy()
network = ts_to_vg.ts_to_cross_vg(data1 = dat1, data2 = dat2, horizontal=True)
network1 = ts_to_vg.ts_to_vg(dat1, horizontal=True)
network2 = ts_to_vg.ts_to_vg(dat2, horizontal=True)

# ts_to_vg.plot_ts_visibility_2(network=network1, data=dat1, horizontal=True)
ts_to_vg.plot_cross_visibility(network, network1, network2, dat1, dat2, horizontal=True)
# print(ts_to_vg.ts_to_cross_vg(data1 = dat1, data2 = dat2, horizontal=True))
