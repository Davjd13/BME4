import matplotlib.pyplot as plt
import numpy as np

# X-axis values
x_axis = [512, 1024, 2048, 4096]

# Results for each method
hvg_svm = [67.67, 69.11, 73.78, 71.78]
multilayer_svm = [67.90, 69.89, 70.27, 70.75]
multilayer_mlpnn = [60.98, 69.32, 65.53, 64.77]
multiplex_svm = [67.92, 71.04, 69.58, 72.71]
multiplex_mlpnn = [68.33, 64.17, 75.83, 72.50]


# hvg_svm = [53.33, 72.50, 73.33, 56.67]
# multilayer_svm = [55.95, 52.31, 56.52, 55.10]
# multilayer_mlpnn = [53.41, 52.27, 57.95, 50.00]
# multiplex_svm = [58.75, 58.13, 53.75, 60.00]
# multiplex_mlpnn = [62.50, 55.00, 62.50, 50.00]
# Set width of bars
barWidth = 0.15
r1 = np.arange(len(x_axis))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Create the bars - SVM methods in blue shades, MLPNN methods in yellow shades
# HVG_SVM with faded color (older result)
hvg_bars = ax.bar(r1, hvg_svm, width=barWidth, edgecolor='black',
                 color='lightblue', alpha=0.5, label='HVG_SVM')

# Other SVM methods in blue
multilayer_svm_bars = ax.bar(r2, multilayer_svm, width=barWidth, edgecolor='black',
                            color='royalblue', label='Multilayer_SVM')
multiplex_svm_bars = ax.bar(r4, multiplex_svm, width=barWidth, edgecolor='black',
                           color='blue', label='Multiplex_SVM')

# MLPNN methods in yellow
multilayer_mlpnn_bars = ax.bar(r3, multilayer_mlpnn, width=barWidth, edgecolor='black',
                              color='lightyellow', label='Multilayer_MLPNN')
multiplex_mlpnn_bars = ax.bar(r5, multiplex_mlpnn, width=barWidth, edgecolor='black',
                             color='yellow', label='Multiplex_MLPNN')

# Add labels, title and axes ticks
ax.set_xlabel('Nodes', fontsize=15)  # Increased font size
ax.set_ylabel('Accuracy (%)', fontsize=15)  # Increased font size
ax.set_xticks([r + barWidth*2 for r in range(len(x_axis))])
ax.set_xticklabels(x_axis, fontsize=25)  # Increased tick font size
ax.set_ylim(50, 80)
ax.tick_params(axis='y', labelsize=25)  # Increased y-axis tick font size

# Add grid to y-axis
ax.yaxis.grid(True)

# Set up a legend inside the plot with larger font
ax.legend(loc='upper right', fontsize=15)  # Increased legend font size and moved inside

# Remove the vertical line that was appearing on the left
# This line was causing the "mysterious black line" issue, so I've removed it:
# ax.plot([0-barWidth, 0-barWidth], [0, ax.get_ylim()[1]], 'k-', lw=1)

plt.tight_layout()
plt.show()