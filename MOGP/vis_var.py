import numpy as np
import matplotlib.pyplot as plt
'''
# Load the predicted variance data
predicted_var = np.load('/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/mipnerf360/360_v2/flowers/gp/ftest_var.npy')[0]
predicted_variance = np.array(predicted_var).reshape(-1, 6)

# Extract variance for x, y, z, r, g, b
r_var = predicted_variance[:, 3]
g_var = predicted_variance[:, 4]
b_var = predicted_variance[:, 5]
rgb_mean = (r_var + g_var + b_var) / 3

# Filtering based on 70th percentile
threshold = np.percentile(rgb_mean, 74)
filtered_indices = rgb_mean <= threshold
filtered_variance = predicted_variance[filtered_indices]

# Load predicted means and apply the same filtering
predict_mean = np.load('/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/mipnerf360/360_v2/flowers/gp/fmean.npy')[0]
predict_mean = np.array(predict_mean).reshape(-1, 6)
filtered_means = predict_mean[filtered_indices]

#filtered_variance = predicted_variance
# Define colors with high contrast
colors = ['#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#46F0F0']
line_styles = ['-', '--', '-.', ':', '-', '--']

# Create a figure with two subplots (one for x, y, z and another for r, g, b)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=600, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Plot variance for spatial coordinates (x, y, z) in the top subplot
axes[0].plot(range(len(filtered_variance)), filtered_variance[:, 0], label='Variance in x', color=colors[0], linestyle=line_styles[0], linewidth=2)
axes[0].plot(range(len(filtered_variance)), filtered_variance[:, 1], label='Variance in y', color=colors[1], linestyle=line_styles[1], linewidth=2)
axes[0].plot(range(len(filtered_variance)), filtered_variance[:, 2], label='Variance in z', color=colors[2], linestyle=line_styles[2], linewidth=2)
#axes[0].set_ylabel('Spatial Variance (x, y, z)', fontsize=20, fontweight='bold')
axes[0].legend(fontsize=14, loc='upper right', frameon=False)

# Plot variance for color channels (r, g, b) in the bottom subplot
axes[1].plot(range(len(filtered_variance)), filtered_variance[:, 3], label='Variance in r', color=colors[3], linestyle=line_styles[3], linewidth=2)
axes[1].plot(range(len(filtered_variance)), filtered_variance[:, 4], label='Variance in g', color=colors[4], linestyle=line_styles[4], linewidth=2)
axes[1].plot(range(len(filtered_variance)), filtered_variance[:, 5], label='Variance in b', color=colors[5], linestyle=line_styles[5], linewidth=2)
#axes[1].set_ylabel('Color Variance (r, g, b)', fontsize=20, fontweight='bold')
axes[1].legend(fontsize=14, loc='upper right', frameon=False)

# Shared x-axis customization
#axes[1].set_xlabel('Sample Index', fontsize=16, fontweight='bold')
#Filtered
# Title for the whole figure
#fig.suptitle('Variance for Spatial and Color Components (MipNeRF 360 - flowers)', fontsize=20, fontweight='bold')

# Adjust spacing between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)  # Reduce space between subplots

# Save as high-quality PNG
plt.savefig('/home/staff/zhihao/Downloads/3dgs/mogp/var_figure/room/f_after.png', dpi=600, bbox_inches='tight')
plt.show()
'''


import numpy as np
import matplotlib.pyplot as plt

# Load the predicted variance data
predicted_var = np.load('/home/staff/zhihao/Downloads/3dgs/mogp/gp_evaluation/mipnerf360/360_v2/flowers/gp/ftest_var.npy')[0]
predicted_variance = np.array(predicted_var).reshape(-1, 6)

# Extract variance for x, y, z, r, g, b
variance_components = {
    'Variance in x': predicted_variance[:, 0],
    'Variance in y': predicted_variance[:, 1],
    'Variance in z': predicted_variance[:, 2],
    'Variance in r': predicted_variance[:, 3],
    'Variance in g': predicted_variance[:, 4],
    'Variance in b': predicted_variance[:, 5]
}

# Define colors for better visualization
colors = ['#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#46F0F0']

# Create a figure with 6 subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(12, 9), dpi=300)

# Plot each variance component in a separate subplot (2 per row)
for i, (key, values) in enumerate(variance_components.items()):
    ax = axes[i // 2, i % 2]  # Map to subplot grid (3 rows, 2 columns)
    ax.plot(range(len(values)), values, color=colors[i], linewidth=2)
    ax.set_title(key, fontsize=14)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Variance')

# Adjust layout for better visibility
plt.tight_layout()

# Save and display the plot
plt.savefig('/home/staff/zhihao/Downloads/3dgs/mogp/var_figure/variance_plots_2perrow.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/staff/zhihao/Downloads/3dgs/mogp/var_figure/variance_plots.svg', bbox_inches='tight', format='svg')
plt.show()
