#plotting:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set custom colors for the plot
colors = ['#9370DB', '#32CD32', '#1E90FF']

# 3D Plot: Voltage, Current, and Temperature with Color-Coded Cells
# Adjust the data to sample only 1%
rpt_sample = rpt_sample.sample(frac=0.0001, random_state=42)

# 3D Plot: Voltage, Current, and Temperature with the data
def plot_3d(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use different colors for each cell
    cell_colors = {'Cell1': colors[0], 'Cell2': colors[1], 'Cell3': colors[2]}
    
    for cell in data['Cell'].unique():
        subset = data[data['Cell'] == cell]
        ax.scatter(subset['Voltage'], subset['Current'], subset['Temperature'],
                   color=cell_colors[cell], label=cell, s=50)

    # Labeling the axes
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (A)')
    ax.set_zlabel('Temperature (°C)')
    ax.set_title('3D Scatter Plot: Voltage, Current, and Temperature by Cell (1% Sample)')

    # Legend
    ax.legend(title='Cell', loc='upper right')

    plt.show()

# Apply the 3D plot to the the data sample
plot_3d(rpt_sample)

import matplotlib.pyplot as plt
import seaborn as sns

# Custom Colors
colors = ['#9370DB', '#32CD32', '#1E90FF']

# Function to plot time series for a specific dataset and cell with custom colors
def plot_time_series(data, dataset_name):
    plt.figure(figsize=(14, 10))

    # Plot Voltage vs. Time
    plt.subplot(2, 2, 1)
    sns.lineplot(x='Time', y='Voltage', data=data, hue='Cell', palette=colors)
    plt.title(f'Voltage vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Voltage (V)')

    # Plot Current vs. Time
    plt.subplot(2, 2, 2)
    sns.lineplot(x='Time', y='Current', data=data, hue='Cell', palette=colors)
    plt.title(f'Current vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Current (A)')

    # Plot Energy vs. Time
    plt.subplot(2, 2, 3)
    sns.lineplot(x='Time', y='Energy', data=data, hue='Cell', palette=colors)
    plt.title(f'Energy vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Energy (Wh)')

    # Plot Temperature vs. Time
    plt.subplot(2, 2, 4)
    sns.lineplot(x='Time', y='Temperature', data=data, hue='Cell', palette=colors)
    plt.title(f'Temperature vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')

    plt.tight_layout()
    plt.show()

# Sample 1% of the RPT and Aging data for clearer plots
rpt_sample = rpt_sample.sample(frac=0.00001, random_state=4)
aging_sample = aging_sample.sample(frac=0.000009, random_state=4)

# Plot time series for RPT and Aging datasets with custom colors
plot_time_series(rpt_sample, 'RPT Dataset')
plot_time_series(aging_sample, 'Aging Dataset')



###################################
# Example: Plotting Energy vs Voltage vs Current for Aging Data (Cell1)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np

# Function to create a 3D surface plot for Energy vs Voltage vs Current
def plot_3d_surface_energy(data, feature_x, feature_y, feature_z, title, color):
    # Create grid data for plotting
    grid_x, grid_y = np.mgrid[data[feature_x].min():data[feature_x].max():100j, 
                              data[feature_y].min():data[feature_y].max():100j]
    
    # Interpolate the z values (Energy) for the grid
    grid_z = griddata((data[feature_x], data[feature_y]), data[feature_z], (grid_x, grid_y), method='cubic')
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm', edgecolor='none', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(f'{feature_x} (V)', labelpad=15)
    ax.set_ylabel(f'{feature_y} (A)', labelpad=15)
    ax.set_zlabel(f'{feature_z} (Energy)', labelpad=15)
    ax.set_title(f'{title}')
    
    # Adjust layout to make space for color bar and prevent overlap
    fig.tight_layout()
    
    # Add color bar with space adjustment
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.15)  # Adjust padding here
    cbar.set_label(f'{feature_z} (Energy)')
    
    plt.show()
subset = rpt_sample[rpt_sample['Cell'] == 'Cell1']
plot_3d_surface_energy(subset, 'Voltage', 'Current', 'Energy', '3D Surface Plot: Energy vs Voltage vs Current', 'coolwarm')

# Example: Plotting Energy vs Voltage vs Current for Cell1 in Aging Data
aging_subset = aging_sample[aging_sample['Cell'] == 'Cell1']
plot_3d_surface_energy(aging_subset, 'Voltage', 'Current', 'Energy', '3D Surface Plot: Energy vs Voltage vs Current (Aging Data)', 'coolwarm')
#################################
# Plot separated time-series plots for each cell in the RPT and Aging samples
rpt_sample = rpt_sample.sample(frac=0.0001, random_state=50)
aging_sample = aging_sample.sample(frac=0.0001, random_state=50)

# Updated function to exclude temperature plot and include 3D plot in the 2x2 layout
def plot_time_series_with_3d(data, dataset_name, plot_3d_func):
    fig = plt.figure(figsize=(14, 10))

    # Plot Voltage vs. Time
    plt.subplot(2, 2, 1)
    sns.lineplot(x='Time', y='Voltage', data=data, hue='Cell', palette=colors)
    plt.title(f'Voltage vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Voltage (V)')

    # Plot Current vs. Time
    plt.subplot(2, 2, 2)
    sns.lineplot(x='Time', y='Current', data=data, hue='Cell', palette=colors)
    plt.title(f'Current vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Current (A)')

    # Plot Energy vs. Time
    plt.subplot(2, 2, 3)
    sns.lineplot(x='Time', y='Energy', data=data, hue='Cell', palette=colors)
    plt.title(f'Energy vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Energy (Wh)')

    # Insert the 3D plot in the last quadrant
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    plot_3d_func(ax_3d, data)  # Pass the ax and data to the provided 3D plotting function

    plt.tight_layout()
    plt.show()

# Function to create the 3D plot based on Voltage, Current, and Energy
def plot_3d_energy_voltage_current(ax, data):
    # Create grid data for plotting
    grid_x, grid_y = np.meshgrid(np.linspace(data['Voltage'].min(), data['Voltage'].max(), 50),
                                 np.linspace(data['Current'].min(), data['Current'].max(), 50))

    # Use scipy's griddata to interpolate the Z (Energy) values
    grid_z = griddata((data['Voltage'], data['Current']), data['Energy'], (grid_x, grid_y), method='cubic')

    # Plot surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm', edgecolor='none', alpha=0.8)
    ax.set_title('3D Surface: Energy vs Voltage vs Current')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (A)')
    ax.set_zlabel('Energy (Wh)')

    # Add colorbar for energy
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Plotting with the new setup for both RPT and Aging datasets
plot_time_series_with_3d(rpt_sample, 'RPT Dataset', plot_3d_energy_voltage_current)
plot_time_series_with_3d(aging_sample, 'Aging Dataset', plot_3d_energy_voltage_current)

# Adjusting color palette dynamically based on the number of unique cells in the dataset


#####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel

# Define colors
colors = ['#9370DB', '#32CD32', '#1E90FF']

plt.yscale('log')
# Weibull Trend: Fit and plot Weibull distribution for degradation
def weibull_trend_plot(data, feature, cell_label):
    # Fit Weibull distribution
    shape, loc, scale = weibull_min.fit(data[feature], floc=0)
    weibull_data = weibull_min.rvs(shape, scale=scale, size=len(data[feature]))
    
    # Plot Weibull distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], bins=30, color=colors[0], kde=False, label=f'{feature} Histogram')
    plt.plot(np.sort(data[feature]), np.sort(weibull_data), color=colors[1], label=f'Weibull Fit (Shape={shape:.2f}, Scale={scale:.2f})')
    plt.title(f'Weibull Trend for {feature} in {cell_label}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Granger Causality Test
def check_granger_causality(data, variables, max_lag=3):
    print(f"\nRunning Granger Causality Test with max_lag={max_lag}")
    results = {}
    for var in variables:
        test_result = grangercausalitytests(data[['Energy', var]].dropna(), max_lag, verbose=False)
        results[var] = test_result
    return results

# Causal Graph for Battery Degradation
def plot_causal_graph():
    causal_graph = CausalGraphicalModel(
        nodes=["Current", "Voltage", "Energy", "Temperature"],
        edges=[("Current", "Voltage"), ("Voltage", "Energy"), ("Energy", "Temperature")]
    )
    causal_graph.draw()

# Run Weibull trend for RPT and Aging datasets
def analyze_weibull_trends_and_causality(rpt_sample, aging_sample):
    # Plot Weibull trends for Voltage, Current, Energy for one cell (RPT)
    cells = rpt_sample['Cell'].unique()
    for cell in cells:
        subset = rpt_sample[rpt_sample['Cell'] == cell]
        weibull_trend_plot(subset, 'Voltage', f'Cell {cell}')
        weibull_trend_plot(subset, 'Current', f'Cell {cell}')
        weibull_trend_plot(subset, 'Energy', f'Cell {cell}')

    # Repeat for Aging dataset
    cells_aging = aging_sample['Cell'].unique()
    for cell in cells_aging:
        subset_aging = aging_sample[aging_sample['Cell'] == cell]
        weibull_trend_plot(subset_aging, 'Voltage', f'Cell {cell} (Aging)')
        weibull_trend_plot(subset_aging, 'Current', f'Cell {cell} (Aging)')
        weibull_trend_plot(subset_aging, 'Energy', f'Cell {cell} (Aging)')

    # Check Granger causality for RPT and Aging datasets
    variables = ['Voltage', 'Current', 'Temperature']
    print("Granger Causality Test for RPT Dataset:")
    rpt_causality = check_granger_causality(rpt_sample, variables)
    print("Granger Causality Test for Aging Dataset:")
    aging_causality = check_granger_causality(aging_sample, variables)

    # Plot causal graph
    plot_causal_graph()

# Run the analysis
analyze_weibull_trends_and_causality(rpt_sample, aging_sample)

