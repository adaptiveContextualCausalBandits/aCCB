import matplotlib.pyplot as plt
import pandas as pd

# Reading the data from the file
df = pd.read_csv('outputs/model_regret_with_exploration_budget.txt', sep='\t')


# Plotting the data
plt.figure(figsize=(8,8), dpi=150)
# Setting a different font style
plt.rc('font', family='serif', size=11)


# Different marker styles
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x', 'D', 'h', 'H']


# Using a different color scheme from the 'tab10' colormap
colors = plt.cm.Set2.colors


for i, column in enumerate(df.columns[1:]):
    # color = 'magenta' if column == 'convex_explorer' else colors[i % len(colors)]
    color = colors[i % len(colors)]
    plt.plot(df["exploration_budget"], df[column], label=column, marker=markers[i % len(markers)], color=color)

plt.ylabel('Normalized Expected Regret', fontsize=14)
plt.xlabel('Exploration Time T', fontsize=14)
plt.title('Plot for Regret with Exploration Budget', fontsize=16, fontweight='bold', color='navy')
plt.legend()
plt.grid(True)
plt.xscale('log')  # Logarithmic scale for x-axis
plt.show()

# Resetting the font style to default for future plots
plt.rcdefaults()