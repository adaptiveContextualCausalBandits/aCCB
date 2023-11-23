import matplotlib.pyplot as plt
import pandas as pd

# Creating a DataFrame from the provided data
data = {
    "diff_in_best_reward": [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 100000],
    "roundrobin_roundrobin": [0.960005, 0.924004, 0.888004, 0.662002, 0.162, 0.02, 0.01, 0.002, 0, 0, 0, 0, 0],
    "roundrobin_ucb": [0.962005, 0.956005, 0.798003, 0.612001, 0.347999, 0.12, 0.016, 0, 0, 0, 0, 0, 0],
    "roundrobin_ts": [0.972005, 0.970005, 0.974005, 0.936004, 0.944005, 0.928004, 0.928004, 0.938004, 0.950005, 0.924004, 0.934004, 0.944005, 0.924004],
    "ucb_over_intervention_pairs": [1.000000, 1.000000, 0.998005, 0.996005, 1.000000, 1.000000, 1.000000, 1.000000, 0.998005, 0.990005, 0.874003, 0.892004, 0.886004],
    "ts_over_intervention_pairs": [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
    "convex_explorer": [0.860004, 0.730002, 0.594001, 0.433999, 0.196, 0.044, 0.01, 0.004, 0.002, 0.002, 0, 0, 0]
}

df = pd.DataFrame(data)

# Plotting the data
plt.figure(figsize=(12, 8))

# Different marker styles
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x', 'D', 'h', 'H']


# Using a different color scheme from the 'tab10' colormap
colors = plt.cm.Set2.colors


for i, column in enumerate(df.columns[1:]):
    # color = 'magenta' if column == 'convex_explorer' else colors[i % len(colors)]
    color = colors[i % len(colors)]
    plt.plot(df["diff_in_best_reward"], df[column], label=column, marker=markers[i % len(markers)], color=color)

plt.xlabel('Difference in Best Reward', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Plot for Regret with Exploration Budget', fontsize=14, fontweight='bold', color='navy')
plt.legend()
plt.grid(True)
plt.xlim(0, 10000)  # Restricting x-axis to 20000
# plt.xscale('log')  # Logarithmic scale for x-axis
plt.show()