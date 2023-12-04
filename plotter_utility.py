import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


def plot_figure(input_file='outputs/model_regret_with_exploration_budget.txt',
                output_filename='outputs/plots/model_regret_with_exploration_budget.png',
                x_axis_label_in_data="exploration_budget",
                label_x_axis='Exploration Time T', label_y_axis='Normalized Expected Regret',
                plot_title='Plot for Regret with Exploration Budget',
                figsize=(8, 8),
                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                markers=None,
                xscale=None):
    # Reading the data from the file
    if markers is None:
        markers = ['o', 'H', 'D', 'h', 'v', '^', 'p', '*', '+', 'x', 'D', 'h', 'H']
    df = pd.read_csv(input_file, sep='\t')

    # Plotting the data
    plt.figure(figsize=figsize, dpi=150)
    # Setting a different font style
    plt.rc('font', family=font_family, size=general_font_size)

    # Different marker styles
    markers = markers

    # Using a different color scheme from the 'tab10' colormap
    # noinspection PyUnresolvedReferences
    # colors = plt.cm.Set2.colors
    # colors = plt.cm.Set1.colors
    colors = ['#911EB4',  # Purple
              '#3CB44B',  # Grass Green
              '#F58231',  # Bright Orange
              '#FFE119',  # Sunshine Yellow
              '#E6194B',  # Vivid Red
              '#4363D8',  # Deep Blue
              '#42D4F4']  # Cyan

    for i, column in enumerate(df.columns[1:]):
        # color = 'magenta' if column == 'convex_explorer' else colors[i % len(colors)]
        color = colors[i % len(colors)]
        plt.plot(df[x_axis_label_in_data], df[column], label=column, marker=markers[i % len(markers)], color=color)

    plt.xlabel(label_x_axis, fontsize=14)
    plt.ylabel(label_y_axis, fontsize=14)
    plt.title(plot_title, fontsize=title_font_size, fontweight='bold', color='navy')
    plt.legend()
    plt.grid(True)
    if xscale:
        plt.xscale(xscale)  # Logarithmic scale for x-axis

    # function to show the plot
    plt.savefig(output_filename, dpi=199)

    # plt.show()
    plt.close()
    # Resetting the font style to default for future plots
    plt.rcdefaults()


def plot_figure_prob_best_simple_regret(input_file1='outputs/prob_best_intervention_with_exploration_budget.txt',
                                      input_file2='outputs/simple_regret_with_exploration_budget.txt',
                                      output_filename='outputs/plots/simple_regret_and_prob_best_with_exploration_budget.png',
                                      x_axis_label_in_data="exploration_budget",
                                      label_x_axis='Exploration Time T',
                                      label_y_axis1='Probability of Best Intervention',
                                      label_y_axis2='Simple Regret',
                                      plot_title1='Plot for Probability of Best Intervention with Exploration Budget',
                                      plot_title2='Plot for Simple Regret with Exploration Budget',
                                      figsize=(16, 8),
                                      font_family='serif', general_font_size=11,
                                      axes_label_font_size=14, title_font_size=16,
                                      markers=None,
                                      xscale='linear'):
    # Reading the data from the file
    if markers is None:
        markers = ['o', 'H', 'D', 'h', 'v', '^', 'p', '*', '+', 'x', 'D', 'h', 'H']
    df1 = pd.read_csv(input_file1, sep='\t')
    df2 = pd.read_csv(input_file2, sep='\t')

    # Plotting the data
    # plt.figure(figsize=figsize, dpi=150)
    # Plotting the two figures side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Setting a different font style
    plt.rc('font', family=font_family, size=general_font_size)

    # Different marker styles
    markers = markers

    # Using a different color scheme from the 'tab10' colormap
    # noinspection PyUnresolvedReferences
    # colors = plt.cm.Set2.colors
    # colors = plt.cm.Set1.colors
    colors = ['#911EB4',  # Purple
              '#3CB44B',  # Grass Green
              '#F58231',  # Bright Orange
              '#FFE119',  # Sunshine Yellow
              '#E6194B',  # Vivid Red
              '#4363D8',  # Deep Blue
              '#42D4F4']  # Cyan

    # Setting labels, titles, and other properties for the first and second plot
    ax1.set_xlabel(label_x_axis, fontfamily='serif', fontsize=14)
    ax1.set_ylabel(label_y_axis1, fontfamily='serif', fontsize=14)
    ax1.set_title(plot_title1, fontsize=title_font_size, fontweight='bold', color='navy')
    ax1.grid(True)

    ax2.set_xlabel(label_x_axis, fontfamily='serif', fontsize=14)
    ax2.set_ylabel(label_y_axis2, fontfamily='serif', fontsize=14)
    ax2.set_title(plot_title2, fontsize=title_font_size, fontweight='bold', color='navy')
    ax2.grid(True)

    for i, column in enumerate(df1.columns[1:]):
        # color = 'magenta' if column == 'convex_explorer' else colors[i % len(colors)]
        color = colors[i % len(colors)]
        ax1.plot(df1[x_axis_label_in_data], df1[column], label=column, marker=markers[i % len(markers)], color=color)
    for i, column in enumerate(df1.columns[1:]):
        # color = 'magenta' if column == 'convex_explorer' else colors[i % len(colors)]
        color = colors[i % len(colors)]
        ax2.plot(df2[x_axis_label_in_data], df2[column], label=column, marker=markers[i % len(markers)], color=color)


    # Adding a shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df1.columns) - 1)

    # plt.legend()

    if xscale:
        ax1.set_xscale(xscale)
        ax2.set_xscale(xscale)

    # function to show the plot
    plt.savefig(output_filename, dpi=199)

    # plt.show()
    plt.close()
    # Resetting the font style to default for future plots
    plt.rcdefaults()


def plot_figure_deterministic_stochastic(input_file='outputs/model_regret_with_exploration_budget.txt',
                                         input_file2='outputs/model_regret_with_exploration_budget_deterministic.txt',
                                         output_filename='outputs/plots/model_regret_with_exploration_budget.png',
                                         x_axis_label_in_data="exploration_budget",
                                         label_x_axis='Exploration Time T', label_y_axis='Normalized Expected Regret',
                                         plot_title='Plot for Regret with Exploration Budget',
                                         plot_title2='Plot for Regret with Exploration Budget',
                                         figsize=(16, 8),
                                         font_family='serif', general_font_size=11, axes_label_font_size=14,
                                         title_font_size=16,
                                         markers=None,
                                         xscale=None):
    # Reading the data from the file
    if markers is None:
        markers = ['o', 'H', 'D', 'h', 'v', '^', 'p', '*', '+', 'x', 'D', 'h', 'H']
    df = pd.read_csv(input_file, sep='\t')
    df2 = pd.read_csv(input_file2, sep='\t')

    # Plotting the data
    # plt.figure(figsize=figsize, dpi=150)
    # Plotting the two figures side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Setting a different font style
    plt.rc('font', family=font_family, size=general_font_size)

    # Different marker styles
    markers = markers

    # Using a different color scheme from the 'tab10' colormap
    # noinspection PyUnresolvedReferences
    # colors = plt.cm.Set2.colors
    # colors = plt.cm.Set1.colors
    colors = ['#911EB4',  # Purple
              '#3CB44B',  # Grass Green
              '#F58231',  # Bright Orange
              '#FFE119',  # Sunshine Yellow
              '#E6194B',  # Vivid Red
              '#4363D8',  # Deep Blue
              '#42D4F4']  # Cyan

    # Setting labels, titles, and other properties for the first and second plot
    ax1.set_xlabel(label_x_axis, fontfamily='serif')
    ax1.set_ylabel('Normalized Expected Regret', fontfamily='serif')
    ax1.set_title('Data Set 2', fontsize=14, fontweight='bold', color='navy')
    ax1.grid(True)

    ax2.set_xlabel(label_x_axis, fontfamily='serif')
    ax2.set_ylabel('Normalized Expected Regret', fontfamily='serif')
    ax2.set_title('Data Set 2', fontsize=14, fontweight='bold', color='navy')
    ax2.grid(True)

    for i, column in enumerate(df.columns[1:]):
        # color = 'magenta' if column == 'convex_explorer' else colors[i % len(colors)]
        color = colors[i % len(colors)]
        ax1.plot(df[x_axis_label_in_data], df[column], label=column, marker=markers[i % len(markers)], color=color)
    for i, column in enumerate(df.columns[1:]):
        # color = 'magenta' if column == 'convex_explorer' else colors[i % len(colors)]
        color = colors[i % len(colors)]
        ax2.plot(df2[x_axis_label_in_data], df2[column], label=column, marker=markers[i % len(markers)], color=color)

    ax1.set_xlabel(label_x_axis, fontsize=14)
    ax1.set_ylabel(label_y_axis, fontsize=14)
    ax1.set_title(plot_title, fontsize=title_font_size, fontweight='bold', color='navy')
    ax1.grid(True)

    ax2.set_xlabel(label_x_axis, fontsize=14)
    ax2.set_ylabel(label_y_axis, fontsize=14)
    ax2.set_title(plot_title2, fontsize=title_font_size, fontweight='bold', color='navy')
    ax2.grid(True)

    # Adding a shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df.columns) - 1)

    # plt.legend()

    if xscale:
        ax1.set_xscale(xscale)
        ax2.set_xscale(xscale)

    # function to show the plot
    plt.savefig(output_filename, dpi=199)

    # plt.show()
    plt.close()

    # Resetting the font style to default for future plots
    plt.rcdefaults()


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    plot_figure(input_file='outputs/model_regret_with_exploration_budget_deterministic.txt',
                output_filename='outputs/plots/model_regret_with_exploration_budget_deterministic.png',
                x_axis_label_in_data="exploration_budget",
                label_x_axis='Exploration Time T',
                label_y_axis='Normalized Expected Regret',
                plot_title='Plot for Regret with Exploration Budget',
                figsize=(8, 8),
                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                markers=None,
                xscale='linear')

    plot_figure_deterministic_stochastic(input_file='outputs/model_regret_with_exploration_budget.txt',
                                         input_file2='outputs/model_regret_with_exploration_budget_deterministic.txt',
                                         output_filename='outputs/plots/model_regret_with_exploration_budget_together.png',
                                         x_axis_label_in_data="exploration_budget",
                                         label_x_axis='Exploration Time T',
                                         label_y_axis='Normalized Expected Regret',
                                         plot_title='Plot for Regret with Exploration Budget',
                                         figsize=(16, 8),
                                         font_family='serif', general_font_size=11, axes_label_font_size=14,
                                         title_font_size=16,
                                         markers=None,
                                         xscale='linear')

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
