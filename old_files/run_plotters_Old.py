import numpy as np
import time
import plotter_utility

if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_exploration_budget.txt',
                                output_filename='../outputs/plots/model_regret_with_exploration_budget.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Time T',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Exploration Budget',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_exploration_budget_deterministic.txt',
                                output_filename='../outputs/plots/model_regret_with_exploration_budget_deterministic.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Time T',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Exploration Budget \nwith Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_diff_in_best_reward.txt',
                                output_filename='../outputs/plots/model_regret_with_diff_in_best_reward.png',
                                x_axis_label_in_data="diff_in_best_reward",
                                label_x_axis='Difference in Best Reward R',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Difference in Best Reward',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_diff_in_best_reward_deterministic.txt',
                                output_filename='../outputs/plots/model_regret_with_diff_in_best_reward_deterministic.png',
                                x_axis_label_in_data="diff_in_best_reward",
                                label_x_axis='Difference in Best Reward R',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Difference in Best Reward \nwith Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_exploration_budget_long_horizon.txt',
                                output_filename='../outputs/plots/model_regret_with_exploration_budget_long_horizon.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Time T',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Exploration Budget For Large Budgets',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='log')
    plotter_utility.plot_figure(
        input_file='../outputs/model_regret_with_exploration_budget_long_horizon_deterministic.txt',
        output_filename='../outputs/plots/model_regret_with_exploration_budget_long_horizon_deterministic.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Normalized Expected Regret',
        plot_title='Plot for Regret with Exploration Budget For Large Budgets \nwith Deterministic Transitions',
        figsize=(8, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_lambda.txt',
                                output_filename='../outputs/plots/model_regret_with_lambda.png',
                                x_axis_label_in_data="lambda",
                                label_x_axis='Lambda Value λ',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Instance Dependent Causal Parameter λ',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')
    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_lambda_deterministic.txt',
                                output_filename='../outputs/plots/model_regret_lambda_deterministic.png',
                                x_axis_label_in_data="lambda",
                                label_x_axis='Lambda Value λ',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Instance Dependent Causal Parameter λ \nwith Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_num_intermediate_contexts.txt',
                                output_filename='../outputs/plots/model_regret_with_num_intermediate_contexts.png',
                                x_axis_label_in_data="num_intermediate_contexts",
                                label_x_axis='Number of Intermediate Contexts',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Number of Intermediate Contexts',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')
    plotter_utility.plot_figure(input_file='../outputs/model_regret_with_num_intermediate_contexts_deterministic.txt',
                                output_filename='../outputs/plots/model_regret_with_num_intermediate_contexts_deterministic.png',
                                x_axis_label_in_data="num_intermediate_contexts",
                                label_x_axis='Number of Intermediate Contexts',
                                label_y_axis='Normalized Expected Regret',
                                plot_title='Plot for Regret with Number of Intermediate Contexts \nwith Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    # ----------------------
    # NOW THE TOGETHER PLOTS
    # ----------------------

    plotter_utility.plot_figure_deterministic_stochastic(input_file='../outputs/model_regret_with_exploration_budget.txt',
                                                         input_file2='outputs/model_regret_with_exploration_budget_deterministic.txt',
                                                         output_filename='../outputs/plots/model_regret_with_exploration_budget_together.png',
                                                         x_axis_label_in_data="exploration_budget",
                                                         label_x_axis='Exploration Time T',
                                                         label_y_axis='Normalized Expected Regret',
                                                         plot_title='Plot for Regret with Exploration Budget',
                                                         plot_title2='Plot for Regret with Exploration Budget \nwith Deterministic Transitions',
                                                         figsize=(16, 8),
                                                         font_family='serif', general_font_size=11,
                                                         axes_label_font_size=14, title_font_size=16,
                                                         markers=None,
                                                         xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(input_file='../outputs/model_regret_with_diff_in_best_reward.txt',
                                                         input_file2='outputs/model_regret_with_diff_in_best_reward_deterministic.txt',
                                                         output_filename='../outputs/plots/model_regret_with_diff_in_best_reward_together.png',
                                                         x_axis_label_in_data="diff_in_best_reward",
                                                         label_x_axis='Difference in Best Reward R',
                                                         label_y_axis='Normalized Expected Regret',
                                                         plot_title='Plot for Regret with Difference in Best Reward',
                                                         plot_title2='Plot for Regret with Difference in Best Reward \nwith Deterministic Transitions',
                                                         figsize=(16, 8),
                                                         font_family='serif', general_font_size=11,
                                                         axes_label_font_size=14, title_font_size=16,
                                                         markers=None,
                                                         xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='../outputs/model_regret_with_exploration_budget_long_horizon.txt',
        input_file2='outputs/model_regret_with_exploration_budget_long_horizon_deterministic.txt',
        output_filename='../outputs/plots/model_regret_with_exploration_budget_long_horizon_together.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Normalized Expected Regret',
        plot_title='Plot for Regret with Exploration Budget \nFor Large Budgets',
        plot_title2='Plot for Regret with Instance Dependent \nFor Large Budgets with Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure_deterministic_stochastic(input_file='../outputs/model_regret_with_lambda.txt',
                                                         input_file2='outputs/model_regret_with_lambda_deterministic.txt',
                                                         output_filename='../outputs/plots/model_regret_with_lambda_together.png',
                                                         x_axis_label_in_data="lambda",
                                                         label_x_axis='Lambda Value λ',
                                                         label_y_axis='Normalized Expected Regret',
                                                         plot_title='Plot for Regret with Instance Dependent \nCausal Parameter λ',
                                                         plot_title2='Plot for Regret with Instance Dependent \nCausal Parameter λ with Deterministic Transitions',
                                                         figsize=(16, 8),
                                                         font_family='serif', general_font_size=11,
                                                         axes_label_font_size=14, title_font_size=16,
                                                         markers=None,
                                                         xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='../outputs/model_regret_with_num_intermediate_contexts.txt',
        input_file2='outputs/model_regret_with_num_intermediate_contexts_deterministic.txt',
        output_filename='../outputs/plots/model_regret_with_num_intermediate_contexts_together.png',
        x_axis_label_in_data="num_intermediate_contexts",
        label_x_axis='Number of Intermediate Contexts',
        label_y_axis='Normalized Expected Regret',
        plot_title='Plot for Regret with Number of Intermediate Contexts',
        plot_title2='Plot for Regret with Number of Intermediate Contexts \nwith Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
