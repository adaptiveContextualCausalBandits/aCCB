import numpy as np
import time
import plotter_utility

if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    # ---------------------------------
    # FIRST THE PLOTS FOR SIMPLE REGRET
    # ---------------------------------

    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_exploration_budget.txt',
                                output_filename='outputs/plots/simple_regret_with_exploration_budget.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Budget T',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret with Exploration Budget',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_exploration_budget_deterministic.txt',
                                output_filename='outputs/plots/simple_regret_with_exploration_budget_deterministic.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Budget T',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret with Exploration Budget \nand Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_diff_in_best_reward.txt',
                                output_filename='outputs/plots/simple_regret_with_diff_in_best_reward.png',
                                x_axis_label_in_data="diff_in_best_reward",
                                label_x_axis='Difference in Best Reward R',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret \nwith Difference in Best Reward',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_diff_in_best_reward_deterministic.txt',
                                output_filename='outputs/plots/simple_regret_with_diff_in_best_reward_deterministic.png',
                                x_axis_label_in_data="diff_in_best_reward",
                                label_x_axis='Difference in Best Reward R',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret with Difference in \nBest Reward and Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_exploration_budget_long_horizon.txt',
                                output_filename='outputs/plots/simple_regret_with_exploration_budget_long_horizon.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Time T',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret with \nExploration Budget For Large Budgets',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='log')
    plotter_utility.plot_figure(
        input_file='outputs/simple_regret_with_exploration_budget_long_horizon_deterministic.txt',
        output_filename='outputs/plots/simple_regret_with_exploration_budget_long_horizon_deterministic.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Simple Regret',
        plot_title='Plot for Simple Regret with Exploration Budget \nFor Large Budgets and Deterministic Transitions',
        figsize=(8, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_lambda.txt',
                                output_filename='outputs/plots/simple_regret_with_lambda.png',
                                x_axis_label_in_data="lambda",
                                label_x_axis='Lambda Value λ',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret \nwith Instance Dependent Causal Parameter λ',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')
    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_lambda_deterministic.txt',
                                output_filename='outputs/plots/simple_regret_with_lambda_deterministic.png',
                                x_axis_label_in_data="lambda",
                                label_x_axis='Lambda Value λ',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret with Instance Dependent \nCausal Parameter λ, Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_num_intermediate_contexts.txt',
                                output_filename='outputs/plots/simple_regret_with_num_intermediate_contexts.png',
                                x_axis_label_in_data="num_intermediate_contexts",
                                label_x_axis='Number of Intermediate Contexts',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret \nwith Number of Intermediate Contexts',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')
    plotter_utility.plot_figure(input_file='outputs/simple_regret_with_num_intermediate_contexts_deterministic.txt',
                                output_filename='outputs/plots/simple_regret_with_num_intermediate_contexts_deterministic.png',
                                x_axis_label_in_data="num_intermediate_contexts",
                                label_x_axis='Number of Intermediate Contexts',
                                label_y_axis='Simple Regret',
                                plot_title='Plot for Simple Regret with Number of Intermediate \nContexts and Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    # ---------------------------------------------------------------
    # NEXT THE PLOTS FOR PROBABILITY OF FINDING THE BEST INTERVENTION
    # ---------------------------------------------------------------

    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_exploration_budget.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_exploration_budget.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Time T',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention \nwith Exploration Budget',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_exploration_budget_deterministic.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_exploration_budget_deterministic.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Time T',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention with \nExploration Budget and Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_diff_in_best_reward.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_diff_in_best_reward.png',
                                x_axis_label_in_data="diff_in_best_reward",
                                label_x_axis='Difference in Best Reward R',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention \nwith Difference in Best Reward',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_diff_in_best_reward_deterministic.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_diff_in_best_reward_deterministic.png',
                                x_axis_label_in_data="diff_in_best_reward",
                                label_x_axis='Difference in Best Reward R',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention with Difference \nin Best Reward and Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_exploration_budget_long_horizon.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_exploration_budget_long_horizon.png',
                                x_axis_label_in_data="exploration_budget",
                                label_x_axis='Exploration Time T',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention \nwith Large Exploration Budgets',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='log')
    plotter_utility.plot_figure(
        input_file='outputs/prob_best_intervention_with_exploration_budget_long_horizon_deterministic.txt',
        output_filename='outputs/plots/prob_best_intervention_with_exploration_budget_long_horizon_deterministic.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Probability of not finding the Best Intervention',
        plot_title='Plot for Probability of Best Intervention with Large \nExploration Budgets and Deterministic Transitions',
        figsize=(8, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_lambda.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_lambda.png',
                                x_axis_label_in_data="lambda",
                                label_x_axis='Lambda Value λ',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention \nwith Instance Dependent Causal Parameter λ',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')
    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_lambda_deterministic.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_lambda_deterministic.png',
                                x_axis_label_in_data="lambda",
                                label_x_axis='Lambda Value λ',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention with \nCausal Parameter λ and Deterministic Transitions',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')

    plotter_utility.plot_figure(input_file='outputs/prob_best_intervention_with_num_intermediate_contexts.txt',
                                output_filename='outputs/plots/prob_best_intervention_with_num_intermediate_contexts.png',
                                x_axis_label_in_data="num_intermediate_contexts",
                                label_x_axis='Number of Intermediate Contexts',
                                label_y_axis='Probability of not finding the Best Intervention',
                                plot_title='Plot for Probability of Best Intervention \nwith Number of Intermediate Contexts',
                                figsize=(8, 8),
                                font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
                                markers=None,
                                xscale='linear')
    plotter_utility.plot_figure(
        input_file='outputs/prob_best_intervention_with_num_intermediate_contexts_deterministic.txt',
        output_filename='outputs/plots/prob_best_intervention_with_num_intermediate_contexts_deterministic.png',
        x_axis_label_in_data="num_intermediate_contexts",
        label_x_axis='Number of Intermediate Contexts',
        label_y_axis='Probability of not finding the Best Intervention',
        plot_title='Plot for Probability of Best Intervention with Number of \nIntermediate Contexts and Deterministic Transitions',
        figsize=(8, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    # ------------------------------------------------------------------------
    # NOW THE PLOTS OF DETERMINISTIC AND STOCHASTIC TOGETHER FOR SIMPLE REGRET
    # ------------------------------------------------------------------------

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/simple_regret_with_exploration_budget.txt',
        input_file2='outputs/simple_regret_with_exploration_budget_deterministic.txt',
        output_filename='outputs/plots/simple_regret_with_exploration_budget_together.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Simple Regret',
        plot_title='Plot for Simple Regret \nwith Exploration Budget',
        plot_title2='Plot for Simple Regret with Exploration \nBudget and Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/simple_regret_with_diff_in_best_reward.txt',
        input_file2='outputs/simple_regret_with_diff_in_best_reward_deterministic.txt',
        output_filename='outputs/plots/simple_regret_with_diff_in_best_reward_together.png',
        x_axis_label_in_data="diff_in_best_reward",
        label_x_axis='Difference in Best Reward R',
        label_y_axis='Simple Regret',
        plot_title='Plot for Simple Regret \nwith Difference in Best Reward',
        plot_title2='Plot for Simple Regret with Difference in \nBest Reward and Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/simple_regret_with_exploration_budget_long_horizon.txt',
        input_file2='outputs/simple_regret_with_exploration_budget_long_horizon_deterministic.txt',
        output_filename='outputs/plots/simple_regret_with_exploration_budget_long_horizon_together.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Simple Regret',
        plot_title='Plot for Simple Regret \nwith Large Exploration Budget',
        plot_title2='Plot for Simple Regret with Large \nExploration Budget and Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure_deterministic_stochastic(input_file='outputs/simple_regret_with_lambda.txt',
                                                         input_file2='outputs/simple_regret_with_lambda_deterministic.txt',
                                                         output_filename='outputs/plots/simple_regret_with_lambda_together.png',
                                                         x_axis_label_in_data="lambda",
                                                         label_x_axis='Lambda Value λ',
                                                         label_y_axis='Simple Regret',
                                                         plot_title='Plot for Simple Regret \nwith Instance Dependent Causal Parameter λ',
                                                         plot_title2='Plot for Simple Regret with \nCausal Parameter λ and Deterministic Transitions',
                                                         figsize=(16, 8),
                                                         font_family='serif', general_font_size=11,
                                                         axes_label_font_size=14, title_font_size=16,
                                                         markers=None,
                                                         xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/simple_regret_with_num_intermediate_contexts.txt',
        input_file2='outputs/simple_regret_with_num_intermediate_contexts_deterministic.txt',
        output_filename='outputs/plots/simple_regret_with_num_intermediate_contexts_together.png',
        x_axis_label_in_data="num_intermediate_contexts",
        label_x_axis='Number of Intermediate Contexts',
        label_y_axis='Simple Regret',
        plot_title='Plot for Simple Regret \nwith Number of Intermediate Contexts',
        plot_title2='Plot for Simple Regret with Number of \nIntermediate Contexts and Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')


    # -------------------------------------------------------------------------------------------
    # NOW THE PLOTS OF DETERMINISTIC AND STOCHASTIC TOGETHER FOR PROBABILITY OF BEST INTERVENTION
    # -------------------------------------------------------------------------------------------

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/prob_best_intervention_with_exploration_budget.txt',
        input_file2='outputs/prob_best_intervention_with_exploration_budget_deterministic.txt',
        output_filename='outputs/plots/prob_best_intervention_with_exploration_budget_together.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Probability of not finding the Best Intervention',
        plot_title='Plot for Probability of Best Intervention \nwith Exploration Budget',
        plot_title2='Plot for Probability of Best Intervention \nwith Exploration Budget and Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/prob_best_intervention_with_diff_in_best_reward.txt',
        input_file2='outputs/prob_best_intervention_with_diff_in_best_reward_deterministic.txt',
        output_filename='outputs/plots/prob_best_intervention_with_diff_in_best_reward_together.png',
        x_axis_label_in_data="diff_in_best_reward",
        label_x_axis='Difference in Best Reward R',
        label_y_axis='Probability of not finding the Best Intervention',
        plot_title='Plot for Probability of Best Intervention \nwith Difference in Best Reward',
        plot_title2='Plot for Probability of Best Intervention with \nDifference in Best Reward, Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/prob_best_intervention_with_exploration_budget_long_horizon.txt',
        input_file2='outputs/prob_best_intervention_with_exploration_budget_long_horizon_deterministic.txt',
        output_filename='outputs/plots/prob_best_intervention_with_exploration_budget_long_horizon_together.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis='Probability of not finding the Best Intervention',
        plot_title='Plot for Probability of Best Intervention \nwith Large Exploration Budget',
        plot_title2='Plot for Probability of Best Intervention with Large \nExploration Budget and Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure_deterministic_stochastic(input_file='outputs/prob_best_intervention_with_lambda.txt',
                                                         input_file2='outputs/prob_best_intervention_with_lambda_deterministic.txt',
                                                         output_filename='outputs/plots/prob_best_intervention_with_lambda_together.png',
                                                         x_axis_label_in_data="lambda",
                                                         label_x_axis='Lambda Value λ',
                                                         label_y_axis='Probability of not finding the Best Intervention',
                                                         plot_title='Plot for Probability of Best Intervention \nwith Instance Dependent Causal Parameter λ',
                                                         plot_title2='Plot for Probability of Best Intervention with \nCausal Parameter λ and Deterministic Transitions',
                                                         figsize=(16, 8),
                                                         font_family='serif', general_font_size=11,
                                                         axes_label_font_size=14, title_font_size=16,
                                                         markers=None,
                                                         xscale='linear')

    plotter_utility.plot_figure_deterministic_stochastic(
        input_file='outputs/prob_best_intervention_with_num_intermediate_contexts.txt',
        input_file2='outputs/prob_best_intervention_with_num_intermediate_contexts_deterministic.txt',
        output_filename='outputs/plots/prob_best_intervention_with_num_intermediate_contexts_together.png',
        x_axis_label_in_data="num_intermediate_contexts",
        label_x_axis='Number of Intermediate Contexts',
        label_y_axis='Probability of not finding the Best Intervention',
        plot_title='Plot for Probability of Best Intervention \nwith Number of Intermediate Contexts',
        plot_title2='Plot for Probability of Best Intervention with Number of \nIntermediate Contexts and Deterministic Transitions',
        figsize=(16, 8),
        font_family='serif', general_font_size=11, axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    # ----------------------------------------------------------------------------
    # NOW THE PLOTS OF SIMPLE REGRET AND PROBABILITY OF BEST INTERVENTION TOGETHER
    # ----------------------------------------------------------------------------

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_exploration_budget.txt',
        input_file2='outputs/simple_regret_with_exploration_budget.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_exploration_budget.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention \nwith Exploration Budget',
        plot_title2='Plot for Simple Regret \nwith Exploration Budget',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_exploration_budget_deterministic.txt',
        input_file2='outputs/simple_regret_with_exploration_budget_deterministic.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_exploration_budget_deterministic.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention \nwith Exploration Budget Deterministic Setting',
        plot_title2='Plot for Simple Regret \nwith Exploration Budget Deterministic Setting',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_exploration_budget_long_horizon.txt',
        input_file2='outputs/simple_regret_with_exploration_budget_long_horizon.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_exploration_budget_long_horizon.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention \nwith Large Exploration Budget',
        plot_title2='Plot for Simple Regret \nwith Large Exploration Budget',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_exploration_budget_long_horizon_deterministic.txt',
        input_file2='outputs/simple_regret_with_exploration_budget_long_horizon_deterministic.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_exploration_budget_long_horizon_deterministic.png',
        x_axis_label_in_data="exploration_budget",
        label_x_axis='Exploration Time T',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention with Large \nExploration Budget Deterministic Setting',
        plot_title2='Plot for Simple Regret with Large \nExploration Budget Deterministic Setting',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='log')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_diff_in_best_reward.txt',
        input_file2='outputs/simple_regret_with_diff_in_best_reward.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_diff_in_best_reward.png',
        x_axis_label_in_data="diff_in_best_reward",
        label_x_axis='Difference in Best Reward',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention \nwith Difference in Best Reward',
        plot_title2='Plot for Simple Regret \nwith Difference in Best Reward',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_diff_in_best_reward_deterministic.txt',
        input_file2='outputs/simple_regret_with_diff_in_best_reward_deterministic.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_diff_in_best_reward_deterministic.png',
        x_axis_label_in_data="diff_in_best_reward",
        label_x_axis='Difference in Best Reward',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention with \nDifference in Best Reward Deterministic Setting',
        plot_title2='Plot for Simple Regret with Difference \nin Best Reward Deterministic Setting',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_lambda.txt',
        input_file2='outputs/simple_regret_with_lambda.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_lambda.png',
        x_axis_label_in_data="lambda",
        label_x_axis='Lambda Value λ',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention \nwith Lambda Value λ',
        plot_title2='Plot for Simple Regret \nwith Lambda Value λ',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_lambda_deterministic.txt',
        input_file2='outputs/simple_regret_with_lambda_deterministic.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_lambda_deterministic.png',
        x_axis_label_in_data="lambda",
        label_x_axis='Lambda Value λ',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention \nwith Lambda Value λ for Deterministic Setting',
        plot_title2='Plot for Simple Regret \nwith Lambda Value λ for Deterministic Setting',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_num_intermediate_contexts.txt',
        input_file2='outputs/simple_regret_with_num_intermediate_contexts.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_with_num_intermediate_contexts.png',
        x_axis_label_in_data="num_intermediate_contexts",
        label_x_axis='Number of Intermediate Contexts',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention \nwith Number of Intermediate Contexts',
        plot_title2='Plot for Simple Regret \nwith Number of Intermediate Contexts',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    plotter_utility.plot_figure_prob_best_simple_regret(
        input_file1='outputs/prob_best_intervention_with_num_intermediate_contexts_deterministic.txt',
        input_file2='outputs/simple_regret_with_num_intermediate_contexts_deterministic.txt',
        output_filename='outputs/plots/simple_regret_and_prob_best_with_with_num_intermediate_contexts_deterministic.png',
        x_axis_label_in_data="num_intermediate_contexts",
        label_x_axis='Number of Intermediate Contexts',
        label_y_axis1='Probability of not finding the Best Intervention',
        label_y_axis2='Simple Regret',
        plot_title1='Plot for Probability of Best Intervention with Number \nof Intermediate Contexts For Deterministic Setting',
        plot_title2='Plot for Simple Regret with Number \nof Intermediate Contexts for Deterministic Setting',
        figsize=(16, 8),
        font_family='serif', general_font_size=11,
        axes_label_font_size=14, title_font_size=16,
        markers=None,
        xscale='linear')

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
