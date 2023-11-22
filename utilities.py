import numpy as np
import importlib


def get_regret(sampled_transition_probabilities, sampled_average_reward_matrix, diff_in_best_reward=0.3):
    """

    :param num_exploration_per_intervention_at_intermediate_context:
    :param sampled_average_reward_matrix:
    :param reward_difference: difference between best expected reward and obtained reward (for example: 0.3)
    :return: computed_regret: regret computed given the exploration
    """
    # Check if the best intervention in the first context is 0
    if np.argmax(sampled_average_reward_matrix[0, :]) != 0:
        return diff_in_best_reward
    max_expected_reward_per_state = np.max(sampled_average_reward_matrix, axis=1)
    # print("max_expected_reward_per_state=", max_expected_reward_per_state)
    expected_reward_per_intervention_at_state_0 = sampled_transition_probabilities @ max_expected_reward_per_state
    # print("expected_reward_per_intervention_at_state_0=", expected_reward_per_intervention_at_state_0)
    if np.argmax(expected_reward_per_intervention_at_state_0) != 0:
        return diff_in_best_reward
    return 0


def run_multiple_sims(num_sims, exploration_budget, diff_in_best_reward, stochastic_transition_matrix, reward_matrix,
                  simulation_module="roundrobin_roundrobin"):
    total_regret = 0
    mymodule = importlib.import_module(simulation_module)
    for i in range(num_sims):
        sampled_transition_probabilities, sampled_average_reward_matrix = \
            mymodule.run_one_sim(exploration_budget, stochastic_transition_matrix, reward_matrix)
        regret = get_regret(sampled_transition_probabilities, sampled_average_reward_matrix, diff_in_best_reward)
        total_regret += regret
    average_regret = total_regret / num_sims
    return average_regret
