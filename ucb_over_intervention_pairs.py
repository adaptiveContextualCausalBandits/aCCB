import numpy as np
import time
import setup_contextualcausalbandit as setup, utilities


def run_one_sim(exploration_budget, transition_matrix, reward_matrix, exploration_alpha=2):
    """
    Here we will run one simulation of the algorithm which uses round robin at the start state, and ucb at the
    intermediate contexts.

    :param exploration_budget:
    :param transition_matrix:
    :param reward_matrix:
    :return:
    :sampled_transition_probabilities: transition probability matrix estimate
    :sampled_average_reward_matrix: reward matrix estimate
    """

    num_interventions_at_state_0, num_intermediate_contexts = transition_matrix.shape
    _, num_interventions = reward_matrix.shape
    total_ucb_interventions = num_interventions_at_state_0 * num_interventions

    num_pulls_per_intervention_pair = np.zeros(total_ucb_interventions, dtype=np.int32)
    sampled_total_reward_matrix = np.zeros(total_ucb_interventions, dtype=np.int32)
    # Initialize the ucb to some high value, so that all interventions are pulled first.
    ucb_per_intervention_pair = 5 * np.ones(total_ucb_interventions, dtype=np.int32)

    # We now compute the effective reward matrix for each ucb intervention pair
    effective_rewards_for_intervention_pair = (transition_matrix @ reward_matrix).flatten()
    # print("effective_rewards_for_intervention_pair=",effective_rewards_for_intervention_pair)

    # Main UCB loop
    for t in range(exploration_budget):
        # Select the arm with the maximum UCB value
        intervention_to_pull = np.argmax(ucb_per_intervention_pair)
        expected_reward_of_ucb_intervention = effective_rewards_for_intervention_pair[intervention_to_pull]
        # Simulate pulling the selected arm and observing the reward (0 or 1)
        reward = np.random.binomial(1, expected_reward_of_ucb_intervention)

        # Update statistics
        num_pulls_per_intervention_pair[intervention_to_pull] += 1
        sampled_total_reward_matrix[intervention_to_pull] += reward

        # Update UCB value only for the pulled intervention
        average_reward = sampled_total_reward_matrix[intervention_to_pull] / (
                num_pulls_per_intervention_pair[intervention_to_pull] + 1e-6)
        exploration_term = exploration_alpha * np.sqrt(
            np.log(exploration_budget + 1) / (num_pulls_per_intervention_pair[intervention_to_pull] + 1e-6))
        ucb_per_intervention_pair[intervention_to_pull] = average_reward + exploration_term


    # Instead of simply dividing, we want to divide where the denominator is non-zero
    # sampled_average_reward_matrix = sampled_total_reward_matrix / num_pulls_per_intervention_in_context
    sampled_average_reward_vector = np.divide(sampled_total_reward_matrix,
                                              num_pulls_per_intervention_pair,
                                              out=np.zeros_like(sampled_total_reward_matrix, dtype=np.float32),
                                              where=(num_pulls_per_intervention_pair != 0))

    # print("sampled_total_reward_matrix=", sampled_total_reward_matrix)
    # print("num_pulls_per_intervention_pair=", num_pulls_per_intervention_pair)
    # print("sum(num_pulls_per_intervention_pair)=", sum(num_pulls_per_intervention_pair))
    # print("num_pulls_per_intervention_pair.shape=", num_pulls_per_intervention_pair.shape)
    # print("sampled_average_reward_vector=", sampled_average_reward_vector)
    return sampled_average_reward_vector


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    np.set_printoptions(threshold=np.inf)

    # Set up the variables required to run the simulation
    num_intermediate_contexts = 5
    num_causal_variables = 5
    num_interventions = num_causal_variables * 2 + 1
    diff_prob_transition = 0.3
    default_reward = 0.5
    diff_in_best_reward = 0.3

    det_transition_matrix = setup.generate_deterministic_transition_matrix(num_intermediate_contexts, num_interventions)
    stochastic_transition_matrix = setup.generate_stochastic_transition_matrix(num_intermediate_contexts,
                                                                               num_interventions, diff_prob_transition)
    reward_matrix = setup.generate_reward_matrix(num_intermediate_contexts, num_interventions,
                                                 default_reward, diff_in_best_reward)

    exploration_budget = 5000
    # deterministic transitions
    sampled_average_reward_vector = run_one_sim(exploration_budget, det_transition_matrix, reward_matrix)

    print("sampled_average_reward_vector, =", sampled_average_reward_vector)
    prob_of_optimal_reward = utilities.get_prob_optimal_reward_simple_setting(sampled_average_reward_vector)
    print("prob_of_optimal_reward = ", prob_of_optimal_reward)
    # stochastic transitions
    sampled_average_reward_vector = run_one_sim(exploration_budget, stochastic_transition_matrix, reward_matrix)
    print("sampled_average_reward_vector, =", sampled_average_reward_vector)

    prob_of_optimal_reward = utilities.get_prob_optimal_reward_simple_setting(sampled_average_reward_vector)
    print("prob_of_optimal_reward = ", prob_of_optimal_reward)

    simple_regret = utilities.get_simple_regret_simple_setting(sampled_average_reward_vector,
                                                               stochastic_transition_matrix, reward_matrix)
    print("simple_regret = ", simple_regret)

    num_sims = 100
    average_prob_optimal_reward = utilities.run_multiple_sims(num_sims, exploration_budget,
                                                              stochastic_transition_matrix, reward_matrix,
                                                              simulation_module="ucb_over_intervention_pairs",
                                                              simple=True,
                                                              regret_metric_name="prob_best_intervention")
    print("average_prob_optimal_reward=", average_prob_optimal_reward)

    average_simple_regret = utilities.run_multiple_sims(num_sims, exploration_budget,
                                                        stochastic_transition_matrix, reward_matrix,
                                                        simulation_module="ucb_over_intervention_pairs",
                                                        simple=True,
                                                        regret_metric_name="simple_regret")
    print("average_simple_regret=", average_simple_regret)
    np.set_printoptions(threshold=False)
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
