import numpy as np
import time
import setup_contextualcausalbandit as setup, utilities


def run_one_sim(exploration_budget, transition_matrix, reward_matrix):
    """
    Here we will run one simulation of the algorithm which uses round robin at the start state, and thompson sampling
    at the intermediate contexts.

    :param exploration_budget:
    :param transition_matrix:
    :param reward_matrix:
    :return:
    :sampled_transition_probabilities: transition probability matrix estimate
    :sampled_average_reward_matrix: reward matrix estimate
    """
    num_interventions_at_state_0, num_intermediate_contexts = transition_matrix.shape
    _, num_interventions = reward_matrix.shape

    # Distribute the budget amongst interventions at state 0
    budget_per_intervention_at_state_0 = exploration_budget // \
                                         num_interventions_at_state_0 * np.ones(num_interventions_at_state_0)
    remaining_budget_per_intervention_at_state_0 = exploration_budget % num_interventions_at_state_0
    budget_per_intervention_at_state_0[:remaining_budget_per_intervention_at_state_0] += 1

    sampled_transition_matrix = np.zeros((num_interventions_at_state_0, num_intermediate_contexts))
    for intervention in range(num_interventions_at_state_0):
        sampled_transition_matrix[intervention] += \
            np.random.multinomial(budget_per_intervention_at_state_0[intervention],
                                  transition_matrix[intervention], None)
    sampled_transition_probabilities = sampled_transition_matrix / budget_per_intervention_at_state_0[:, np.newaxis]
    num_exploration_per_intermediate_context = np.sum(sampled_transition_matrix, axis=0)

    num_pulls_per_intervention_in_context = np.zeros(reward_matrix.shape, dtype=np.int32)
    sampled_total_reward_matrix = np.zeros(reward_matrix.shape, dtype=np.int32)
    # Initialize the priors to all ones
    prior_success_per_intervention_in_context = np.ones(reward_matrix.shape, dtype=np.int32)
    prior_failures_per_intervention_in_context = np.ones(reward_matrix.shape, dtype=np.int32)
    # Compute the initial beta values
    sampled_beta_probs = np.random.beta(prior_success_per_intervention_in_context,
                                        prior_failures_per_intervention_in_context)

    for context_index in range(num_intermediate_contexts):
        context_reward = reward_matrix[context_index]
        context_budget = int(num_exploration_per_intermediate_context[context_index])
        # Run Thompson Sampling for this context for T time steps
        for round in range(context_budget):
            # Select the intervention with the highest sampled beta value
            intervention_to_pull = np.argmax(sampled_beta_probs[context_index,:])
            expected_reward_of_ts_arm = context_reward[intervention_to_pull]

            # Simulate pulling the selected arm and observing the reward (0 or 1)
            reward = np.random.binomial(1, expected_reward_of_ts_arm)

            # Update the Beta distribution parameters based on the observed reward
            if reward == 1:
                prior_success_per_intervention_in_context[context_index, intervention_to_pull] += 1
            else:
                prior_failures_per_intervention_in_context[context_index, intervention_to_pull] += 1

            # Update statistics
            num_pulls_per_intervention_in_context[context_index, intervention_to_pull] += 1
            sampled_total_reward_matrix[context_index, intervention_to_pull] += reward

            # Update beta probabilities only for the pulled intervention
            sampled_beta_probs[context_index,intervention_to_pull] = np.random.beta(
                prior_success_per_intervention_in_context[context_index, intervention_to_pull],
                prior_failures_per_intervention_in_context[context_index, intervention_to_pull])


    # Instead of simply dividing, we want to divide where the denominator is non-zero
    # sampled_average_reward_matrix = sampled_total_reward_matrix / num_pulls_per_intervention_in_context
    sampled_average_reward_matrix = np.divide(sampled_total_reward_matrix,
                                              num_pulls_per_intervention_in_context,
                                              out=np.zeros_like(sampled_total_reward_matrix, dtype=np.float32),
                                              where=(num_pulls_per_intervention_in_context != 0))

    # print("sampled_total_reward_matrix=", sampled_total_reward_matrix)
    # print("num_pulls_per_intervention_in_context=", num_pulls_per_intervention_in_context)
    # print("sampled_average_reward_matrix=", sampled_average_reward_matrix)
    return sampled_transition_probabilities, sampled_average_reward_matrix


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

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

    exploration_budget = 1000000
    # deterministic transitions
    sampled_transition_probabilities, sampled_average_reward_matrix = \
        run_one_sim(exploration_budget, det_transition_matrix, reward_matrix)

    print("sampled_transition_probabilities, =", sampled_transition_probabilities)
    print("sampled_average_reward_matrix, =", sampled_average_reward_matrix)
    # stochastic transitions
    sampled_transition_probabilities, sampled_average_reward_matrix = \
        run_one_sim(exploration_budget, stochastic_transition_matrix, reward_matrix)
    print("sampled_transition_probabilities =", sampled_transition_probabilities)
    print("sampled_average_reward_matrix =", sampled_average_reward_matrix)

    regret = utilities.get_regret(sampled_transition_probabilities, sampled_average_reward_matrix, diff_in_best_reward)
    print("regret = ", regret)
    num_sims = 100
    average_regret = utilities.run_multiple_sims(num_sims, exploration_budget, diff_in_best_reward,
                                                 stochastic_transition_matrix, reward_matrix,
                                                 simulation_module="roundrobin_ts")
    print("average_regret=", average_regret)
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
