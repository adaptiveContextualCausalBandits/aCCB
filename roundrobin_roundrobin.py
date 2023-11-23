import numpy as np
import time
import setup_contextualcausalbandit as setup, utilities


def run_one_sim(exploration_budget, transition_matrix, reward_matrix):
    """
    Here we will run one simulation of the algorithm which uses round robin at the start state, and round robin
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
    # print("budget_per_intervention_at_state_0", budget_per_intervention_at_state_0)
    # print("sampled_transition_matrix", sampled_transition_matrix)
    # print("sampled_transition_probabilities", sampled_transition_probabilities)
    num_exploration_per_intermediate_context = np.sum(sampled_transition_matrix, axis=0)
    # print("num_exploration_per_intermediate_context",
    #       num_exploration_per_intermediate_context)
    sampled_total_reward_matrix = np.zeros(reward_matrix.shape)
    budget_per_intervention_in_context = np.zeros(reward_matrix.shape, dtype=np.int32)
    for context_index in range(num_intermediate_contexts):
        budget_in_context = int(num_exploration_per_intermediate_context[context_index])
        # distribute the budget equally amongst the interventions
        budget_per_intervention_in_context[context_index, :] = budget_in_context // num_interventions * np.ones(
            num_interventions, dtype=np.int32)
        remaining_budget = budget_in_context % num_interventions
        budget_per_intervention_in_context[context_index, :remaining_budget] += 1

        sampled_total_reward_matrix[context_index] = np.random.binomial(
            budget_per_intervention_in_context[context_index, :], reward_matrix[context_index], None)


    # Instead of simply dividing, we want to divide where the denominator is non-zero
    # sampled_average_reward_matrix = sampled_total_reward_matrix / budget_per_intervention_in_context
    sampled_average_reward_matrix = np.divide(sampled_total_reward_matrix,
                                              budget_per_intervention_in_context,
                                              out=np.zeros_like(sampled_total_reward_matrix, dtype=np.float32),
                                              where=(budget_per_intervention_in_context != 0))

    # print("sampled_total_reward_matrix=", sampled_total_reward_matrix)
    # print("budget_per_intervention_in_context=", budget_per_intervention_in_context)
    # print("sampled_average_reward_matrix=", sampled_average_reward_matrix)
    return sampled_transition_probabilities, sampled_average_reward_matrix


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    # Set up the variables required to run the simulation
    num_intermediate_contexts = 25
    num_causal_variables = 25
    num_interventions = num_causal_variables * 2 + 1
    diff_prob_transition = 0.1
    default_reward = 0.5
    diff_in_best_reward = 0.3

    det_transition_matrix = setup.generate_deterministic_transition_matrix(num_intermediate_contexts, num_interventions)
    stochastic_transition_matrix = setup.generate_stochastic_transition_matrix(num_intermediate_contexts,
                                                                               num_interventions, diff_prob_transition)
    reward_matrix = setup.generate_reward_matrix(num_intermediate_contexts, num_interventions,
                                                 default_reward, diff_in_best_reward)

    exploration_budget = 50000
    # deterministic transitions
    sampled_transition_probabilities, sampled_average_reward_matrix = \
        run_one_sim(exploration_budget, det_transition_matrix, reward_matrix)

    print("sampled_transition_probabilities, =", sampled_transition_probabilities)
    print("sampled_average_reward_matrix, =", sampled_average_reward_matrix)
    # stochastic transitions
    sampled_transition_probabilities, sampled_average_reward_matrix = \
        run_one_sim(exploration_budget, stochastic_transition_matrix, reward_matrix)
    print("sampled_transition_probabilities, =", sampled_transition_probabilities)
    print("sampled_average_reward_matrix, =", sampled_average_reward_matrix)

    regret = utilities.get_regret(sampled_transition_probabilities, sampled_average_reward_matrix, diff_in_best_reward)
    print("regret = ", regret)
    num_sims = 1000
    average_regret = utilities.run_multiple_sims(num_sims, exploration_budget, diff_in_best_reward,
                                                 stochastic_transition_matrix, reward_matrix,
                                                 simulation_module="roundrobin_roundrobin")
    print("average_regret=", average_regret)
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
