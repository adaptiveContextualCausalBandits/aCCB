import numpy as np
import importlib
from gekko import GEKKO
import cvxpy as cp
import setup_contextualcausalbandit as setup


def get_prob_optimal_reward(sampled_transition_probabilities, sampled_average_reward_matrix):
    """

    :param num_exploration_per_intervention_at_intermediate_context:
    :param sampled_average_reward_matrix:
    :param reward_difference: difference between best expected reward and obtained reward (for example: 0.3)
    :return: computed_regret: regret computed given the exploration
    """
    # Check if the best intervention in the first context is 0
    if np.argmax(sampled_average_reward_matrix[0, :]) != 0:
        return 1
    max_expected_reward_per_state = np.max(sampled_average_reward_matrix, axis=1)
    # print("max_expected_reward_per_state=", max_expected_reward_per_state)
    expected_reward_per_intervention_at_state_0 = sampled_transition_probabilities @ max_expected_reward_per_state
    # print("expected_reward_per_intervention_at_state_0=", expected_reward_per_intervention_at_state_0)

    # We want worst case, so we will take the last argmax index rather than the first.
    # Reverse the vector
    rev_expected_reward_per_intervention_at_state_0 = expected_reward_per_intervention_at_state_0[::-1]

    # Find the index of the maximum value in the reversed vector
    last_argmax_index = len(expected_reward_per_intervention_at_state_0) - 1 - np.argmax(
        rev_expected_reward_per_intervention_at_state_0)

    if last_argmax_index != 0:
        return 1
    return 0


def get_prob_optimal_reward_simple_setting(sampled_average_reward_vector):
    """
    We check if the best intervention across the interventions is 0 [corresponding to pair (0,0)].
    Then the regret is 0, else it is 1

    :param num_exploration_per_intervention_at_intermediate_context:
    :param sampled_average_reward_matrix:
    :param reward_difference: difference between best expected reward and obtained reward (for example: 0.3)
    :return: computed_regret: regret computed given the exploration

    """
    # We want the last rather than the first instance of such a vector
    # Reverse the vector
    rev_sampled_average_reward_vector = sampled_average_reward_vector[::-1]

    # Find the index of the maximum value in the reversed vector
    last_argmax_index = len(sampled_average_reward_vector) - 1 - np.argmax(rev_sampled_average_reward_vector)

    if last_argmax_index != 0:
        return 1
    return 0


def get_simple_regret(sampled_transition_probabilities, sampled_average_reward_matrix, transition_matrix,
                      reward_matrix):
    """

    :param sampled_transition_probabilities: sampled transition probabilities of size num_intervetions x num_intermediate contexts
    :param sampled_average_reward_matrix: sampled reward matrix of size num_intermediate_contexts x num_interventions_at_intermediate_contexts
    :param transition_matrix: actual transition probabilities of size num_intervetions x num_intermediate contexts
    :param reward_matrix: actual reward matrix of size num_intermediate_contexts x num_interventions_at_intermediate_contexts
    :return:
    """
    full_sampled_reward_matrix = sampled_transition_probabilities @ sampled_average_reward_matrix
    full_sampled_reward_vector = full_sampled_reward_matrix.reshape(-1)

    # argmax_full_sampled_vector = np.argmax(full_sampled_reward_vector)
    # We want the last rather than the first instance of such a vector
    # Reverse the vector
    rev_full_sampled_reward_vector = full_sampled_reward_vector[::-1]

    # Find the index of the maximum value in the reversed vector
    last_argmax_full_sampled_vector = len(rev_full_sampled_reward_vector) - 1 - np.argmax(
        rev_full_sampled_reward_vector)

    if last_argmax_full_sampled_vector == 0:
        return 0
    full_reward_matrix = transition_matrix @ reward_matrix
    full_reward_vector = full_reward_matrix.reshape(-1)
    reward_of_alg_best_intervention = full_reward_vector[last_argmax_full_sampled_vector]
    reward_of_best_intervention = full_reward_vector[0]
    regret = reward_of_best_intervention - reward_of_alg_best_intervention
    return regret


def get_simple_regret_simple_setting(sampled_average_reward_vector, transition_matrix, reward_matrix):
    # We want the last rather than the first instance of such a vector
    # Reverse the vector
    rev_full_sampled_avg_reward_vector = sampled_average_reward_vector[::-1]

    # Find the index of the maximum value in the reversed vector
    last_argmax_full_sampled_vector = len(rev_full_sampled_avg_reward_vector) - 1 - np.argmax(
        rev_full_sampled_avg_reward_vector)

    if last_argmax_full_sampled_vector == 0:
        return 0
    full_reward_matrix = transition_matrix @ reward_matrix
    full_reward_vector = full_reward_matrix.reshape(-1)
    reward_of_alg_best_intervention = full_reward_vector[last_argmax_full_sampled_vector]
    reward_of_best_intervention = full_reward_vector[0]
    regret = reward_of_best_intervention - reward_of_alg_best_intervention
    return regret


def run_multiple_sims(num_sims, exploration_budget, transition_matrix, reward_matrix,
                      simulation_module="roundrobin_roundrobin", simple=False, regret_metric_name="simple_regret"):
    """
    
    :param num_sims: 
    :param exploration_budget: 
    :param diff_in_best_reward: 
    :param transition_matrix:
    :param reward_matrix: 
    :param simulation_module: 
    :param simple: 
    :param regret_metric_name: Can either be "simple_regret" or "prob_best_intervention"
    :return: regret
    """
    total_regret_metric = 0
    mymodule = importlib.import_module(simulation_module)
    for i in range(num_sims):
        if not simple:
            sampled_transition_probabilities, sampled_average_reward_matrix = \
                mymodule.run_one_sim(exploration_budget, transition_matrix, reward_matrix)
            if regret_metric_name == "simple_regret":
                regret_metric = get_simple_regret(sampled_transition_probabilities, sampled_average_reward_matrix,
                                                  transition_matrix, reward_matrix)
            else:
                regret_metric = get_prob_optimal_reward(sampled_transition_probabilities, sampled_average_reward_matrix)
        else:
            sampled_average_reward_vector = mymodule.run_one_sim(exploration_budget, transition_matrix,
                                                                 reward_matrix)
            if regret_metric_name == "simple_regret":
                regret_metric = get_simple_regret_simple_setting(sampled_average_reward_vector,
                                                                 transition_matrix, reward_matrix)
            else:
                regret_metric = get_prob_optimal_reward_simple_setting(sampled_average_reward_vector)
        total_regret_metric += regret_metric
    average_regret_metric = total_regret_metric / num_sims
    return average_regret_metric


def solveFConvexProgram(transition_matrix_estimate, causal_parameters_estimate):
    numInterventions, numStates = transition_matrix_estimate.shape
    f = cp.Variable(numInterventions)

    ones = np.ones(numInterventions)
    constraints = [0 <= f, f <= 1, f @ ones == 1]
    MHalf = np.power(causal_parameters_estimate, 0.5)
    A = transition_matrix_estimate @ MHalf
    objective = cp.Minimize(cp.max(A @ (cp.power(transition_matrix_estimate.T @ f, -0.5))))
    prob = cp.Problem(objective, constraints)
    f.value = 0.5 * np.ones(numInterventions)
    prob.solve(solver=cp.SCS, warm_start=True)
    fStar = f.value
    maximin = np.amax(transition_matrix_estimate @ MHalf @ (np.power((transition_matrix_estimate.T @ fStar), -0.5)))
    return maximin, fStar


def getFMaxMin(PHat):
    numInterventions, numStates = PHat.shape
    f = cp.Variable(numInterventions)
    ones = np.ones(numInterventions)
    constraints = [0 <= f, f <= 1, f @ ones == 1]
    objective = cp.Maximize(cp.min(PHat.T @ f))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    ftilde = f.value
    maximin = np.amin(PHat.T @ ftilde)
    return maximin, ftilde


def run_multiple_sims_multiple_models(models, num_sims, exploration_budget, num_intermediate_contexts,
                                      num_interventions, diff_prob_transition, default_reward,
                                      diff_in_best_reward, stochastic=True, regret_metric_name="simple_regret",
                                      simple_modules=None, mParam=2):
    """

    :param models:

    :param num_sims:
    :param exploration_budget:
    :param num_intermediate_contexts:
    :param num_interventions:
    :param diff_prob_transition:
    :param default_reward:
    :param diff_in_best_reward:
    :param stochastic:
    :param regret_metric_name: Can either be "simple_regret" or "prob_best_intervention"
    :return:
    """
    if simple_modules is None:
        simple_modules = ["ucb_over_intervention_pairs", "ts_over_intervention_pairs"]

    total_regret_metric = np.zeros((len(models)), dtype=np.float32)
    for i in range(num_sims):
        # Generate the required matrices from the above hyperparameters
        if stochastic:
            transition_matrix = setup.generate_stochastic_transition_matrix(num_intermediate_contexts,
                                                                            num_interventions,
                                                                            diff_prob_transition)
        else:
            transition_matrix = setup.generate_deterministic_transition_matrix(num_intermediate_contexts,
                                                                               num_interventions)

        reward_matrix = setup.generate_reward_matrix(num_intermediate_contexts, num_interventions,
                                                     default_reward, diff_in_best_reward)
        for model_num in range(len(models)):
            model = models[model_num]
            simple_flag = True if model in simple_modules else False
            mymodule = importlib.import_module(model)
            if not simple_flag:
                if model == "convex_explorer":
                    sampled_transition_probabilities, sampled_average_reward_matrix = \
                        mymodule.run_one_sim(exploration_budget, transition_matrix, reward_matrix,
                                             m_parameter_at_intermediate_states=mParam)
                else:
                    sampled_transition_probabilities, sampled_average_reward_matrix = \
                        mymodule.run_one_sim(exploration_budget, transition_matrix, reward_matrix)
                if regret_metric_name == "simple_regret":
                    regret_metric = get_simple_regret(sampled_transition_probabilities,
                                                      sampled_average_reward_matrix,
                                                      transition_matrix=transition_matrix,
                                                      reward_matrix=reward_matrix)
                else:
                    regret_metric = get_prob_optimal_reward(sampled_transition_probabilities,
                                                            sampled_average_reward_matrix)
            else:
                sampled_average_reward_vector = mymodule.run_one_sim(exploration_budget, transition_matrix,
                                                                     reward_matrix)
                if regret_metric_name == "simple_regret":
                    regret_metric = get_simple_regret_simple_setting(sampled_average_reward_vector, transition_matrix,
                                                                     reward_matrix)
                else:
                    regret_metric = get_prob_optimal_reward_simple_setting(sampled_average_reward_vector)
            total_regret_metric[model_num] += regret_metric
    average_regret_metric = total_regret_metric / num_sims
    return average_regret_metric
