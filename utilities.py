import numpy as np
import importlib
from gekko import GEKKO
import cvxpy as cp
import setup_contextualcausalbandit as setup
from tqdm import tqdm


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
                                      simple_models=None, m_param=2):
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
    # print("models=",models)
    # print("num_intermediate_contexts=",num_intermediate_contexts)
    # print("num_interventions=", num_interventions)
    # print("exploration_budget=", exploration_budget)
    # print("diff_prob_transition=", diff_prob_transition)
    # print("default_reward=", default_reward)
    # print("diff_in_best_reward=", diff_in_best_reward)
    # print("num_sims=", num_sims)
    # print("m_param=", m_param)
    # print("stochastic=", stochastic)
    # print("regret_metric_name=", regret_metric_name)

    if simple_models is None:
        simple_models = ["ucb_over_intervention_pairs", "ts_over_intervention_pairs"]

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
            simple_flag = True if model in simple_models else False
            mymodule = importlib.import_module(model)
            if not simple_flag:
                if model == "convex_explorer":
                    sampled_transition_probabilities, sampled_average_reward_matrix = \
                        mymodule.run_one_sim(exploration_budget, transition_matrix, reward_matrix,
                                             m_parameter_at_intermediate_states=m_param)
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


def get_lambda_values(m_parameters_range, num_intermediate_contexts, num_causal_variables, diff_prob_transition,
                      stochastic_flag):
    """

    :param m_parameters_range:
    :param num_intermediate_contexts:
    :param num_causal_variables:
    :param diff_prob_transition:
    :param stochastic_flag:
    :return: lambda values
    """
    # Initialize the output vector
    lambda_values = np.zeros(len(m_parameters_range), dtype=np.float32)

    # Generate the required matrices from the above hyperparameters
    num_interventions = num_causal_variables * 2 + 1
    if stochastic_flag:
        transition_matrix_initialization = setup.generate_stochastic_transition_matrix(
            num_intermediate_contexts, num_interventions, diff_prob_transition)
    else:
        transition_matrix_initialization = setup.generate_deterministic_transition_matrix(
            num_intermediate_contexts, num_interventions)

    for index in range(len(m_parameters_range)):
        causal_parameters_vector = m_parameters_range[index] * np.ones(num_intermediate_contexts,
                                                                       dtype=np.int8)
        causal_parameters_diag_matrix = np.diag(causal_parameters_vector)
        lambdaValue, _ = solveFConvexProgram(transition_matrix_initialization,
                                             causal_parameters_diag_matrix)
        lambda_values[index] = lambdaValue ** 2

    return lambda_values


def run_all_with_feature(models=None, simple_models=None,
                         num_intermediate_contexts=5, num_causal_variables=5,
                         exploration_budget=25_000, diff_prob_transition=0.1, default_reward=0.5,
                         diff_in_best_reward=0.3, num_sims=100, m_param=2,
                         varying_feature_name="diff_in_best_reward",
                         varying_feature_values=None,
                         regret_metric_names=None, stochastic_flags=None,
                         x_axis_values_for_print=None,
                         experiment_variation=''
                         ):
    if models is None:
        models = ['roundrobin_roundrobin', 'roundrobin_ucb', 'roundrobin_ts',
                  'ucb_over_intervention_pairs', 'ts_over_intervention_pairs', 'convex_explorer']
    if simple_models is None:
        simple_models = ["ucb_over_intervention_pairs", "ts_over_intervention_pairs"]
    if regret_metric_names is None:
        regret_metric_names = ["simple_regret", "prob_best_intervention"]
    if stochastic_flags is None:
        stochastic_flags = [True, False]
    if varying_feature_values is None:
        varying_feature_values = [(x + 1) * 0.05 for x in range(9)]
    if x_axis_values_for_print is None:
        x_axis_values_for_print = varying_feature_values
    for regret_metric_name in ["simple_regret", "prob_best_intervention"]:
        for stochastic_flag in [True, False]:
            # The outputs are stored in the below matrix
            average_regret_metric_matrix = np.zeros((len(varying_feature_values), len(models)), dtype=np.float32)
            for index in tqdm(range(len(varying_feature_values)), desc="Progress"):
                globals()['diff_in_best_reward'] = diff_in_best_reward
                globals()['exploration_budget'] = exploration_budget
                globals()['m_param'] = m_param
                globals()['num_intermediate_contexts'] = num_intermediate_contexts

                globals()[varying_feature_name] = varying_feature_values[index]

                if varying_feature_name == "num_intermediate_contexts":
                    num_causal_variables = globals()['num_intermediate_contexts']

                num_interventions = num_causal_variables * 2 + 1

                avg_regret_metric_for_models = run_multiple_sims_multiple_models(models, num_sims,
                                                                                 globals()['exploration_budget'],
                                                                                 globals()['num_intermediate_contexts'],
                                                                                 num_interventions,
                                                                                 diff_prob_transition,
                                                                                 default_reward,
                                                                                 globals()['diff_in_best_reward'],
                                                                                 stochastic=stochastic_flag,
                                                                                 regret_metric_name=regret_metric_name,
                                                                                 m_param=globals()['m_param'])

                avg_regret_metric_for_models = np.minimum(avg_regret_metric_for_models, 1.0)
                average_regret_metric_matrix[index] = avg_regret_metric_for_models

                # Set print options to display the entire array
                np.set_printoptions(threshold=np.inf)
                # Print the progress as a log
                print("\naverage regret so far = ", average_regret_metric_matrix)
                # Reset the threshold for printing

                np.set_printoptions(threshold=False)

            # Now we saved the obtained values to file.
            if stochastic_flag:
                file_path = "outputs/" + regret_metric_name + "_with_" + varying_feature_name + experiment_variation + ".txt"
            else:
                file_path = "outputs/" + regret_metric_name + "_with_" + varying_feature_name + experiment_variation + "_deterministic.txt"

            # Headers for each column
            headers = [varying_feature_name] + models

            # Let the x-axis values for the lambda experiment be equal to lambda rather than the m_params.
            if varying_feature_name == "m_param":
                x_axis_values_for_print = get_lambda_values(x_axis_values_for_print, num_intermediate_contexts,
                                                            num_causal_variables, diff_prob_transition, stochastic_flag)

            # Prepend the row headings
            average_regret_metric_matrix_for_print = np.hstack((np.array(x_axis_values_for_print).reshape(-1, 1),
                                                                average_regret_metric_matrix))

            # Open the file for writing
            with open(file_path, 'w') as file:
                # Write the headers as the first line
                header_line = '\t'.join(headers)  # Use a tab separator
                file.write(header_line + '\n')

                # noinspection PyTypeChecker
                # Save the matrix to the file
                np.savetxt(file, average_regret_metric_matrix_for_print, delimiter='\t', fmt='%0.6f')

    # Success flag
    return 1
