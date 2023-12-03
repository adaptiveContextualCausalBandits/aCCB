import numpy as np
import time
import setup_contextualcausalbandit as setup
import utilities
import importlib
from tqdm import tqdm


def run_multiple_sims_multiple_models(models, num_sims, exploration_budget, num_intermediate_contexts,
                                      num_interventions, diff_prob_transition, default_reward,
                                      diff_in_best_reward, stochastic=True, m_param=2,
                                      regret_metric_name="simple_regret"):
    total_regret = np.zeros((len(models)), dtype=np.float32)

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
                    regret = utilities.get_prob_optimal_reward(sampled_transition_probabilities,
                                                               sampled_average_reward_matrix)
                else:
                    sampled_transition_probabilities, sampled_average_reward_matrix = \
                        mymodule.run_one_sim(exploration_budget, transition_matrix, reward_matrix)
                    regret = utilities.get_prob_optimal_reward(sampled_transition_probabilities,
                                                               sampled_average_reward_matrix)
            else:
                sampled_average_reward_vector = mymodule.run_one_sim(exploration_budget, transition_matrix,
                                                                     reward_matrix)
                regret = utilities.get_prob_optimal_reward_simple_setting(sampled_average_reward_vector)
            total_regret[model_num] += regret
    average_regret = total_regret / num_sims
    return average_regret


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    models = ['roundrobin_roundrobin', 'roundrobin_ucb', 'roundrobin_ts',
              'ucb_over_intervention_pairs', 'ts_over_intervention_pairs', 'convex_explorer']

    # models = ['ucb_over_intervention_pairs', 'ts_over_intervention_pairs']

    # Set up the variables required to run the simulation

    num_intermediate_contexts = 10
    num_causal_variables = num_intermediate_contexts
    num_interventions = num_causal_variables * 2 + 1

    exploration_budget = 7500
    diff_prob_transition = 0.1
    default_reward = 0.5
    diff_in_best_reward = 0.3
    # Set the varying parameter
    m_parametersRange = np.array(range(num_intermediate_contexts, 1, -1))
    # m_parametersRange = [10,4,2] # testing parameter range
    lambdaValues = np.zeros(len(m_parametersRange))

    num_sims = 500
    # The below is a flag for models that treat the problem as a one stage problem
    simple_models = ["ucb_over_intervention_pairs", "ts_over_intervention_pairs"]
    for regret_metric_name in ["simple_regret", "prob_best_intervention"]:
        for stochastic_flag in [True, False]:
            # The outputs are stored in the below matrix
            average_regret_matrix = np.zeros((len(m_parametersRange), len(models)), dtype=np.float32)
            for index in tqdm(range(len(m_parametersRange)), desc="Progress"):
                m_param = m_parametersRange[index]
                causal_parameters_vector = m_param * np.ones(num_intermediate_contexts, dtype=np.int8)
                causal_parameters_diag_matrix = np.diag(causal_parameters_vector)
                # Generate the required matrices from the above hyperparameters
                if stochastic_flag:
                    transition_matrix_initialization = setup.generate_stochastic_transition_matrix(
                        num_intermediate_contexts,
                        num_interventions,
                        diff_prob_transition)
                else:
                    transition_matrix_initialization = setup.generate_deterministic_transition_matrix(
                        num_intermediate_contexts,
                        num_interventions)

                lambdaValue, _ = utilities.solveFConvexProgram(transition_matrix_initialization,
                                                               causal_parameters_diag_matrix)
                lambdaValues[index] = lambdaValue ** 2

                print("\nm_param=", m_param)
                avg_regret_for_models = utilities.run_multiple_sims_multiple_models(models, num_sims,
                                                                                    exploration_budget,
                                                                                    num_intermediate_contexts,
                                                                                    num_interventions,
                                                                                    diff_prob_transition,
                                                                                    default_reward,
                                                                                    diff_in_best_reward,
                                                                                    stochastic=stochastic_flag,
                                                                                    regret_metric_name=regret_metric_name,
                                                                                    m_param=m_param)

                avg_regret_for_models = np.minimum(avg_regret_for_models, 1.0)
                average_regret_matrix[index] = avg_regret_for_models

                # Set print options to display the entire array
                np.set_printoptions(threshold=np.inf)
                # Print the progress as a log
                print("\naverage regret so far = ", average_regret_matrix)
                # Reset the threshold for printing
                np.set_printoptions(threshold=False)

            # Now we saved the obtained values to file.
            if stochastic_flag:
                file_path = "outputs/" + regret_metric_name + "_with_" + "lambda.txt"
            else:
                file_path = "outputs/" + regret_metric_name + "_with_" + "lambda_deterministic.txt"

            # Headers for each column
            headers = ['lambda'] + models
            # Prepend the row headings
            average_regret_matrix_for_print = np.hstack((np.array(lambdaValues).reshape(-1, 1),
                                                         average_regret_matrix))

            # Open the file for writing
            with open(file_path, 'w') as file:
                # Write the headers as the first line
                header_line = '\t'.join(headers)
                file.write(header_line + '\n')

                # noinspection PyTypeChecker
                # Save the matrix to the file
                np.savetxt(file, average_regret_matrix_for_print, delimiter='\t', fmt='%0.6f')

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
