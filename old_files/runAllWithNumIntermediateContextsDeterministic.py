import numpy as np
import time
import setup_contextualcausalbandit as setup
import utilities
import importlib
from tqdm import tqdm


def run_multiple_sims_multiple_models_deterministic(models, num_sims, exploration_budget, num_intermediate_contexts,
                                                    num_interventions, default_reward, diff_in_best_reward):
    total_regret = np.zeros((len(models)), dtype=np.float32)

    for i in range(num_sims):
        # Generate the required matrices from the above hyperparameters
        transition_matrix = setup.generate_deterministic_transition_matrix(num_intermediate_contexts,
                                                                           num_interventions)
        reward_matrix = setup.generate_reward_matrix(num_intermediate_contexts, num_interventions,
                                                     default_reward, diff_in_best_reward)
        for model_num in range(len(models)):
            model = models[model_num]
            simple_flag = True if model in simple_models else False
            mymodule = importlib.import_module(model)
            if not simple_flag:
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
    num_intermediate_contexts_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 50, 100]
    # num_intermediate_contexts_list = [100, 2, 5, 10, 20]
    exploration_budget = 10_000
    diff_prob_transition = 0.1
    default_reward = 0.5
    diff_in_best_reward = 0.3

    num_sims = 100
    # The below is a flag for models that treat the problem as a one stage problem
    simple_models = ["ucb_over_intervention_pairs", "ts_over_intervention_pairs"]
    # The outputs are stored in the below matrix
    average_regret_matrix = np.zeros((len(num_intermediate_contexts_list), len(models)), dtype=np.float32)
    for context_index in tqdm(range(len(num_intermediate_contexts_list)), desc="Budget Progress"):
        num_intermediate_contexts = num_intermediate_contexts_list[context_index]
        num_causal_variables = num_intermediate_contexts
        num_interventions = num_causal_variables * 2 + 1

        print("\nnum_intermediate_contexts=", num_intermediate_contexts)
        avg_regret_for_models = run_multiple_sims_multiple_models_deterministic(models, num_sims, exploration_budget,
                                                                                num_intermediate_contexts,
                                                                                num_interventions,
                                                                                default_reward, diff_in_best_reward)
        average_regret_matrix[context_index] = avg_regret_for_models

        # Set print options to display the entire array
        np.set_printoptions(threshold=np.inf)
        # Print the progress as a log
        print("\naverage regret so far = ", average_regret_matrix)
        # Reset the threshold for printing
        np.set_printoptions(threshold=False)

    # Now we saved the obtained values to file.
    file_path = "../outputs/model_regret_with_num_intermediate_contexts_deterministic_transitions.txt"
    # Headers for each column
    headers = ['num_intermediate_contexts'] + models
    # Open the file for writing
    with open(file_path, 'w') as file:
        # Write the headers as the first line
        header_line = '\t'.join(headers)  # Use a tab separator (you can change it to a different delimiter if needed)
        file.write(header_line + '\n')

        # noinspection PyTypeChecker
        # Save the matrix to the file
        np.savetxt(file, average_regret_matrix, delimiter='\t', fmt='%0.6f')

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
