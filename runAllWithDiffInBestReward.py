import numpy as np
import time
import setup_contextualcausalbandit as setup
import utilities
import importlib
from tqdm import tqdm

if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    models = ['roundrobin_roundrobin', 'roundrobin_ucb', 'roundrobin_ts',
              'ucb_over_intervention_pairs', 'ts_over_intervention_pairs', 'convex_explorer']

    # models = ['ucb_over_intervention_pairs', 'ts_over_intervention_pairs']

    # Set up the variables required to run the simulation
    diff_in_best_reward_list = [0.01] + [(x + 1) * 0.05 for x in range(9)]
    # diff_in_best_reward_list = [0.01,0.1,0.45]

    # num_intermediate_contexts = 10
    num_intermediate_contexts = 5
    num_causal_variables = num_intermediate_contexts
    num_interventions = num_causal_variables * 2 + 1

    exploration_budget = 25_000
    diff_prob_transition = 0.1
    default_reward = 0.5
    # diff_in_best_reward = 0.3

    num_sims = 100
    # The below is a flag for models that treat the problem as a one stage problem
    simple_modules = ["ucb_over_intervention_pairs", "ts_over_intervention_pairs"]
    for regret_metric_name in ["simple_regret", "prob_best_intervention"]:
        for stochastic_flag in [True, False]:
            # The outputs are stored in the below matrix
            average_regret_matrix = np.zeros((len(diff_in_best_reward_list), len(models)), dtype=np.float32)
            for index in tqdm(range(len(diff_in_best_reward_list)), desc="Progress"):
                diff_in_best_reward = diff_in_best_reward_list[index]

                print("\ndiff_in_best_reward=", diff_in_best_reward)
                avg_regret_for_models = utilities.run_multiple_sims_multiple_models(models, num_sims,
                                                                                    exploration_budget,
                                                                                    num_intermediate_contexts,
                                                                                    num_interventions,
                                                                                    diff_prob_transition,
                                                                                    default_reward,
                                                                                    diff_in_best_reward,
                                                                                    stochastic=stochastic_flag,
                                                                                    regret_metric_name=regret_metric_name,
                                                                                    m_param=2)

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
                file_path = "outputs/" + regret_metric_name + "_with_" + "diff_in_best_reward.txt"
            else:
                file_path = "outputs/" + regret_metric_name + "_with_" + "diff_in_best_reward_deterministic.txt"

            # Headers for each column
            headers = ['diff_in_best_reward'] + models
            # Prepend the row headings
            average_regret_matrix_for_print = np.hstack((np.array(diff_in_best_reward_list).reshape(-1, 1),
                                                         average_regret_matrix))

            # Open the file for writing
            with open(file_path, 'w') as file:
                # Write the headers as the first line
                header_line = '\t'.join(headers)  # Use a tab separator
                file.write(header_line + '\n')

                # noinspection PyTypeChecker
                # Save the matrix to the file
                np.savetxt(file, average_regret_matrix_for_print, delimiter='\t', fmt='%0.6f')

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))