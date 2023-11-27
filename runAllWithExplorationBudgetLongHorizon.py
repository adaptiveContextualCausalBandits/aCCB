import numpy as np
import time
import setup_contextualcausalbandit as setup
import utilities
import importlib
from tqdm import tqdm


def run_multiple_sims_multiple_models(models, num_sims, exploration_budget, num_intermediate_contexts,
                                      num_interventions, diff_prob_transition, default_reward,
                                      diff_in_best_reward, stochastic=True):
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
            simple_flag = True if model in simple_modules else False
            mymodule = importlib.import_module(model)
            if not simple_flag:
                sampled_transition_probabilities, sampled_average_reward_matrix = \
                    mymodule.run_one_sim(exploration_budget, transition_matrix, reward_matrix)
                regret = utilities.get_prob_optimal_reward(sampled_transition_probabilities,
                                                           sampled_average_reward_matrix)
            else:
                sampled_average_reward_vector = mymodule.run_one_sim(exploration_budget, transition_matrix,
                                                                     reward_matrix)
                regret = utilities.get_prob_optimal_reward_simple(sampled_average_reward_vector)
            total_regret[model_num] += regret
    average_regret = total_regret / num_sims
    return average_regret


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(9)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    models = ['roundrobin_roundrobin', 'roundrobin_ucb', 'roundrobin_ts',
              'ucb_over_intervention_pairs', 'ts_over_intervention_pairs', 'convex_explorer']

    # models = ['ucb_over_intervention_pairs', 'ts_over_intervention_pairs']

    # Set up the variables required to run the simulation
    exploration_budgets = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 100000]

    # Set up the variables required to run the simulation

    # num_intermediate_contexts = 10
    num_intermediate_contexts = 5
    num_causal_variables = num_intermediate_contexts
    num_interventions = num_causal_variables * 2 + 1

    diff_prob_transition = 0.1
    default_reward = 0.5
    diff_in_best_reward = 0.3

    num_sims = 100
    # The below is a flag for models that treat the problem as a one stage problem
    simple_modules = ["ucb_over_intervention_pairs", "ts_over_intervention_pairs"]

    for stochastic_flag in [True, False]:
        # The outputs are stored in the below matrix
        average_regret_matrix = np.zeros((len(exploration_budgets), len(models)), dtype=np.float32)
        for index in tqdm(range(len(exploration_budgets)), desc="Progress"):
            exploration_budget = exploration_budgets[index]

            print("\nexploration_budget=", exploration_budget)
            avg_regret_for_models = run_multiple_sims_multiple_models(models, num_sims, exploration_budget,
                                                                      num_intermediate_contexts,
                                                                      num_interventions,
                                                                      diff_prob_transition, default_reward,
                                                                      diff_in_best_reward,
                                                                      stochastic=stochastic_flag)

            # In this case we need the avg regret as a ratio to diff in best reward
            avg_regret_for_models = avg_regret_for_models / diff_in_best_reward
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
            file_path = "outputs/model_regret_with_exploration_budget_long_horizon.txt"
        else:
            file_path = "outputs/model_regret_with_exploration_budget_long_horizon_deterministic.txt"
        # Headers for each column
        headers = ['exploration_budget'] + models
        # Prepend the row headings
        average_regret_matrix_for_print = np.hstack((np.array(exploration_budgets).reshape(-1, 1),
                                                     average_regret_matrix))

        # Open the file for writing
        with open(file_path, 'w') as file:
            # Write the headers as the first line
            header_line = '\t'.join(
                headers)  # Use a tab separator
            file.write(header_line + '\n')

            # noinspection PyTypeChecker
            # Save the matrix to the file
            np.savetxt(file, average_regret_matrix_for_print, delimiter='\t', fmt='%0.6f')

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
