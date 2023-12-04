import numpy as np
import time
import utilities


def set_experiment_dict(
        num_intermediate_contexts=5,
        num_causal_variables=5,
        exploration_budget=25_000,
        diff_prob_transition=0.1,
        default_reward=0.5,
        diff_in_best_reward=0.3,
        num_sims=100,
        m_param=2,
        varying_feature_name="diff_in_best_reward",
        varying_feature_values=[0.01] + [(x + 1) * 0.05 for x in range(9)],
        experiment_variation=""
):
    out_dict = {}
    out_dict["num_intermediate_contexts"] = num_intermediate_contexts
    out_dict["num_causal_variables"] = num_causal_variables
    out_dict["exploration_budget"] = exploration_budget
    out_dict["diff_prob_transition"] = diff_prob_transition
    out_dict["default_reward"] = default_reward
    out_dict["diff_in_best_reward"] = diff_in_best_reward
    out_dict["diff_in_best_reward"] = diff_in_best_reward
    out_dict["num_sims"] = num_sims
    out_dict["m_param"] = m_param
    out_dict["varying_feature_name"] = varying_feature_name
    out_dict["varying_feature_values"] = varying_feature_values
    out_dict["experiment_variation"] = experiment_variation
    return out_dict


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    experiments = []
    experiment = set_experiment_dict(
        num_intermediate_contexts=5,
        num_causal_variables=5,
        exploration_budget=25_000,
        diff_prob_transition=0.1,
        default_reward=0.5,
        diff_in_best_reward=0.3,
        num_sims=100,
        m_param=2,
        varying_feature_name="diff_in_best_reward",
        varying_feature_values=[0.01] + [(x + 1) * 0.05 for x in range(9)]
    )
    experiments.append(experiment)
    experiment = set_experiment_dict(
        num_intermediate_contexts=20,
        num_causal_variables=20,
        exploration_budget=25_000,
        diff_prob_transition=0.1,
        default_reward=0.5,
        diff_in_best_reward=0.3,
        num_sims=500,
        m_param=2,
        varying_feature_name="exploration_budget",
        varying_feature_values=[500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000]
    )
    experiments.append(experiment)

    experiment = set_experiment_dict(
        num_intermediate_contexts=5,
        num_causal_variables=5,
        exploration_budget=25_000,
        diff_prob_transition=0.1,
        default_reward=0.5,
        diff_in_best_reward=0.3,
        num_sims=100,
        m_param=2,
        varying_feature_name="exploration_budget",
        varying_feature_values=[100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 100000],
        experiment_variation="_long_horizon"
    )
    experiments.append(experiment)

    experiment = set_experiment_dict(
        num_intermediate_contexts=10,
        num_causal_variables=10,
        exploration_budget=7500,
        diff_prob_transition=0.1,
        default_reward=0.5,
        diff_in_best_reward=0.3,
        num_sims=500,
        m_param=2,
        varying_feature_name="m_param",
        varying_feature_values=[10, 9, 8, 7, 6, 5, 4, 3, 2]
    )
    experiments.append(experiment)
    experiment = set_experiment_dict(
        num_intermediate_contexts=10,
        num_causal_variables=10,
        exploration_budget=10_000,
        diff_prob_transition=0.1,
        default_reward=0.5,
        diff_in_best_reward=0.3,
        num_sims=100,
        m_param=2,
        varying_feature_name="num_intermediate_contexts",
        varying_feature_values=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 50, 100]
    )
    experiments.append(experiment)
    experiment_number = 1
    for experiment in experiments:
        exp_start_time = time.time()
        print("-" * 50)
        print("Experiment " + str(experiment_number))
        print("-" * 50)

        utilities.run_all_with_feature(num_intermediate_contexts=experiment["num_intermediate_contexts"],
                                       num_causal_variables=experiment["num_causal_variables"],
                                       exploration_budget=experiment["exploration_budget"],
                                       diff_prob_transition=experiment["diff_prob_transition"],
                                       default_reward=experiment["default_reward"],
                                       diff_in_best_reward=experiment["diff_in_best_reward"],
                                       num_sims=experiment["num_sims"],
                                       m_param=experiment["m_param"],
                                       varying_feature_name=experiment["varying_feature_name"],
                                       varying_feature_values=experiment["varying_feature_values"],
                                       experiment_variation=experiment["experiment_variation"])
        experiment_number += 1

        print("time taken to run experiment = %0.6f seconds" % (time.time() - exp_start_time))

    print("time taken to run all experiments = %0.6f seconds" % (time.time() - start_time))
