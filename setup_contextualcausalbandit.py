import numpy as np
import time


def generate_deterministic_transition_matrix(num_intermediate_contexts, num_interventions):
    """
    Generate the transition matrix where we deterministically transition to one of the intermediate contexts
    on choosing an intervention do(X_i=1) at the start state
    Assume: m_0 = num_intermediate_contexts

    :param num_intermediate_contexts: integer
    :param num_causal_variables: integer
    :return: transition probability matrix: float R^{2*num_causal_variables+1,num_intermediate_contexts}

    """
    # Create an identity matrix for the first num_intermediate_contexts rows
    identity_matrix = np.eye(num_intermediate_contexts)

    # Create a matrix with all 1s for the remaining rows beyond num_intermediate_contexts
    remaining_rows = np.ones((num_interventions - num_intermediate_contexts, num_intermediate_contexts))
    # Now change the matrix to make it a uniform transition matrix
    uniform_transitions = 1 / num_intermediate_contexts * remaining_rows

    # Stack the identity matrix on top of the matrix with 1s
    transition_matrix = np.vstack((identity_matrix, uniform_transitions))

    return transition_matrix


def generate_stochastic_transition_matrix(num_intermediate_contexts, num_interventions, diff_prob_transition=0.25):
    """
    Generate the transition matrix where we stochastically transition to one of the intermediate contexts
    on choosing an intervention do(X_i=1) at the start state
    Assume: m_0 = num_intermediate_contexts

    :param num_intermediate_contexts: integer
    :param num_causal_variables: integer
    :return: transition probability matrix: float R^{2*num_causal_variables+1,num_intermediate_contexts}

    """

    # Create a matrix with all 1s for the remaining rows beyond num_intermediate_contexts
    remaining_rows = np.ones((num_interventions - num_intermediate_contexts, num_intermediate_contexts))
    # Now change the matrix to make it a uniform transition matrix
    uniform_transitions = 1 / num_intermediate_contexts * remaining_rows

    # Create a matrix where transition to 1 state is epsilon more probable than transition to other states.
    epsilon_part_first_matrix = diff_prob_transition * np.eye(num_intermediate_contexts)
    first_matrix = epsilon_part_first_matrix + (1 - diff_prob_transition) / num_intermediate_contexts * \
                   np.ones((num_intermediate_contexts, num_intermediate_contexts))

    # Stack the identity matrix on top of the matrix with 1s
    transition_matrix = np.vstack((first_matrix, uniform_transitions))

    return transition_matrix


def generate_reward_matrix(num_intermediate_contexts, num_interventions_at_each_context, default_reward=0.5,
                           diff_in_best_reward=0.1):
    """

    :param num_intermediate_contexts: integer
    :param num_interventions_at_each_context: integer
    :param reward: float giving average reward for an intervention at a context
    :param diff_in_best_reward: float giving improvement in reward for best chosen arm
    :return: reward_matrix: float R^{num_intermediate_contexts,num_interventions_at_each_context}

    Assume: the m_i values are greater or equal to 2. Hence there can exist at least one intervention at q_i^j of 0
    Let this intervention be the first, with an expected reward of default_reward+diff_in_best_reward, with rest of the
    interventions having expected reward as default_reward
    """
    reward_matrix = default_reward * np.ones((num_intermediate_contexts, num_interventions_at_each_context))
    reward_matrix[0, 0] = default_reward + diff_in_best_reward
    return reward_matrix


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    num_intermediate_contexts = 25
    num_causal_variables = 25
    num_interventions = num_causal_variables * 2 + 1
    det_transition_matrix = generate_deterministic_transition_matrix(num_intermediate_contexts, num_interventions)
    print("deterministic_transition_matrix=", det_transition_matrix)

    diff_prob_transition = 0.1
    stochastic_transition_matrix = generate_stochastic_transition_matrix(num_intermediate_contexts, num_interventions,
                                                                         diff_prob_transition)
    print("stochastic_transition_matrix =", stochastic_transition_matrix)
    print("stochastic_transition_matrix.shape =", stochastic_transition_matrix.shape)
    default_reward = 0.5
    diff_in_best_reward = 0.3
    reward_matrix = generate_reward_matrix(num_intermediate_contexts, num_interventions, default_reward,
                                           diff_in_best_reward)
    print("reward_matrix = ", reward_matrix)
    print("reward_matrix.shape = ", reward_matrix.shape)

    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
