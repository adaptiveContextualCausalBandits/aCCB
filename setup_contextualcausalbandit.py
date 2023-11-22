import numpy as np


def generate_deterministic_transition_matrix(num_intermediate_contexts, num_interventions):
    """
    Generate the transition matrix where we deterministically transition to one of the intermediate contexts
    on choosing an intervention do(X_i=1) at the start state
    Note: m_0 = num_intermediate_contexts

    :param num_intermediate_contexts: integer
    :param num_causal_variables: integer
    :return: transition probability matrix: float R^{num_intermediate_contexts,2*num_causal_variables+1}

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


def generate_stochastic_transition_matrix(num_intermediate_contexts, num_interventions, epsilon=0.25):
    """
    Generate the transition matrix where we stochastically transition to one of the intermediate contexts
    on choosing an intervention do(X_i=1) at the start state
    Note: m_0 = num_intermediate_contexts

    :param num_intermediate_contexts: integer
    :param num_causal_variables: integer
    :return: transition probability matrix: float R^{num_intermediate_contexts,2*num_causal_variables+1}

    """

    # Create a matrix with all 1s for the remaining rows beyond num_intermediate_contexts
    remaining_rows = np.ones((num_interventions - num_intermediate_contexts, num_intermediate_contexts))
    # Now change the matrix to make it a uniform transition matrix
    uniform_transitions = 1 / num_intermediate_contexts * remaining_rows

    # Create a matrix where transition to 1 state is epsilon more probable than transition to other states.
    epsilon_part_first_matrix = epsilon * np.eye(num_intermediate_contexts)
    first_matrix = epsilon_part_first_matrix + (1 - epsilon) / num_intermediate_contexts * \
                   np.ones((num_intermediate_contexts, num_intermediate_contexts))

    # Stack the identity matrix on top of the matrix with 1s
    transition_matrix = np.vstack((first_matrix, uniform_transitions))

    return transition_matrix


# def generate_reward_matrix()


if __name__ == "__main__":
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    num_intermediate_contexts = 25
    num_causal_variables = 25
    num_interventions = num_causal_variables * 2 + 1
    det_transition_matrix = generate_deterministic_transition_matrix(num_intermediate_contexts, num_interventions)
    print("deterministic_transition_matrix=", det_transition_matrix)

    stochastic_transition_matrix = generate_stochastic_transition_matrix(num_intermediate_contexts, num_interventions, 0.3)
    print("stochastic_transition_matrix =", stochastic_transition_matrix)
