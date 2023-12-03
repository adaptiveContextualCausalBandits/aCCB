import numpy as np
import time
import setup_contextualcausalbandit as setup, utilities


def getPOnDoNothing(exploration_budget, numStates, transition_matrix, m0param, prob_x0j_is_one=0.5):
    numVariables = numStates
    numInterventions = 2 * numVariables + 1
    realizationsMatrixOnDoNothing = np.zeros((numInterventions, numStates))
    numTimesInterventionPerformed = np.zeros(numInterventions)

    numTimesDoNothingPerformed = exploration_budget
    transitionsOnDoNothing = np.random.multinomial(exploration_budget, transition_matrix[numVariables, :], None)
    numTimesInterventionPerformed[-(m0param):] = numTimesDoNothingPerformed
    realizationsMatrixOnDoNothing[-(m0param):, :] = transitionsOnDoNothing
    for i in range(m0param, numInterventions - m0param):
        ni = np.random.binomial(numTimesDoNothingPerformed, prob_x0j_is_one)
        numTimesInterventionPerformed[i] = ni
        transitionOni = np.random.multinomial(ni, transition_matrix[i, :], None)
        realizationsMatrixOnDoNothing[i, :] = transitionOni
    numTimesInterventionPerformed[numVariables] = numTimesDoNothingPerformed
    realizationsMatrixOnDoNothing[numVariables, :] = transitionsOnDoNothing
    numTimesInterventionPerformed[numTimesInterventionPerformed == 0] = 1e-8
    realizationsProbsMatrixOnDoNothing = realizationsMatrixOnDoNothing / numTimesInterventionPerformed[:, None]
    return realizationsProbsMatrixOnDoNothing


def getPOnInterventions(exploration_budget, numStates, transition_matrix, m0param):
    numVariables = numStates
    numInterventions = 2 * numVariables + 1
    realizedProbsMatrixOnInterventions = np.zeros((numInterventions, numStates))
    numRoundsPerIntervention = exploration_budget // m0param
    for i in range(m0param):
        numTimesVector = np.random.multinomial(numRoundsPerIntervention, transition_matrix[i, :], size=None)
        reqdProbVector = numTimesVector / numRoundsPerIntervention
        realizedProbsMatrixOnInterventions[i, :] = reqdProbVector
    return realizedProbsMatrixOnInterventions


def getRealizedProbMatrix(exploration_budget, numStates, transition_matrix, m0param, prob_x0j_is_one=0.5):
    realizedProbsMatrixOnDoNothing = getPOnDoNothing(exploration_budget // 2, numStates=numStates,
                                                     transition_matrix=transition_matrix, m0param=m0param,
                                                     prob_x0j_is_one=prob_x0j_is_one)
    realizedProbsMatrixOnInterventions = getPOnInterventions(exploration_budget // 2, numStates,
                                                             transition_matrix=transition_matrix, m0param=m0param)
    realizedProbsMatrix = realizedProbsMatrixOnInterventions + realizedProbsMatrixOnDoNothing
    return realizedProbsMatrix


def findMFromQVector(qVector):
    """

    :param qVector:
    :return: the value of m from the q vector
    """
    np.sort(qVector)
    for m in range(2, len(qVector)):
        if (m + 1) * qVector[m] > 1:
            return m
    return len(qVector)


def estimateMVectorFromQHat(QHat):
    """

    :param QHat:
    :return: the value of M estimate vector from QHat matrix
    """
    numStates, _ = QHat.shape
    numVariables = numStates
    mHat = np.zeros(numStates)
    for state in range(numStates):
        qlower = np.zeros(numVariables)
        for i in range(numVariables):
            qlower[i] = min(QHat[state, i], QHat[state, -(i + 1)])
        mHat[state] = findMFromQVector(qlower)
    return mHat


def sample_qhat(budget_per_intermediate_context_for_mi_estimation, num_intermediate_contexts, num_interventions,
                m_parameter_at_intermediate_state, prob_of_var_not_in_m_being_1=0.5):
    qhat = np.ones((num_intermediate_contexts, num_interventions))
    numTimesStateSeen = budget_per_intermediate_context_for_mi_estimation

    bernoulli_sd = prob_of_var_not_in_m_being_1 * (1 - prob_of_var_not_in_m_being_1)
    CLT_sd_per_state = bernoulli_sd * (budget_per_intermediate_context_for_mi_estimation ** -0.5)
    # Reshape it to (num_intermediate_contexts, 1)
    CLT_sd_per_state = CLT_sd_per_state[:, np.newaxis]
    # Broadcast to (num_interventions, num_intermediate_contexts-m_parameter_at_intermediate_state)
    CLT_sd_per_state_broadcast = CLT_sd_per_state * np.ones(
        (1, num_intermediate_contexts - m_parameter_at_intermediate_state))
    # print("CLT_sd_per_state_broadcast=", CLT_sd_per_state_broadcast,CLT_sd_per_state_broadcast.shape)
    error_terms = np.random.normal(
        np.zeros((num_intermediate_contexts, num_intermediate_contexts - m_parameter_at_intermediate_state)),
        CLT_sd_per_state_broadcast, size=None)
    # print("error_terms=",error_terms,error_terms.shape)

    # Set the first m_param columns to 0
    qhat[:, :m_parameter_at_intermediate_state] = 0
    qhat[:, m_parameter_at_intermediate_state:num_intermediate_contexts] = 0.5 - error_terms
    # Now add the reversed error terms for the remaining columns
    qhat[:, num_intermediate_contexts: 2 * num_intermediate_contexts - m_parameter_at_intermediate_state] = \
        0.5 + error_terms[:, ::-1]
    return qhat


def estimate_rewards(budget_per_intermediate_context_for_reward_estimation, reward_matrix,
               m_parameter_at_intermediate_state, prob_of_var_not_in_m_being_1=0.5):
    num_intermediate_contexts, num_interventions = reward_matrix.shape
    num_times_intervention_at_state_seen = np.rint(0.5 * (
            budget_per_intermediate_context_for_reward_estimation[:, np.newaxis] @ np.ones((1, num_interventions))))

    # print("num_times_intervention_at_state_seen init=", num_times_intervention_at_state_seen)
    bernoulli_sd = prob_of_var_not_in_m_being_1 * (1 - prob_of_var_not_in_m_being_1)
    CLT_sd_per_state = bernoulli_sd * (budget_per_intermediate_context_for_reward_estimation ** -0.5)
    # Reshape it to (num_intermediate_contexts, 1)
    CLT_sd_per_state = CLT_sd_per_state[:, np.newaxis]
    # Broadcast to (num_interventions, num_intermediate_contexts-m_parameter_at_intermediate_state)
    CLT_sd_per_state_broadcast = CLT_sd_per_state * np.ones(
        (1, num_intermediate_contexts - m_parameter_at_intermediate_state))
    # print("CLT_sd_per_state_broadcast=", CLT_sd_per_state_broadcast, CLT_sd_per_state_broadcast.shape)
    error_terms = np.random.normal(
        np.zeros((num_intermediate_contexts, num_intermediate_contexts - m_parameter_at_intermediate_state)),
        CLT_sd_per_state_broadcast, size=None)
    # Multiply error terms with number of times each intermediate context is seen
    error_terms = 0.5 * budget_per_intermediate_context_for_reward_estimation[:, np.newaxis] * error_terms
    # print("error_terms=", error_terms, error_terms.shape)

    # Set the first m_param columns values due to Do interventions
    num_times_intervention_at_state_seen[:, :m_parameter_at_intermediate_state] = \
        budget_per_intermediate_context_for_reward_estimation[:, np.newaxis] / (2 * m_parameter_at_intermediate_state)
    # print("num_times_intervention_at_state_seen after adding first do interventions =",
    #       num_times_intervention_at_state_seen)
    # For the m_parameter_at_intermediate_state - num_intermediate_contexts remaining terms, subtract the error terms
    num_times_intervention_at_state_seen[:, m_parameter_at_intermediate_state:num_intermediate_contexts] -= error_terms
    # Now add the reversed error terms for the remaining columns
    num_times_intervention_at_state_seen[:, \
    num_intermediate_contexts: 2 * num_intermediate_contexts - m_parameter_at_intermediate_state] += \
        error_terms[:, ::-1]
    num_times_intervention_at_state_seen = np.rint(num_times_intervention_at_state_seen)
    num_times_intervention_at_state_seen = num_times_intervention_at_state_seen.astype(np.int32)
    # print("num_times_intervention_at_state_seen=", num_times_intervention_at_state_seen)
    # Sample the reward estimates from multiple binomials at once
    # print("num_times_intervention_at_state_seen.shape, reward_matrix.shape",num_times_intervention_at_state_seen.shape, reward_matrix.shape)
    total_reward_estimate = np.random.binomial(num_times_intervention_at_state_seen, reward_matrix)
    sampled_reward_estimate = total_reward_estimate/num_times_intervention_at_state_seen
    # print("total_reward_estimate=", total_reward_estimate)
    # print("sampled_reward_estimate=", sampled_reward_estimate)
    return sampled_reward_estimate


def run_one_sim(exploration_budget, transition_matrix, reward_matrix, prob_x0j_is_one=0.5,
                m_parameter_at_intermediate_states=2, prob_of_var_not_in_m_being_1=0.5):
    """
    Here we will run one simulation of the algorithm which uses round robin at the start state, and ucb at the
    intermediate contexts.

    :param exploration_budget:
    :param transition_matrix:
    :param reward_matrix:
    :return:
    :sampled_transition_probabilities: transition probability matrix estimate
    :sampled_average_reward_matrix: reward matrix estimate
    """

    num_interventions_at_state_0, num_intermediate_contexts = transition_matrix.shape
    _, num_interventions = reward_matrix.shape
    reward_epsilon = reward_matrix[0, 0] - reward_matrix[0, 1]

    # Get budget per intervention for transition matrix estimation
    budget_per_intervention_for_P_estimation = np.zeros(num_interventions_at_state_0)  # Initialize with zeros

    budget_per_intervention_for_P_estimation[:num_intermediate_contexts] = exploration_budget // \
                                                                           (6 * num_intermediate_contexts)
    budget_per_intervention_for_P_estimation[-1] = exploration_budget // 6
    # print("budget_per_intervention_for_P_estimation", budget_per_intervention_for_P_estimation)
    # print("sum(budget_per_intervention_for_P_estimation)", sum(budget_per_intervention_for_P_estimation))
    # print("budget_per_intervention_for_P_estimation.shape", budget_per_intervention_for_P_estimation.shape)

    # Estimate the transition matrix
    transition_matrix_estimate = getRealizedProbMatrix(exploration_budget // 3, numStates=num_intermediate_contexts,
                                                       transition_matrix=transition_matrix,
                                                       m0param=num_intermediate_contexts,
                                                       prob_x0j_is_one=prob_x0j_is_one)
    # print("transition_matrix_estimate=", transition_matrix_estimate)
    _, ftilde = utilities.getFMaxMin(transition_matrix_estimate)
    budget_per_intervention_for_mi_estimation = (exploration_budget // 3 * ftilde).astype(np.int32)
    total_budget_for_mi_estimate_per_intervention = budget_per_intervention_for_mi_estimation + \
                                                    budget_per_intervention_for_P_estimation
    # print("ftilde=", ftilde)
    # print("budget_per_intervention_for_mi_estimation=", budget_per_intervention_for_mi_estimation)
    # print("total_budget_for_mi_estimate_per_intervention=", total_budget_for_mi_estimate_per_intervention)

    sampled_transition_matrix = np.zeros((num_interventions_at_state_0, num_intermediate_contexts))
    for intervention in range(num_interventions_at_state_0):
        sampled_transition_matrix[intervention] += \
            np.random.multinomial(total_budget_for_mi_estimate_per_intervention[intervention],
                                  transition_matrix[intervention], None)
    budget_per_intermediate_context_for_mi_estimation = np.sum(sampled_transition_matrix, axis=0)

    # print("ftilde=", ftilde)
    # print("ftilde.shape = ", ftilde.shape)
    # print("transition_matrix_estimate=", transition_matrix_estimate)
    # print("sampled_transition_probabilities=", sampled_transition_probabilities)
    # print("budget_per_intermediate_context_for_mi_estimation=", budget_per_intermediate_context_for_mi_estimation)
    # print("total budget so far", sum(budget_per_intermediate_context_for_mi_estimation))
    qhat = sample_qhat(budget_per_intermediate_context_for_mi_estimation, num_intermediate_contexts, num_interventions,
                       m_parameter_at_intermediate_states, prob_of_var_not_in_m_being_1=prob_of_var_not_in_m_being_1)
    mHat = estimateMVectorFromQHat(qhat)
    MHat = np.diag(mHat)
    # Set print options to display the entire array
    np.set_printoptions(threshold=np.inf)
    # print("qhat=", qhat)
    # Reset print options to their defaults

    # print("MHat=", MHat)
    minimax, fStar = utilities.solveFConvexProgram(transition_matrix_estimate, MHat)
    # print("minimax, fStar, fStar.shape", minimax, fStar, fStar.shape)
    budget_per_intervention_for_reward_estimation = (exploration_budget // 3 * fStar).astype(np.int32)

    for intervention in range(num_interventions_at_state_0):
        sampled_transition_matrix[intervention] += \
            np.random.multinomial(budget_per_intervention_for_reward_estimation[intervention],
                                  transition_matrix[intervention], None)
    budget_per_intermediate_context_for_reward_estimation = np.sum(sampled_transition_matrix, axis=0)

    # print("budget_per_intermediate_context_for_reward_estimation=",
    #       budget_per_intermediate_context_for_reward_estimation,
    #       budget_per_intermediate_context_for_reward_estimation.shape)
    # print("sum(budget_per_intermediate_context_for_reward_estimation)=",
    #       sum(budget_per_intermediate_context_for_reward_estimation))
    sampled_reward_estimate = estimate_rewards(budget_per_intermediate_context_for_reward_estimation, reward_matrix,
                      m_parameter_at_intermediate_states, prob_of_var_not_in_m_being_1=prob_of_var_not_in_m_being_1)
    np.set_printoptions(threshold=False)

    return transition_matrix_estimate, sampled_reward_estimate



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

    exploration_budget = 2700
    # deterministic transitions
    sampled_transition_probabilities, sampled_average_reward_matrix = \
        run_one_sim(exploration_budget, det_transition_matrix, reward_matrix)

    print("sampled_transition_probabilities =", sampled_transition_probabilities)
    print("sampled_average_reward_matrix =", sampled_average_reward_matrix)
    # stochastic transitions
    sampled_transition_probabilities, sampled_average_reward_matrix = \
        run_one_sim(exploration_budget, stochastic_transition_matrix, reward_matrix)
    print("sampled_transition_probabilities, =", sampled_transition_probabilities)
    print("sampled_average_reward_matrix, =", sampled_average_reward_matrix)

    prob_of_optimal_reward = utilities.get_prob_optimal_reward(sampled_transition_probabilities,
                                                               sampled_average_reward_matrix)
    print("prob_of_optimal_reward = ", prob_of_optimal_reward)

    simple_regret = utilities.get_simple_regret(sampled_transition_probabilities, sampled_average_reward_matrix,
                                                stochastic_transition_matrix, reward_matrix)
    print("simple_regret = ", simple_regret)

    num_sims = 100
    average_prob_optimal_reward = utilities.run_multiple_sims(num_sims, exploration_budget,
                                                              stochastic_transition_matrix, reward_matrix,
                                                              simulation_module="convex_explorer",
                                                              simple=False,
                                                              regret_metric_name="prob_best_intervention")
    print("average_prob_optimal_reward=", average_prob_optimal_reward)

    average_simple_regret = utilities.run_multiple_sims(num_sims, exploration_budget,
                                                              stochastic_transition_matrix, reward_matrix,
                                                              simulation_module="convex_explorer",
                                                              simple=False,
                                                              regret_metric_name="simple_regret")
    print("average_simple_regret=", average_simple_regret)
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
