# Causal Contextual Bandits with Adaptive Contexts

## Introduction

This repository contains the implementation of our research on causal contextual bandits with adaptive contexts. Our
work introduces a novel approach to reducing the bandit problem to a convex minimization problem, providing significant
improvements in algorithmic efficiency and effectiveness as shown
by our [experimental results](https://github.com/adaptiveContextualCausalBandits/aCCB/tree/main/outputs/plots).

## Features

We study an algorithm to minimize regret in the causal contextual bandit setting. Here we consider bandits whose
contexts can be reached stochastically by interventions at a start state.

- Implementation of a convex minimization approach to causal contextual bandit problems with adaptive context.
- Experiments comparing various exploration strategies, including uniform exploration (UE), UCB-based, and Thompson
  Sampling-based approaches.
- Analysis of the performance across different exploration budgets and contexts.

## Motivating Example

Consider an advertiser looking to post ads on a web-page, say Amazon. They may make requests for a certain type of user
demographic to Amazon. Based on this initial request, the platform may actually choose one particular user to show the
ad to. At this time, certain details about the user are revealed to the advertiser. For example, the platform may reveal
some user demographics, as well as certain details about their device. Based on these details, the advertiser may
choose one particular ad to show the user. In case the user clicks the ad, the advertiser receives a reward. The goal of
the learner is to find optimal choices for initial user preference, as well as ad-content such that user clicks are
maximized. We illustrate this example through the advertiser-motivation figure below where we indicate the choices
available for template and content interventions.

![Advertiser Motivation Figure below](images/adCCB.svg "Motivation for Adaptive Causal Contextual Bandits through an advertising example.")

## Getting Started

### Prerequisites

- Python 3.x
- Other requirements
  in [requirements.txt](https://github.com/adaptiveContextualCausalBandits/aCCB/blob/main/requirements.txt).

### Initial Setup

Clone the repository to your local machine:

```bash
git clone https://github.com/adaptiveContextualCausalBandits/aCCB.git
```

Check into the required directory and create a local environment
```bash
cd aCCB
# create a virtual environment
python -m venv venv

# activate the virtual environment
source venv/Scripts/activate
```

Install the required dependencies in the virtual environment:
```bash
pip install -r requirements.txt
```
NOTE: Since python version 3.12 does not come with setuptools pre-installed, additional setup may be required for the 
cvxpy package. Suggested to go for earlier versions of python.

### Running the Experiments

To run the experiments, use the following command:

```bash
python runAllExperiments.py
```

The above step may take a few hours depending on the speed of your machine. If you instead want to run the experiments 
in parallel, you may instead run the following four commands on four separate terminals.

```bash
python runAllWithDiffInBestReward.py
python runAllWithExplorationBudget.py
python runAllWithExplorationBudgetLongHorizon.py
python runAllWithLambda.py
python runAllWithNumIntermediateContexts.py
```

The above may take of the order of an hour on an Intel i7 CPU.


### Plot the results of the experiments

```bash
python run_plotters.py
```

## Results

Our experiments demonstrate the efficacy of the convex minimization approach, particularly in comparison to traditional
bandit algorithms. The results are detailed in the following plots:

- [Variation of Expected Regret with Exploration Budget](https://drive.google.com/file/d/1qWSt7Kv-sEi85dD4sjflLnQqRC7V_TCN/view?usp=sharing)
- [Additional Experiments](https://drive.google.com/drive/folders/1VMkeenDM797NtsR25_Fnsc3t3yZkuqy1?usp=sharing)

## Contributing

We welcome contributions from the community. If you would like to contribute, please fork the repository and submit a
pull request.


## License

This project is licensed under the MIT License - see
the [LICENSE](https://github.com/adaptiveContextualCausalBandits/aCCB/blob/main/LICENSE) file for details.

