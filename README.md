
# Causal Contextual Bandits with Adaptive Contexts

## Introduction

This repository contains the implementation of our research on causal contextual bandits with adaptive contexts. Our work introduces a novel approach to reducing the bandit problem to a convex minimization problem, providing significant improvements in algorithmic efficiency and effectiveness as shown 
by our [experimental results](https://github.com/adaptiveContextualCausalBandits/aCCB/tree/main/outputs/plots).

## Features
We study an algorithm 
- Implementation of the convex minimization approach to bandit problems.
- Experiments comparing various exploration strategies, including uniform exploration (UE), UCB-based, and Thompson Sampling-based approaches.
- Analysis of the performance across different exploration budgets and contexts.

## Getting Started

### Prerequisites

- Python 3.x
- Other requirements in [requirements.txt](https://github.com/adaptiveContextualCausalBandits/aCCB/blob/main/requirements.txt).

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/adaptiveContextualCausalBandits/aCCB.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Experiments

To run the experiments, use the following command:

```bash
python run_plotters.py
```

## Results

Our experiments demonstrate the efficacy of the convex minimization approach, particularly in comparison to traditional bandit algorithms. The results are detailed in the following plots:

- [Variation of Expected Regret with Exploration Budget](https://drive.google.com/file/d/1qWSt7Kv-sEi85dD4sjflLnQqRC7V_TCN/view?usp=sharing)
- [Additional Experiments](https://drive.google.com/drive/folders/1VMkeenDM797NtsR25_Fnsc3t3yZkuqy1?usp=sharing)

## Contributing

We welcome contributions from the community. If you would like to contribute, please fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License] - see the [LICENSE](https://github.com/adaptiveContextualCausalBandits/aCCB/blob/main/LICENSE) file for details.

