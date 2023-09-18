# Risk-Averse Trading Agent Using Dueling DDQN with Penalized Reward
---

## Introduction

The application of Deep Reinforcement Learning (DRL) in stock trading is a rising trend, providing models with the ability to learn and act within dynamic stock markets. This research emphasizes the exploration of critic-only, actor-only, and actor-critic DRL methods within trading scenarios, with a special emphasis on the Deep Q-Network (DQN) structure.

## Previous Works

1. **Critic-only DRL**: Chen et al. (2019) developed a Deep Recurrent Q-Network (DRQN) that integrates a recurrent neural network with DQN to enhance temporal sequence processing. Using the S&P500 ETF price history, this DRQN variant outperformed benchmark strategies such as buy-and-hold, with an annual expected return of about 22-23%.

2. **Actor-only DRL**: Notable methods include LSTM and Direct Deep Reinforcement (DDR) systems, with research from Deng et al. (2017) and Wu et al. (2019). Deng et al. applied a recurrent deep neural network (RDNN) for optimal trading strategies within the Chinese stock market, witnessing marked improvements in risk-adjusted returns over traditional strategies.

3. **Actor-Critic DRL**: This combines actor and critic training, with the potential to be a more robust technique, although it remains relatively underexplored.

Other studies, such as Théate et al. (2021), have championed the effectiveness of DQN in algorithmic trading, with the Trading Deep Q-Network (TDQN) achieving success in optimizing trading positions in stock markets. Zejnullahu et al. (2022) explored the Double Deep Q-learning (DDQN) algorithm in trading single assets, illustrating the promise of DDQN models in the versatile world of stock trading.

## Methodology

The research builds upon Zejnullahu et al. (2022), introducing two critical enhancements:

1. **Dueling Structured Network**: The conventional fully-connected network is replaced with a dueling structure, focusing on improved performance.

2. **Varying Penalty Strategy**: This modification gradually adjusts penalty factors, aiming to train a more risk-averse agent.

### Dueling Structure

This architecture, introduced by Wang et al. (2015), extends traditional DRL models by bifurcating the Q-function into state value function (V) and action advantage function (A). The decomposition aids in learning the state's value and each action's relative advantage, refining Q-value estimations.

In financial settings, the dueling structure proves intuitive. Given the unpredictable nature of financial markets, dueling networks allow for better action selection even when state values might not be accurately defined.

### Penalization Factor

The primary objective in the DRL context is to attain the optimal policy. The classic Bellman Q-update formulation is employed to achieve this. However, the goal is to train a more conservative agent. Thus, by introducing an additional constant, negative rewards are manually amplified, making the temporal difference more conservative. Selecting the appropriate value for this constant is integral to the study.

## Numerical Experiment

This section presents a comprehensive numerical experiment to validate the model and algorithms, mirroring the methods of Zejnullahu et al. (2022).

### Dataset

The data incorporates the E-mini S&P 500 continuous futures contract's daily price data from 2010 to 2023, with a focus on the 'Close' column. Excluding weekends and holidays, there are a total of 3,352 data points, with the last 500 days earmarked for out-of-sample assessment.

### MDP and Trading Simulator

All RL problems are primarily formulated as a Markov Decision Process (MDP), with distinct elements:

- **State**: 1-day and 5-day asset return.
- **Action Space**: 0: short, 1: stay-out-of-the-market, 2: long.
- **Reward**: Net portfolio value percentage change, considering immediate returns and transaction costs.
- **State Transition**: Update based on 1-day and 5-day return.

A specialized environment, named StockTradingEnv, derived from Open AI's Gym class, was created to accommodate the MDP.

### Model and Environment Hyper-Parameters

Two primary models, the Normal-DQN and Dueling-DQN, are detailed, showcasing the layered structure of each. The environment's hyperparameters, vital for the training process, are also elaborated upon, encompassing aspects like the number of episodes, learning rate, batch size, and more.

## Conclusion

Deep Reinforcement Learning, particularly the DQN structure, demonstrates significant potential in optimizing trading strategies within stock markets. By integrating innovative structures like the Dueling network and adjusting penalty factors, the research seeks to drive better investment decisions in dynamic trading environments. The methodologies explored set the stage for future research in this exciting confluence of machine learning and finance.

---
normal DQN:  

<img src="https://github.com/cl6530/DLfinalproject/blob/cl6530-patch-1/normalDQN.jpg" alt="normal DQN" width="400"/>  

Dueling DQN:  

<img src="https://github.com/cl6530/DLfinalproject/blob/cl6530-patch-1/duelingDQN.jpg" alt="dueling DQN" width="400"/>  

Action Counts for Different Penalty Value:  

<img src="https://github.com/cl6530/DLfinalproject/blob/cl6530-patch-1/actioncounts.png" alt="Action Counts for Different Penalty Value" width="400"/>  

Protfolio Value for Different Penalties:  

<img src="https://github.com/cl6530/DLfinalproject/blob/cl6530-patch-1/varpen.png" alt="Protfolio Value for Different Penalties" width="400"/>


