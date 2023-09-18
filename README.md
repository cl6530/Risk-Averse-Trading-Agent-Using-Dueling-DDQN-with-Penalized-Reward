# DL_final_project
# Risk-Averse Trading Agent Using Dueling DDQN with Penalized Reward

**Authors**: Jalil Douglas, Dailin Ji, Chenxi Liu  
**Affiliation**: New York University, Tandon School of Engineering

---

## I. Introduction

Machine Learning has been widely applied in modern finance, with one of the most popular areas being model-enabled autonomous trading. In this approach, models are trained on historical data to predict future prices and make trading decisions, without the need for human intervention. Among various types of models, Reinforcement Learning (RL) has a unique advantage due to its ability to make sequential decisions. However, RL agents often struggle to find a risk-neutral strategy in the high volatility of financial markets. In this paper, we experiment a heuristics method to address the issue. Specifically, we adjust the reward function to penalize investment actions with negative returns and high risk. By tuning the penalization factor, we train investment agents that are more conservative in making investment decisions. The effectiveness of our proposed method is validated through extensive experiments, demonstrating its potential in improving the performance of RL-based autonomous trading systems. The purpose of this study is to demonstrate the ability to interfere DQN agent by adjusting rewards, instead of to make a profitable trading strategy. The code are available at [GitHub Repository](https://github.com/cl6530/DLfinalproject).

The structure of the paper is organized as follows: The subsequent section provides a concise overview of the existing literature, exploring the various applications of reinforcement learning (RL) in the field of trading. Section III primarily focuses on presenting the problem formulation and our methodology, encompassing the dueling network structure and the penalization factor. In Section IV, we demonstrate the effectiveness of trading simulators in facilitating our experiments, and present the numerical results obtained from our research. Finally, we conclude the project by summarizing our findings and suggesting potential future directions for further exploration.

## II. Literature Review

### Reinforcement Learning and DQN

Reinforcement learning (RL) is a subdomain of machine learning where an agent learns to make decisions by interacting with an environment to achieve a goal. As being reviewed by Li (2018), The agent selects actions based on its current state. The environment responds by providing a reward signal.The agent then chooses actions based on its present condition. Learning an ideal strategy that maximizes the cumulative reward over time is the goal of RL. The RL problem is formalized using a Markov Decision Process (MDP), which is characterized by states, actions, rewards, and transition probabilities that represent the dynamics of the environment.

Deep Q-Networks (DQN), introduced by Mnih et al. (2015), combine traditional Q-learning with deep neural networks. This combination makes it possible to learn more complex patterns and decision-making criterias, which will be particularly useful when dealing with high-dimensional state spaces and large-scale action sets. DQNs use a neural network to approximate the Q-function, which estimates the future rewards expectations of taking a specific action in a given state. By continuously updating the Q-function approximation, the agent learns to select actions that aim to maximize the cumulative rewards.

### Reinforcement Learning in Trading

As concluded by Pricope (2021), being inspired by the excellent performance of Deep Reinforcement Learning (DRL) in complex games, DRL has shown great potential in stock trading that it could compete with experienced traders.

Under a financial context, RL may be used to simulate and optimize different elements of trading and investing, including portfolio management, algorithmic trading, and risk management. For instance, the states may be market data, financial indicators, or other pertinent information, and the agent could stand in for a trader or an algorithm. The actions could involve buying, selling, or holding, and the rewards can be based on the profit or loss resulting from these actions.

[... continues ...]

