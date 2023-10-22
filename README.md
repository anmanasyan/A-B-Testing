# A/B Testing with Epsilon Greedy and Thompson Sampling

## Project Overview
This assignment explores the implementation and comparison of two popular multi-armed bandit algorithms: Epsilon Greedy and Thompson Sampling. Multi-armed bandit problems involve a trade-off between exploration (trying different arms) and exploitation (choosing the best-known arm). These algorithms are commonly used in scenarios like online advertising, recommendation systems, and more.


1. **Create a Bandit Class**: Implement a class to represent the bandit arms, which are used in both Epsilon Greedy and Thompson Sampling algorithms.

2. **Create EpsilonGreedy() and ThompsonSampling() Classes and Methods**: Implement classes and methods for the Epsilon Greedy and Thompson Sampling algorithms, both of which are inherited from the Bandit class.

   - **Epsilon Greedy**:
     - Implement an Epsilon Greedy class that decays epsilon by 1/t.
     - Design the experiment using the Epsilon Greedy algorithm.

   - **Thompson Sampling**:
     - Implement a Thompson Sampling class with known precision.
     - Design the experiment using the Thompson Sampling algorithm.

3. **Report**:
   - **Visualize the Learning Process**: Implement a method `plot1()` to visualize the learning process for each algorithm.
   - **Visualize Cumulative Rewards**: Compare and visualize cumulative rewards from Epsilon Greedy and Thompson Sampling.
   - **Store the Rewards in a CSV File**: Create CSV files that store experiment data, including Bandit, Reward, and Algorithm information.
   - **Print Cumulative Reward**: Display the cumulative reward achieved by the algorithms.
   - **Print Cumulative Regret**: Display the cumulative regret of each the algorithms.


## Assignment Structure

The assignment code is organized as follows:


- `bandit.py`: Defines the abstract class for bandit arms, Epsilon Greedy and Thompson Sampling algorithm implementations and functions for visualizing experiment results.
- `Report.ipynb`: A report of experiment results. 
