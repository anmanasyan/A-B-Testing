from abc import ABC, abstractmethod
from logs import *
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
    
    """

    This class is for initializing bandit arms.

    Parameters:
    p (float): The true win rate of the arm.

    Attributes:
    p (float): The true win rate of the arm.
    p_estimate (float): The estimated win rate.
    N (int): The number of pulls.

    Methods:
    pull(): Pull the arm and return the sampled reward.
    update(): Update the estimated win rate with a new reward value.
    experiment(): Run the experiment..
    report(): Generate a report with statistics about the experiment.
    """

    #@abstractmethod
    def __init__(self, p):
        """
        Initialize the EpsilonGreedy arm.

        Parameters:
        p (float): The win rate of the arm.
        """
        self.p = p
        self.p_estimate = 0 #estimate of average reward
        self.N = 0
        self.r_estimate = 0 #estimate of average regret

    #@abstractmethod
    def __repr__(self):
        
        """
        Return a string representation of the arm.

        Returns:
        str: A string describing the arm.
        """
        return f'An Arm with {self.p} Win Rate'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    #@abstractmethod
    def report(self, N, results, algorithm = "Epsilon Greedy"):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        
        """
        Generate a report with statistics about the experiment.

        Parameters:
        N (int): The number of trials in the experiment.
        results (tuple): A tuple of experiment results.
        algorithm (string): Name of the algorithm used.

        Prints:
        Statistics and saves data to CSV files.
        """
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal = results 
        else:
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward = results 
        
        # Save experiment data to a CSV file
        data_df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        data_df.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save Final Results to a CSV file
        data_df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })


        data_df1.to_csv(f'{algorithm}_Final.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
            print("--------------------------------------------------")
        
        
        print(f"Cumulative Reward : {sum(reward)}")
        
        print(" ")
        
        print(f"Cumulative Regret : {cumulative_regret[-1]}")
              
        print(" ")
        
        if algorithm == 'EpsilonGreedy':                            
            print(f"Percent suboptimal : {round((float(count_suboptimal) / N), 4)}")


#--------------------------------------#

class Visualization:
    def plot1(self, N, results, algorithm='EpsilonGreedy'):
        """
        Visualize the performance of the algorithm in terms of cumulative average reward.
        
        Parameters:
        N (int): The number of trials in the experiment.
        results (tuple): A tuple of experiment results.
        algorithm (str): Name of the algorithm used, defaults to 'EpsilonGreedy'.

        Prints:
        Linear and log scale plots of cumulative average reward and optimal reward.
        """
        
        #Retrieving the bandits and Cumulative Average Reward
        
        cumulative_reward_average = results[0]
        bandits = results[3]
        
        ## LINEAR SCALE
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        ## LOG SCALE
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def plot2(self, results_eg, results_ts):
        """
        Compare Epsilon-Greedy and Thompson Sampling in terms of cumulative rewards and regrets.

        Parameters:
        results_eg (tuple): A tuple of experiment results for Epsilon-Greedy.
        results_ts (tuple): A tuple of experiment results for Thompson Sampling.

        Prints:
        Plots comparing cumulative rewards and cumulative regrets for Epsilon-Greedy and Thompson Sampling.
        """
        # Retrieving Cumulative reward and regret
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        ## Cumulative Reward
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        ## Cumulative Regret
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):
    
    """
    Epsilon-Greedy multi-armed bandit algorithm.

    This class represents a multi-armed bandit problem solver using the Epsilon-Greedy algorithm.

    Parameters:
    p (float): The true win rate of the arm.

    Attributes:
    p (float): The true win rate of the arm.
    p_estimate (float): The estimated win rate.
    N (int): The number of pulls.

    Methods:
    pull(): Pull the arm and return the sampled reward.
    update(x): Update the estimated win rate with a new reward value.
    experiment(BANDIT_REWARDS, N, t=1): Run the experiment..
    report(N, results): Generate a report with statistics about the experiment.
    """

    def __init__(self, p):
        
        """
        Initialize the EpsilonGreedy arm.

        Parameters:
        p (float): The win rate of the arm.
        """
        super().__init__(p)

    def pull(self):
        
        """
        Pull the arm and return the sampled reward.

        Returns:
        float: The sampled reward from the arm.
        """
        return np.random.randn() + self.p

    def update(self, x):
        
        """
        Update the estimated win rate with a new reward value.

        Parameters:
        x (float): The observed reward.
        """
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate


    def experiment(self, BANDIT_REWARDS, N, t = 1):
        
        """
        Run the experiment using Epsilon Greedy Algorithm.

        Parameters:
        BANDIT_REWARDS (list): List of true win rates for each arm.
        N (int): The number of Trials.
        t (int): Time step, defaults to 1.

        Returns:
        tuple: A tuple containing average cumulative reward, cumulative reward,  cumulative regret, updated bandits, chosen bandit at each trial, reward at each trial, count of suboptimal pulls
        """
        
        #Initializing Bandits
        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)  
        count_suboptimal = 0
        EPS = 1/t

        #Keep Track of Which Bandit was chosen and the resulting reward 
        reward = np.empty(N)
        chosen_bandit = np.empty(N)


        for i in range(N):
            p = np.random.random()
            
            if p < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            
            bandits[j].update(x)
    

            if j != true_best:
                count_suboptimal += 1
            
            reward[i] = x
            chosen_bandit[i] = j
            
            t+=1
            EPS = 1/t

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        for i in range(len(reward)):
            cumulative_regret[i] = N*max(means) - cumulative_reward[i]

        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal



class ThompsonSampling(Bandit):
    """
    ThompsonSampling is a class for implementing the Thompson Sampling algorithm for multi-armed bandit problems.

    Attributes:
    - p (float): The win rate of the bandit arm.
    - lambda_ (float): A parameter for the Bayesian prior.
    - tau (float): A parameter for the Bayesian prior.
    - N (int): The number of times the bandit arm has been pulled.
    - p_estimate (float): The estimated win rate of the bandit arm.

    Methods:
    - pull(): Pull the bandit arm and return the observed reward.
    - sample(): Sample from the posterior distribution of the bandit arm's win rate.
    - update(x): Update the bandit arm's parameters and estimated win rate based on the observed reward.
    - plot(bandits, trial): Plot the probability distribution of the bandit arm's win rate after a given number of trials.
    - experiment(BANDIT_REWARDS, N): Run an experiment to estimate cumulative reward and regret for Thompson Sampling.

    """
    
    def __init__(self, p):
        """
        Initialize a ThompsonSampling bandit arm with the given win rate.

        Parameters:
        p (float): The win rate of the bandit arm.
        """
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1


    def pull(self):
        """
        Pull the bandit arm and return the observed reward.

        Returns:
        float: The observed reward from the bandit arm.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.p
    
    def sample(self):
        """
        Sample from the posterior distribution of the bandit arm's win rate.

        Returns:
        float: The sampled win rate from the posterior distribution.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
    
    def update(self, x):
        """
        Update the bandit arm's parameters and estimated win rate based on the observed reward.

        Parameters:
        x (float): The observed reward.
        """
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
        
    def plot(self, bandits, trial):
        
        """
        Plot the probability distribution of the bandit arm's win rate after a given number of trials.

        Parameters:
        bandits (list): List of ThompsonSampling bandit arms.
        trial (int): The number of trials or rounds.

        Displays a plot of the probability distribution of the bandit arm's win rate.

        """
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.p:.4f}, num plays: {b.N}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, BANDIT_REWARDS, N):
        """
        Run an experiment to estimate cumulative reward and regret for Thompson Sampling.

        Parameters:
        BANDIT_REWARDS (list): List of true win rates for each bandit arm.
        N (int): The number of rounds in the experiment.

        Returns:
        tuple: A tuple containing cumulative reward statistics, bandits, and other information.

        """
        
        bandits = [ThompsonSampling(m) for m in BANDIT_REWARDS]

        sample_points = [5, 20, 50,100,200,500,1000,1999, 5000,10000, 19999]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)
        
        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])

            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()

            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        
        for i in range(len(reward)):
            cumulative_regret[i] = N*max([b.p for b in bandits]) - cumulative_reward[i]


        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward 
 



def comparison(N, results_eg, results_ts):
    # think of a way to compare the performances of the two algorithms VISUALLY 
    
    """
    Compare performance of Epsilon Greedy and Thompson Sampling algorithms in terms of cumulative average reward.

    Parameters:
    N (int): The number of trials in the experiment.
    results_eg (tuple): A tuple of Epsilon Greedy experiment results.
    results_ts (tuple): A tuple of Thompson Sampling experiment results.
    
    Prints:
    Linear and log scale plots of cumulative average reward and optimal reward of both algorithms.
    """

    #Retrieving the bandits and Cumulative Average Reward

    cumulative_reward_average_eg = results_eg[0]
    cumulative_reward_average_ts = results_ts[0]
    bandits_eg = results_eg[3]
    reward_eg = results_eg[5]
    reward_ts = results_ts[5]
    regret_eg = results_eg[2][-1]
    regret_ts = results_ts[2][-1]

    
    print(f"Total Reward Epsilon Greedy : {sum(reward_eg)}")
    print(f"Total Reward Thomspon Sampling : {sum(reward_ts)}")
        
    print(" ")
        
    print(f"Total Regret Epsilon Greedy : {regret_eg}")
    print(f"Total Regret Thomspon Sampling : {regret_ts}")
        

    plt.figure(figsize=(12, 5))

    ## LINEAR SCALE
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")


    ## LOG SCALE
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")
    
    
    plt.tight_layout()
    plt.show()
    

    