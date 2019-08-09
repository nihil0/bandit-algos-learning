
import abc
import numpy as np
import pandas as pd

class Agent(abc.ABC):
    """
    Representation of an agent who tries to optimize interactions with the bandit

    Args:
        bandit (mab.MultiArmedBandit): the bandit being sampled
        horizon (int): Number of periods to optimize over
    """
    def __init__(self, bandit, horizon=1000):
        self.bandit = bandit
        self.horizon = horizon
        self.data = pd.DataFrame({
            'actions': pd.Series([], dtype='int'),
            'rewards': pd.Series([], dtype='float')
            })
        self.state = {}
    
    @abc.abstractmethod
    def initialize_state(self):
        """
        Implements initialization of the agent state
        """
        pass

    @abc.abstractmethod
    def update_state(self):
        """
        Update what agent knows
        """
        pass

    @abc.abstractmethod
    def next_arm(self):
        """ 
        Abstract method to be implemented to decide which arm to pull
        """
        pass

    def run(self):
        """
        Runs the bandit experiment over the horizon
        """
        self.initialize_state()

        for t in range(self.horizon):
            arm_idx = self.next_arm()
            reward = self.bandit.pull(arm_idx)

            # record action and reward
            self.data.loc[t] = [arm_idx, reward]
            self.update_state()

    def report(self):
        """
        Agent stats
        """
        print(f"Total reward: {self.data.rewards.sum()}")


class IntoxicatedAgent(Agent):
    def initialize_state(self):
        self.state = {
            "estimates": np.zeros(self.bandit.n_arms)
        }

    def update_state(self):
        pass
    
    def next_arm(self):
        return np.random.randint(self.bandit.n_arms, size=1)[0]