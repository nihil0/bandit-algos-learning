import abc
import numpy as np
import pandas as pd

class MultiArmedBandit(object):
    """
    Representation of the fictitious Multi Armed Bandit machine

    Args:
        n_arms (int): number of arms in bandit
        true_probs (List[Callable[[], float]]): a list of `n_arms` callables which return a reward  
    """
    def __init__(self, n_arms, true_probs):
        self.n_arms = n_arms
        self.true_probs = true_probs

    def pull(self, arm_idx):
        """
        Simulates a pull on the bandit arm

        Args:
            arm_idx (int): number between `0` and `n_arms-1`
        """
        return self.true_probs[arm_idx]()


class BernoulliBandit(MultiArmedBandit):
    """
    A MAB with Bernoulli payouts

    Args:
        n_arms (int): number of arms in bandit
        true_probs (List[float]): A list of payout probabilities 
    """

    def __init__(self, n_arms, true_probs):
        prob_samplers = [
            lambda m=m: np.random.binomial(1,m,1)[0] for m in true_probs
        ]
        super().__init__(n_arms, prob_samplers)