from __future__ import division
import numpy as np

from rl.util import *


class Policy(object):
    """Abstract base class for all implemented policies.

    Each policy helps with selection of action to take on an environment.

    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:

    - `select_action`

    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy

        # Returns
            Configuration as dict
        """
        return {}

class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1, linear_decrease = 0, step_to_decrease = 0):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.linear_decrease = linear_decrease
        self.step_to_decrease = step_to_decrease

    def select_action(self, q_values, step = 0):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        if (self.linear_decrease > 0):
            if step % self.step_to_decrease == 0 and step > 0:
                if self.eps >= self.linear_decrease:
                    self.eps -= self.linear_decrease
                    print(self.eps)
                else:
                    self.eps = 0
                    print(self.eps)
        exploit = False
        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            exploit = True
            action = np.argmax(q_values)
        return action, exploit
    
    def select_linear_desceasing_action(self, q_values, step):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        
        if step % self.step_to_decrease == 0:
            if self.eps >= self.linear_decrease:
                self.eps -= self.linear_decrease
            else:
                self.eps = 0

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
    
class DynamicExpForBDQL(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, baseline=.1):
        super(DynamicExpForBDQL, self).__init__()
        self.baseline = baseline

    def select_action(self, q_values, action):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        exploit = False
        if np.random.uniform() < self.baseline - action:
            action = np.random.randint(0, nb_actions)
        else:
            exploit = True
            action = np.argmax(q_values)
        return action, exploit

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(DynamicExpForBDQL, self).get_config()
        config['baseline'] = self.baseline
        return config

class EpsGreedyFuzzyPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1, linear_decrease = 0, step_to_decrease = 0):
        super(EpsGreedyFuzzyPolicy, self).__init__()
        self.eps = eps
        self.linear_decrease = linear_decrease
        self.step_to_decrease = step_to_decrease

    def select_action(self, q_values, step = 0):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        exploration = False
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        if (self.linear_decrease > 0):
            if step % self.step_to_decrease == 0 and step > 0:
                if self.eps >= self.linear_decrease:
                    self.eps -= self.linear_decrease
                    print(self.eps)
                else:
                    self.eps = 0
                    print(self.eps)
        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
            exploration = True
        else:
            action = np.argmax(q_values)
        return action, exploration
    
    def select_linear_desceasing_action(self, q_values, step):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        
        if step % self.step_to_decrease == 0:
            if self.eps >= self.linear_decrease:
                self.eps -= self.linear_decrease
            else:
                self.eps = 0

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyFuzzyPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyFuzzyPolicy, self).get_config()
        config['eps'] = self.eps
        return config
    
class BoltzmannQPolicy(Policy):
    """Implement the Boltzmann Q Policy
    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values, step = 0):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action, 1
    
class MaxBoltzmannQPolicy(Policy):
    """
    A combination of the eps-greedy and Boltzman q-policy.
    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)
    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        super(MaxBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values, step = 0):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]
        
        exploration = False
        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
            exploration = True
        else:
            action = np.argmax(q_values)
        return action, exploration
    
class BoltzmannGumbelQPolicy(Policy):
    """Implements Boltzmann-Gumbel exploration (BGE) adapted for Q learning
    based on the paper Boltzmann Exploration Done Right
    (https://arxiv.org/pdf/1705.10257.pdf).
    BGE is invariant with respect to the mean of the rewards but not their
    variance. The parameter C, which defaults to 1, can be used to correct for
    this, and should be set to the least upper bound on the standard deviation
    of the rewards.
    BGE is only available for training, not testing. For testing purposes, you
    can achieve approximately the same result as BGE after training for N steps
    on K actions with parameter C by using the BoltzmannQPolicy and setting
    tau = C/sqrt(N/K)."""

    def __init__(self, C=1.0):
        assert C > 0, "BoltzmannGumbelQPolicy C parameter must be > 0, not " + repr(C)
        super(BoltzmannGumbelQPolicy, self).__init__()
        self.C = C
        self.action_counts = None

    def select_action(self, q_values, step = 0):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        # We can't use BGE during testing, since we don't have access to the
        # action_counts at the end of training.
        assert self.agent.training, "BoltzmannGumbelQPolicy should only be used for training, not testing"

        assert q_values.ndim == 1, q_values.ndim
        q_values = q_values.astype('float64')

        # If we are starting training, we should reset the action_counts.
        # Otherwise, action_counts should already be initialized, since we
        # always do so when we begin training.
        if self.agent.step == 0:
            self.action_counts = np.ones(q_values.shape)
        assert self.action_counts is not None, self.agent.step
        assert self.action_counts.shape == q_values.shape, (self.action_counts.shape, q_values.shape)

        beta = self.C/np.sqrt(self.action_counts)
        Z = np.random.gumbel(size=q_values.shape)

        perturbation = beta * Z
        perturbed_q_values = q_values + perturbation
        action = np.argmax(perturbed_q_values)

        self.action_counts[action] += 1
        return action, 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    