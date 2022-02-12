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