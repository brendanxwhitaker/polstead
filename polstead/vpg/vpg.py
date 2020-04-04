""" Functions for a simple policy gradient. """
from typing import List, Tuple

import scipy
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from asta import dims, shapes, symbols, Array, Tensor, typechecked


X = symbols.X
OB = shapes.OB
N_ACTS = dims.N_ACTS


class Actor(nn.Module):
    """ The parameterized policy, which estimates the action-value function. """

    def __init__(self, observation_size: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self._policy = nn.Sequential(
            nn.Linear(observation_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
            nn.Identity(),
        )

    @typechecked
    def forward(self, ob: Tensor[float, (..., *OB)]) -> Tensor[float, ..., N_ACTS]:
        r""" Actor which implements the policy $\pi_{\theta}$. """
        logits = self._policy(ob)
        return logits


class Critic(nn.Module):
    """ A parametrized estimator of the state-value function. """

    def __init__(self, observation_size: int, hidden_dim: int):
        super().__init__()

        self._critic = nn.Sequential(
            nn.Linear(observation_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Identity(),
        )

    @typechecked
    def forward(self, ob: Tensor[float, (..., *OB)]) -> Tensor[float, ..., 1]:
        r""" Just a wrapper around the forward method to provide typechecking. """
        value = self._critic(ob)
        return value


class ActorCritic:
    """ The parameterized policy and state-value functions. """

    def __init__(self, observation_size: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.pi = Actor(observation_size, num_actions, hidden_dim)
        self.v = Critic(observation_size, hidden_dim)


@typechecked
def get_distribution(ac: ActorCritic, ob: Tensor[float, OB]) -> Categorical:
    """ Computes the policy distribution for the state given by ``ob``. """
    logits = ac.pi(ob)
    return Categorical(logits=logits)


@typechecked
def get_batch_distribution(
    ac: ActorCritic, obs: Tensor[float, (-1, *OB)]
) -> Categorical:
    """ Computes the policy distribution for a batch of observations. """
    logits = ac.pi(obs)
    return Categorical(logits=logits)


@typechecked
def get_action(ac: ActorCritic, obs: Tensor[float, OB]) -> int:
    """ Sample action from policy. """
    distribution = get_distribution(policy, obs)
    sample: Tensor[int, ()] = distribution.sample()
    act: int = sample.item()
    return act


@typechecked
def compute_loss(
    ac: ActorCritic,
    obs: Tensor[float, (X, *OB)],
    act: Tensor[int, X],
    weights: Tensor[float, X],
) -> Tensor[float, ()]:
    """ Computes the loss function. """
    logp = get_batch_distribution(ac, obs).log_prob(act)
    return -(logp * weights).mean()


@typechecked
def reward_to_go(rews: List[float]) -> List[float]:
    """ Weight function which only uses sum of rewards after an action is taken. """
    n = len(rews)
    rtgs = np.zeros((n,))
    for i in reversed(range(n)):
        # The subsequent rewards are the sum of ``rews[i + 1:]``.
        if i + 1 < n:
            subsequent_rews = rtgs[i + 1]
        else:
            subsequent_rews = 0
        rtgs[i] = rews[i] + subsequent_rews
    return list(rtgs)


def discounted_cumulative_sum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    
    input:
    vector x,
    [x0,
    x1,
    x2]
    
    output:
    [x0 + discount * x1 + discount^2 * x2,
    x1 + discount * x2,
    x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def fast_reward_to_go(rews: List[float], gamma: float) -> List[float]:
    """ Weight function which only uses sum of rewards after an action is taken. """
    return discounted_cumulative_sum(rews, gamma)


@typechecked
def uniform_weights(rews: List[float]) -> List[float]:
    """ Weight function where the weights are all just the episode return. """
    ep_ret = sum(rews)
    ep_len = len(rews)
    weights = [ep_ret] * ep_len
    return weights


class RolloutStorage:
    """ An object to hold trajectories. """

    def __init__(self) -> None:
        self.obs: List[Array[float, OB]] = []
        self.acts: List[int] = []
        self.rews: List[float] = []
        self.weights: List[float] = []
        self.ep_rets: List[float] = []
        self.ep_lens: List[int] = []

        self.rets: List[float] = []
        self.lens: List[int] = []

    def add(self, ob: Array[float, OB], act: int, rew: float) -> None:
        """ Add an observation, action, and reward to storage. """
        self.obs.append(ob)
        self.acts.append(act)
        self.rews.append(rew)

    @typechecked
    def get(
        self,
    ) -> Tuple[
        Tensor[float, (X, *OB)], Tensor[int, X], Tensor[float, X],
    ]:
        """ Return observations, actions, and weights for a batch. """
        assert len(self.obs) == len(self.acts) == len(self.weights)

        # Cast buffer storage to tensors.
        obs_t = torch.Tensor(self.obs)
        acts_t = torch.Tensor(self.acts).int()
        weights_t = torch.Tensor(self.weights)

        # Reset.
        self.obs = []
        self.acts = []
        self.weights = []
        self.rets = []
        self.lens = []

        return obs_t, acts_t, weights_t

    def stats(self) -> Tuple[float, float]:
        """ Return the current mean episode return and length. """
        mean_ret = np.mean(self.rets)
        mean_len = np.mean(self.lens)
        return mean_ret, mean_len


def finish(
    rews: List[float], vals: List[float], last_val: int
) -> Tuple[List[float], List[float]]:
    """
    Compute and save weights for a (possibly partially completed) episode.
    """
    n = len(rews)
    rews.append(last_val)
    vals.append(last_val)

    # See the GAE paper, definition of $\delta$ is between equations (9) and (10).
    # $\delta_{t}^{V} = r_t + \gamma V(s_{t + 1}) - V(s_t)$.
    # We compute deltas for 0 <= t <= n - 1.
    deltas = rews[:-1] + ox.gamma * vals[1:] - vals[:-1]
    advantages = discounted_cumulative_sum(deltas, ox.gamma * ox.lam)
    returns = fast_reward_to_go(rews[:-1], ox.gamma)

    assert len(deltas) == len(advantages) == len(returns) == n

    return advantages, returns
