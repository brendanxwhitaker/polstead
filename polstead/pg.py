""" An implementation of a simple policy gradient agent. """
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions.categorical import Categorical

# import asta.check
from asta import Array, Tensor, typechecked, dims

# pylint: disable=too-few-public-methods

OBS_SHAPE = dims.OBS_SHAPE
NUM_ACTIONS = dims.NUM_ACTIONS


class Trajectory:
    """ An object to hold a trajectory composed of numpy arrays. """

    def __init__(self) -> None:
        self.obs: List[Array[float, OBS_SHAPE]] = []
        self.acts: List[int] = []
        self.rews: List[float] = []
        self.weights: List[float] = []
        self.ep_rets: List[float] = []
        self.ep_lens: List[int] = []

        self.ep_start_idx = 0
        self.weights_idx = 0
        self.batch_idx = 0

        self.ema_ret = 0.0
        self.ema_alpha = 0.9

    def add(self, ob: Array[float, OBS_SHAPE], act: int, rew: float,) -> None:
        """ Add an observation, action, and reward to storage. """
        self.obs.append(ob)
        self.acts.append(act)
        self.rews.append(rew)

    @typechecked
    def get(
        self,
    ) -> Tuple[
        Tensor[float, (-1, *OBS_SHAPE)], Tensor[int, -1, NUM_ACTIONS], Tensor[float, -1]
    ]:
        """ Return observations, actions, and weights for a batch. """
        obs_batch = self.obs[self.batch_idx :]
        acts_batch = self.acts[self.batch_idx :]
        weights_batch = self.weights[self.weights_idx :]

        self.batch_idx = len(self.obs)
        self.weights_idx = len(self.weights)

        obs_t = torch.Tensor(obs_batch)
        acts_t = torch.Tensor(acts_batch)
        weights_t = torch.Tensor(weights_batch)

        return obs_t, acts_t, weights_t

    def finish(self) -> float:
        """ Compute and save weights for a (possibly partially completed) episode. """
        ep_ret = sum(self.rews[self.ep_start_idx :])
        ep_len = len(self.rews) - self.ep_start_idx
        self.weights.extend([ep_ret] * ep_len)
        self.ep_start_idx = len(self.rews)
        self.ep_rets.append(ep_ret)
        self.ep_lens.append(ep_len)

        self.ema_ret = self.ema_alpha * self.ema_ret + (1 - self.ema_alpha) * ep_ret
        return self.ema_ret

    def __len__(self) -> int:
        """ Returns length of the buffer. """
        return len(self.obs)


class Policy(nn.Module):
    """ The parameterized action-value and state-value functions. """

    def __init__(self, observation_size: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.num_actions: int = num_actions
        self._policy = nn.Sequential(
            nn.Linear(observation_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
            nn.Identity(),
        )

    def forward(
        self, ob: Tensor[float, (-1, *OBS_SHAPE)]
    ) -> Tensor[float, -1, NUM_ACTIONS]:
        r""" Actor which implements the policy $\pi_{\theta}$. """
        logits = self._policy(ob)
        return logits


def get_action(policy: nn.Module, ob: Tensor[float, OBS_SHAPE]) -> Tensor[float, ()]:
    """ Sample action from policy. """
    obs: Tensor[float, (1, *OBS_SHAPE)] = ob.unsqueeze(0)
    distribution = get_policy_distribution(policy, obs)
    action = distribution.sample()
    return action


def get_policy_distribution(
    policy: nn.Module, obs: Tensor[float, (-1, *OBS_SHAPE)]
) -> Categorical:
    """ Computes the policy distribution for the state given by ``ob``. """
    logits: Tensor[float, NUM_ACTIONS] = policy(obs)
    distribution = Categorical(logits=logits)
    return distribution


@typechecked
def compute_loss(
    policy: nn.Module,
    obs: Tensor[float, (-1, *OBS_SHAPE)],
    acts: Tensor[float, -1],
    weights: Tensor[float, -1],
) -> Tensor[float, ()]:
    """ See ``LOSS.txt``. """
    # TODO: It would be useful to have variable shape elements which have to
    # all be the same within a single function call, but can be anything.
    assert len(acts) == len(weights) == obs.shape[0]
    policy_distribution = get_policy_distribution(policy, obs)
    logp = policy_distribution.log_prob(acts)
    loss = -(logp * weights).mean()
    return loss
