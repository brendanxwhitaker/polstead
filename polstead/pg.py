""" An implementation of a simple policy gradient agent. """
from typing import List

import torch
from torch import nn
from torch.distributions.categorical import Categorical

from asta import Array, Tensor, typechecked, check, dims

# pylint: disable=too-few-public-methods

check.on()
OBS_SHAPE = dims.OBS_SHAPE
ACT_SHAPE = dims.ACT_SHAPE
NUM_ACTIONS = dims.NUM_ACTIONS


class Trajectories:
    """ An object to hold trajectories of numpy arrays. """

    def __init__(self) -> None:
        self.obs: List[Array[float, OBS_SHAPE]] = []
        self.actions: List[Array[float, ACT_SHAPE]] = []
        self.rewards: List[float] = []
        self.weights: List[float] = []

    @typechecked
    def add(
        self,
        ob: Array[float, OBS_SHAPE],
        action: Array[float, ACT_SHAPE],
        reward: float,
    ) -> None:
        """ Add an observation, action, and reward to storage. """
        self.obs.append(ob)
        self.actions.append(action)
        self.rewards.append(reward)


class ActorCritic:
    """ The parameterized action-value and state-value functions. """

    def __init__(self, observation_size: int, num_actions: int, hidden_size: int):
        self.num_actions: int = num_actions
        self._policy = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1),
        )

    @typechecked
    def policy(self, ob: Tensor[float, OBS_SHAPE]) -> Tensor[float, NUM_ACTIONS]:
        r""" Actor which implements the policy $\pi_{\theta}$. """
        logits = self._policy(ob)
        return logits

    @typechecked
    def get_policy(self, ob: Tensor[float, OBS_SHAPE]) -> Categorical:
        """ Computes the policy distribution for the state given by ``ob``. """
        logits: torch.Tensor = self.policy(ob)
        policy = Categorical(logits=logits)
        return policy


@typechecked
def compute_loss(
    obs: Tensor[float, (-1, *OBS_SHAPE)],
    actions: Tensor[float, (-1, *ACT_SHAPE)],
    weights: Tensor[float, -1],
) -> Tensor[float, ()]:
    r"""
    The loss function is derived from the formula for the gradient estimator.
    Let $D$ denote the set of all trajectories $\tau$, and let $T$ be the
    number of environment steps in a given trajectory. The index $t$ is the
    current timestep in the current trajectory $\tau$, and the log probability
    of an action $a_t$ given that action is taken from a state $s_t$ under our
    policy $\pi_{\theta}$ is given by $\log \pi_{\theta}(a_t|s_t)$. The
    trajectory return $R(\tau)$ is given by summing the rewards from every
    timestep of the trajectory $\tau$. Note that trajectory lengths may vary,
    and thus may have multiple trajectories in a single policy update.

    $\tau$ : a trajectory.
    $D$ : set of all trajectories $\tau$.
    #T$ : number of environment steps in a given trajectory.
    $t$ : index of current timestep in a trajectory.
    $a_t$ : action at timestep $t$.
    $s_t$ : state at timestep $t$.
    $\theta$ : parameters/weights of the policy.
    $\pi_{\theta}$ : the policy parameterized by $\theta$.
    $R(\tau)$ : the return given by summing all rewards in a trajectory.

    In this case, the gradient estimator is given by:
    $$
        \hat(g)
        =
            \frac{1}{|D|} \sum_{\tau \in D} \sum_{t = 0}^{T}
            \grad_{\theta} \log \pi_{\theta}(a_t|s_t) R(\tau).
    $$
    """
    raise NotImplementedError
