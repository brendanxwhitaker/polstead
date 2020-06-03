""" Vanilla policy gradient classes and functions. """
from abc import abstractmethod, ABC
from typing import Tuple

import gym
import numpy as np

import torch
from torch import nn

from asta import Array, Tensor, shapes, dims, symbols, typechecked
from oxentiel import Oxentiel

from polstead.losses import get_distribution, get_value_loss, get_policy_loss
from polstead.rollouts import RolloutStorage

# pylint: disable=too-few-public-methods

T = symbols.T


@typechecked
class ActorCritic(nn.Module, ABC):
    r""" An actor-critic module with GAE and rewards-to-go. """

    def __init__(
        self,
        ox: Oxentiel,
        env: gym.Env,
        actor: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()
        self.i = 0
        self.ox = ox
        self.env = env
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.actor = actor
        self.critic = critic

        self.ob: Array[float, shapes.OB]
        self.act: Array[int, ()]
        self.val: Array[float, ()]
        self.rollouts = RolloutStorage(ox.batch_size, env.observation_space.shape)

    @abstractmethod
    def adv_fn(
        self,
        gamma: float,
        lam: float,
        rews: Array[float],
        vals: Array[float],
        last_val: float,
    ) -> Array[float]:
        """ The advantage function. """
        raise NotImplementedError

    @abstractmethod
    def tgt_fn(self, gamma: float, rews: Array[float]) -> Array[float]:
        """ The value prediction target function. """
        raise NotImplementedError

    def get_distribution(
        self, ob: Tensor[float, shapes.OB]
    ) -> torch.distributions.Distribution:
        """ Returns the distribution class for the action space. """
        space = self.env.action_space
        logits = self.actor(ob)
        if isinstance(space, gym.spaces.Discrete):
            distribution = torch.distributions.categorical.Categorical(logits=logits)
        elif isinstance(space, gym.spaces.Box):
            stddevs = self.distribution_parameters["stddevs"]
            distribution = torch.distributions.normal.Normal(logits, stddevs)
        elif isinstance(space, gym.spaces.MultiBinary):
            distribution = torch.distributions.binomial.Binomial(1, logits=logits)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            pass

    def forward(
        self, ob: Array[float, shapes.OB], rew: float, done: bool
    ) -> Array[int, ()]:
        """ Returns an observation given an action, learning as necessary. """
        # Record data from the previous timestep.
        if self.i > 0:
            self.rollouts.add(self.ob, self.act, self.val, rew)

        # If we reached a terminal state, or we completed a batch.
        if done or self.rollouts.batch_len == self.ox.batch_size:

            # ==============================================
            # STEP 1: Compute advantages and critic targets.
            # ==============================================

            # Get episode length.
            dims.EP_LEN = self.rollouts.ep_len

            # Types and shapes.
            vals: Array[float, dims.EP_LEN]
            rews: Array[float, dims.EP_LEN]
            advs: Array[float, dims.EP_LEN]
            tgts: Array[float, dims.EP_LEN]

            # Retrieve values and rewards for the current episode.
            vals, rews = self.rollouts.get_episode_values_and_rewards()

            # The last value should be zero if this is the end of an episode.
            last_val: float = 0.0 if done else vals[-1]

            # Compute advantages and value targets.
            advs = self.adv_fn(self.ox.lam, self.ox.gamma, rews, vals, last_val)
            tgts = self.tgt_fn(self.ox.gamma, rews)

            # Record the episode length and return.
            if done:
                self.rollouts.lens.append(dims.EP_LEN)
                self.rollouts.rets.append(np.sum(rews))

            # =====================================================================
            # Step 2: Reset vals and rews in buffer and record computed quantities.
            # =====================================================================

            # Reset the values and rewards arrays to zeroes.
            self.rollouts.vals[:] = 0
            self.rollouts.rews[:] = 0

            # Get a pointer to the start of the episode we just finished.
            j = self.rollouts.ep_start
            assert j + dims.EP_LEN <= self.ox.batch_size

            # Append advantages and rewards-to-go to arrays in rollouts.
            self.rollouts.advs[j : j + dims.EP_LEN] = advs
            self.rollouts.tgts[j : j + dims.EP_LEN] = tgts

            # Set the new episode starting point, and set episode length to zero.
            self.rollouts.ep_start = j + dims.EP_LEN
            self.rollouts.ep_len = 0

        # If we completed a batch.
        if self.rollouts.batch_len == self.ox.batch_size:

            # Types and shapes.
            obs: Tensor[float, (self.ox.batch_size, *shapes.OB)]
            acts: Tensor[int, (self.ox.batch_size)]

            # Get batch data from the buffer.
            obs, acts, advs, tgts = self.rollouts.get_batch()

            # Run a backward pass on the policy (actor) and value function (critic).
            self.optimizer.zero_grad()
            policy_loss = get_policy_loss(self.actor, obs, acts, advs)
            value_loss = get_value_loss(self.critic, obs, tgts)
            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Reset pointers.
            self.rollouts.batch_len = 0
            self.rollouts.ep_start = 0

            self.rollouts.rets = []
            self.rollouts.lens = []

        # Sample an action from the policy and estimate the value of current state.
        self.act, self.val = get_action(self.actor, self.critic, ob)

        self.ob = ob
        self.i += 1
        return self.act


@typechecked
def get_action(
    actor: nn.Module, critic: nn.Module, ob: Array[float, shapes.OB],
) -> Tuple[Array[int, ()], Array[float, ()]]:
    """
    Samples an action from the policy distribution, and returns the value
    prediction given an observation.

    Parameters
    ----------
    ac : ``ActorCritic``.
        The policy and value function approximator.
    ob : ``Array[float, shapes.OB]``.
        A single observation.

    Returns
    -------
    act : ``Array[int, ()]``.
        An integer action.
    val : ``Array[float, ()]``.
        The value prediction for ``ob``.
    """
    ob = torch.Tensor(ob)
    act = get_distribution(actor, ob).sample().numpy()

    # Detach from GPU, convert to numpy, then reshape (1,) -> ().
    val = critic(ob).detach().numpy().reshape(()).astype(np.float64)
    return act, val
