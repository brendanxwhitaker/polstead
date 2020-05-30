""" Vanilla policy gradient classes and functions. """
from typing import Tuple

import gym
import numpy as np

import torch
from torch import nn

from asta import Array, Tensor, shapes, dims, symbols, typechecked
from oxentiel import Oxentiel

from polstead.losses import get_distribution, get_value_loss, get_policy_loss
from polstead.rollouts import RolloutStorage
from polstead.functional import get_advantages, get_rewards_to_go

# pylint: disable=too-few-public-methods

T = symbols.T


@typechecked
class VPG(nn.Module):
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

        # Wrap the actor and critic in typecheckers (is this necessary?).
        self.actor = Actor(actor)
        self.critic = Critic(critic)

        self.ob: Array[float, shapes.OB]
        self.act: Array[int, ()]
        self.val: Array[float, ()]
        self.rollouts = RolloutStorage(ox.batch_size, env.observation_space.shape)

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
            rtgs: Array[float, dims.EP_LEN]

            # Retrieve values and rewards for the current episode.
            vals, rews = self.rollouts.get_episode_values_and_rewards()

            # The last value should be zero if this is the end of an episode.
            last_val: float = 0.0 if done else vals[-1]

            # Compute advantages and rewards-to-go.
            advs = get_advantages(self.ox.gamma, self.ox.lam, rews, vals, last_val)
            rtgs = get_rewards_to_go(self.ox.gamma, rews)

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
            self.rollouts.rtgs[j : j + dims.EP_LEN] = rtgs

            # Set the new episode starting point, and set episode length to zero.
            self.rollouts.ep_start = j + dims.EP_LEN
            self.rollouts.ep_len = 0

        # If we completed a batch.
        if self.rollouts.batch_len == self.ox.batch_size:

            # Types and shapes.
            obs: Tensor[float, (self.ox.batch_size, *shapes.OB)]
            acts: Tensor[int, (self.ox.batch_size)]

            # Get batch data from the buffer.
            obs, acts, advs, rtgs = self.rollouts.get_batch()

            # Run a backward pass on the policy (actor) and value function (critic).
            self.optimizer.zero_grad()
            policy_loss = get_policy_loss(self.actor, obs, acts, advs)
            value_loss = get_value_loss(self.critic, obs, rtgs)
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


@typechecked
class Actor(nn.Module):
    r"""
    The policy module. Computes logits for the action distribution.

    Parameters
    ----------
    ob_dim : ``int``.
        The last dimension size of the observations.
    hidden_dim : ``int``.
        Dimensionality of hidden layer.
    num_actions : ``int``.
        The ``action_space.n`` attribute for categorical action spaces.
    """

    def __init__(self, actor: nn.Module):
        super().__init__()
        self._actor = actor

    def forward(
        self, x: Tensor[float, (..., *shapes.OB)]
    ) -> Tensor[float, (..., dims.ACTS)]:
        r"""
        Takes as input observations or batches of observations.

        Parameters
        ----------
        x : ``Tensor[float, (..., *shapes.OB)]``.
            The ``...`` allows for a shape of ``shapes.OB`` as well as
            ``(X, *shapes.OB)``.

        Returns
        -------
        <torch.Tensor> : ``Tensor[float, (..., *dims.ACTS)]``.
            Either logits for constructing an action distribution, or a batch
            of the same.
        """
        return self._actor(x)


@typechecked
class Critic(nn.Module):
    r"""
    The value function module. Computes an estimate of the value function:
        $$
        V^{\pi}(s_t)
        $$
    where $\pi$ is the policy and $s_t$ is the observation at time $t$.

    Parameters
    ----------
    ob_dim : ``int``.
        The last dimension size of the observations.
    hidden_dim : ``int``.
        Dimensionality of hidden layer.
    """

    def __init__(self, critic: nn.Module):
        super().__init__()
        self._critic = critic

    def forward(self, x: Tensor[float, (..., *shapes.OB)]) -> Tensor[float, (..., 1)]:
        r"""
        Takes as input observations or batches of observations.

        Parameters
        ----------
        x : ``Tensor[float, (..., 1)]``.
            The ``...`` allows for a shape of ``1`` or ``(X, 1)`` for some
            positive integer ``X``.

        Returns
        -------
        <torch.Tensor> : ``Tensor[float, (..., 1)]``.
            Either an estimate of the value function, or a batch of the same.
        """
        return self._critic(x)
