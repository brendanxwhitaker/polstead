""" Vanilla policy gradient classes and functions. """
from typing import Tuple

import numpy as np

import torch
from torch import nn

from asta import Array, shapes, symbols, typechecked

from polstead.core import ActorCritic
from polstead.losses import get_distribution
from polstead.functional import get_advantages, get_rewards_to_go

# pylint: disable=too-few-public-methods

T = symbols.T


@typechecked
class VPG(ActorCritic):
    r""" An actor-critic module with GAE and rewards-to-go. """

    def adv_fn(
        self,
        lam: float,
        gamma: float,
        rews: Array[float],
        vals: Array[float],
        last_val: float,
    ) -> Array[float]:
        """ The advantage function. """
        return get_advantages(lam, gamma, rews, vals, last_val)

    def tgt_fn(self, gamma: float, rews: Array[float]) -> Array[float]:
        """ The value prediction target function. """
        return get_rewards_to_go(gamma, rews)


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
