""" Vanilla policy gradient classes and functions. """
from typing import Tuple

import scipy.signal
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical

from asta import Array, Tensor, shapes, dims, symbols, typechecked
from oxentiel import Oxentiel

# pylint: disable=too-few-public-methods

T = symbols.T


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

    def __init__(self, ob_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self._actor = nn.Sequential(
            nn.Linear(ob_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
            nn.Identity(),
        )

    def forward(
        self, x: Tensor[float, (..., *shapes.OB)]
    ) -> Tensor[float, (..., *dims.ACTS)]:
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

    def __init__(self, ob_dim: int, hidden_dim: int):
        super().__init__()
        self._critic = nn.Sequential(
            nn.Linear(ob_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Identity(),
        )

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


@typechecked
class ActorCritic:
    r""" A wrapper around Actor and Critic objects. """

    def __init__(self, ob_dim: int, hidden_dim: int, num_actions: int):
        self.pi = Actor(ob_dim, hidden_dim, num_actions)
        self.v = Critic(ob_dim, hidden_dim)


@typechecked
def get_value_loss(
    v: Critic,
    obs: Tensor[float, (dims.BATCH, *shapes.OB)],
    rtgs: Tensor[float, dims.BATCH],
) -> Tensor[float, ()]:
    """
    Computes mean squared error of value function predictions.

    Parameters
    ----------
    v : ``Critic``.
        The network which approximates the value function. This maps states
        (observations) to estimates of the infinite discounted horizon total
        reward expected from that state.
    obs : ``Tensor[float, (dims.BATCH, *shapes.OB)]``.
        A batch of observations.
    rtgs : ``Tensor[float, dims.BATCH]``.
        A batch of rewards-to-go, the targets for the value predictions.

    Returns
    -------
    <loss> : ``Tensor[float, ()]``.
        The mean squared error of the value predictions as a scalar tensor.
    """
    return torch.mean((v(obs) - rtgs) ** 2)


@typechecked
def get_policy_loss(
    pi: Actor,
    obs: Tensor[float, (dims.BATCH, *shapes.OB)],
    acts: Tensor[int, dims.BATCH],
    advs: Tensor[float, dims.BATCH],
) -> Tensor[float, ()]:
    r"""
    Computes the loss used to update the policy/actor.

    Note that we wish to maximize our expected returns $J(\theta)$, which we do
    by maximizing the gradient of our expected returns with respect to the
    policy parameters $\theta$. This is equivalent to maximizing our gradient
    estimator $\hat{g}$ (see LOSS.txt). This is a parametrized scalar value, so
    we can achieve this by a form of gradient ascent. Equivalently, we minimize
    the negative of this value by gradient descent. This amounts to minimizing
    the log probability of actions taken given states under our policy
    multiplied by some weight approximating the return/utility of the
    trajectory, which is the advantage in this case. Formally, we have
    $$
    L(\pi_{\theta})
    =
        -sum_{t = 0}^T \log \pi_{\theta}(a_t | s_t) * A_t(s_t).
    $$

    Parameters
    ----------
    pi : ``Actor``.
        The network/nn.Module which computes logits for the policy distribution.
    obs : ``Tensor[float, (dims.BATCH, *shapes.OB)]``.
        A batch of observations.
    advs : ``Tensor[float, dims.BATCH]``.
        A batch of advantages, which are the weights for the action
        log-probabilities. These serve as a measure of the utility of the
        observations/states. Specifically, they measure how much better this
        state is than the other options.

    Returns
    -------
    <loss> : ``Tensor[float, ()]``.
        The negative of the sum of the products of log probs and weights, since
        this sum is what we wish to maximize.
    """
    log_probs = get_batch_distribution(pi, obs).log_prob(acts)
    return -torch.sum(log_probs * advs)


@typechecked
def get_action(
    ac: ActorCritic, ob: Array[float, shapes.OB],
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
    act = get_distribution(ac.pi, ob).sample().numpy()
    val = ac.v(ob)
    return act, val


@typechecked
def get_batch_distribution(
    pi: Actor, obs: Tensor[float, (dims.BATCH, *shapes.OB)],
) -> Categorical:
    """ Returns a distribution over actions given a batch of observations. """
    return Categorical(logits=pi(obs))


@typechecked
def get_distribution(pi: Actor, ob: Tensor[float, shapes.OB],) -> Categorical:
    """ Returns a distribution over actions given a single observation. """
    return Categorical(logits=pi(ob))


@typechecked
class RolloutStorage:
    """ A buffer for trajectory data. """

    def __init__(self, batch_size: int, ob_shape: Tuple[int, ...]):
        self.obs: Array[float] = np.zeros((batch_size, *ob_shape))
        self.acts: Array[int] = np.zeros((batch_size,))
        self.vals: Array[float] = np.zeros((batch_size,))
        self.rews: Array[float] = np.zeros((batch_size,))
        self.advs: Array[float] = np.zeros((batch_size,))
        self.rtgs: Array[float] = np.zeros((batch_size,))
        self.lens: List[int] = []
        self.batch_len = 0
        self.ep_start = 0
        self.ep_len = 0

        self.ob_shape = ob_shape
        self.batch_size = batch_size

    def add(
        self,
        ob: Array[float, shapes.OB],
        act: Array[int, ()],
        val: Array[float, ()],
        rew: int,
    ) -> None:
        """ Add data for timestep to the buffer. """
        # Add to objects which get reset each batch.
        t = self.batch_len
        self.obs[t] = ob
        self.acts[t] = act
        self.batch_len += 1

        # Add to objects that get reset each episode.
        t = self.ep_len
        self.vals[t] = val
        self.rews[t] = rew
        self.ep_len += 1

    def get_episode_values_and_rewards(
        self,
    ) -> Tuple[Array[float, dims.EP_LEN], Array[float, dims.EP_LEN]]:
        """ Returns values and rewards up to current episode length. """
        t = self.ep_len
        return self.vals[:t], self.rews[:t]

    def get_batch(
        self,
    ) -> Tuple[
        Tensor[float, (dims.BATCH, *shapes.OB)],
        Tensor[int, dims.BATCH],
        Tensor[float, dims.BATCH],
        Tensor[float, dims.BATCH],
    ]:
        """
        Casts batch data to tensors and returns it.

        Returns
        -------
        obs : ``Tensor[float, (dims.BATCH, *shapes.OB)]``.
            A batch of observations.
        acts : ``Tensor[int, dims.BATCH]``.
            A batch of actions.
        advs : ``Tensor[float, dims.BATCH]``.
            A batch of advantages.
        rtgs : ``Tensor[float, dims.BATCH]``.
            A batch of rewards-to-go.
        """
        assert self.batch_len == self.batch_size
        obs = torch.Tensor(self.obs)
        acts = torch.Tensor(self.acts)
        advs = torch.Tensor(self.advs)
        rtgs = torch.Tensor(self.rtgs)

        # Reset buffer.
        self.batch_len = 0
        self.obs = np.zeros((self.batch_size, *self.ob_shape))
        self.acts = np.zeros((self.batch_size,))
        self.vals = np.zeros((self.batch_size,))
        self.rews = np.zeros((self.batch_size,))

        return obs, acts, advs, rtgs


@typechecked
def discounted_cumulative_sum(arr: Array[float, T], discount: float) -> Array[float, T]:
    r"""
    Computes
    $$
    b_t = \sum_{l = 0}^T (discount)^l * a_{t + l}.
    $$
    which is easily seen to map:
    ```
    arr = [a0, a1, a2, ...]
    ```
    to
    res = [
        a0 + (a1 * discount) + (a2 * (discount)^2) + ...,
        a1 + (a2 * discount) + ...,
        a2 + ...,
    ]

    Parameters
    ----------
    arr : ``Array[float, T]``.
        Rank-1 array containing the values we wish to discount-sum.
    discount : ``float``.
        The discount factor, should be in $[0, 1]$.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], arr[::-1], axis=0)[::-1]


@typechecked
def get_advantages(
    ox: Oxentiel,
    rews: Array[float, dims.EP_LEN],
    vals: Array[float, dims.EP_LEN],
    last_val: float,
) -> Array[float, dims.EP_LEN]:
    """ Computes advantages to be used as weights for log probabilities of actions. """
    # Get a version of vals with the last timestep added.
    augmented_vals: Array[float, dims.EP_LEN + 1] = np.append(vals, last_val)

    # Compute the TD-residual of $V$ with discount $\gamma$ (GAE, Schulman et al. 2016).
    # $\delta_t^V  = r_t + \gamma V(s_{t + 1}) - V(s_t)$.
    deltas: Array[float, dims.EP_LEN] = rews + ox.gamma * augmented_vals[1:] - vals

    # Compute the below sum, where $\delta_j^V$ is taken to be zero for $t > EP_LEN$.
    # $\hat{A}_t^{GAE} = \sum_{l = 0}^{\infty} (\gamma \lambda)^l \delta_{t + l}^V$.
    advs = discounted_cumulative_sum(deltas, ox.gamma * ox.lam)
    return advs


@typechecked
def get_rewards_to_go(
    ox: Oxentiel, rews: Array[float, dims.EP_LEN],
) -> Array[float, dims.EP_LEN]:
    """ Compute rewards-to-go to be used as targets for value function approx. """
    rtgs: Array[float, dims.EP_LEN] = discounted_cumulative_sum(rews, ox.gamma)
    return rtgs
