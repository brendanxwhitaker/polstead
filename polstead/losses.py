""" Vanilla policy gradient classes and functions. """
import torch
import torch.nn as nn
from torch.distributions import Categorical
from asta import Tensor, shapes, dims, symbols, typechecked

# pylint: disable=too-few-public-methods

T = symbols.T


@typechecked
def get_batch_distribution(
    pi: nn.Module, obs: Tensor[float, (dims.BATCH, *shapes.OB)],
) -> Categorical:
    """ Returns a distribution over actions given a batch of observations. """
    return Categorical(logits=pi(obs))


@typechecked
def get_distribution(pi: nn.Module, ob: Tensor[float, shapes.OB]) -> Categorical:
    """ Returns a distribution over actions given a single observation. """
    return Categorical(logits=pi(ob))


@typechecked
def get_value_loss(
    v: nn.Module,
    obs: Tensor[float, (dims.BATCH, *shapes.OB)],
    rtgs: Tensor[float, dims.BATCH],
) -> Tensor[float, ()]:
    """
    Computes mean squared error of value function predictions.

    Parameters
    ----------
    v : ``nn.Module``.
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
def get_policy_entropy(
    pi: nn.Module, obs: Tensor[float, (dims.BATCH, *shapes.OB)],
) -> Tensor[float, ()]:
    r"""
    Computes the entropy of the action distribution of the policy given a batch
    of observations.

    This is given by $H(\pi(s_t;\theta'))$ in the A3C paper.

    Parameters
    ----------
    pi : ``nn.Module``.
        The network/nn.Module which computes logits for the policy distribution.
    obs : ``Tensor[float, (dims.BATCH, *shapes.OB)]``.
        A batch of observations.

    Returns
    -------
    entropy : ``Tensor[float, ()]``.
        Entropy as a float tensor.
    """
    dist = get_batch_distribution(pi, obs)
    entropy = dist.entropy()
    return entropy


@typechecked
def get_policy_loss(
    pi: nn.Module,
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
    pi : ``nn.Module``.
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
