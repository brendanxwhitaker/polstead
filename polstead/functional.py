""" Vanilla policy gradient classes and functions. """
import scipy.signal
import numpy as np

from asta import Array, dims, symbols, typechecked

# pylint: disable=too-few-public-methods

T = symbols.T


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
    lam: float,
    gamma: float,
    rews: Array[float, dims.EP_LEN],
    vals: Array[float, dims.EP_LEN],
    last_val: float,
) -> Array[float, dims.EP_LEN]:
    """ Computes advantages to be used as weights for log probabilities of actions. """
    # Get a version of vals with the last timestep added.
    augmented_vals: Array[float, dims.EP_LEN + 1] = np.append(vals, last_val)

    # Compute the TD-residual of $V$ with discount $\gamma$ (GAE, Schulman et al. 2016).
    # $\delta_t^V  = r_t + \gamma V(s_{t + 1}) - V(s_t)$.
    deltas: Array[float, dims.EP_LEN] = rews + gamma * augmented_vals[1:] - vals

    # Compute the below sum, where $\delta_j^V$ is taken to be zero for $t > EP_LEN$.
    # $\hat{A}_t^{GAE} = \sum_{l = 0}^{\infty} (\gamma \lambda)^l \delta_{t + l}^V$.
    advs = discounted_cumulative_sum(deltas, gamma * lam)
    return advs


@typechecked
def get_rewards_to_go(
    gamma: float, rews: Array[float, dims.EP_LEN],
) -> Array[float, dims.EP_LEN]:
    """ Compute rewards-to-go to be used as targets for value function approx. """
    rtgs: Array[float, dims.EP_LEN] = discounted_cumulative_sum(rews, gamma)
    return rtgs
