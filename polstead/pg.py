""" An implementation of a simple policy gradient agent. """
from typing import List

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


class Policy(nn.Module):
    """ The parameterized action-value and state-value functions. """

    def __init__(self, observation_size: int, num_actions: int, hidden_size: int):
        super().__init__()
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
    def forward(self, ob: Tensor[float, (-1, OBS_SHAPE)]) -> Tensor[float, NUM_ACTIONS]:
        r""" Actor which implements the policy $\pi_{\theta}$. """
        logits = self._policy(ob)
        return logits


@typechecked
def get_policy_distribution(
    policy: nn.Module, obs: Tensor[float, (-1, *OBS_SHAPE)]
) -> Categorical:
    """ Computes the policy distribution for the state given by ``ob``. """
    logits: Tensor[float, NUM_ACTIONS] = policy(obs)
    distribution = Categorical(logits=logits)
    return distribution


COMPUTE_LOSS_DOCSTRING = r"""
    The loss function is derived from the formula for the gradient estimator.

    $J$ : expected return function.
    $\theta$ : parameters/weights of the policy.
    $\pi_{\theta}$ : the policy parameterized by $\theta$.
    $\tau$ : a trajectory.
    $R(\tau)$ : the return given by summing all rewards in a trajectory.
    $D$ : set of all trajectories $\tau$.
    #T$ : number of environment steps in a given trajectory.
    $t$ : index of current timestep in a trajectory.
    $a_t$ : action at timestep $t$.
    $s_t$ : state at timestep $t$.

    In this case, the gradient estimator is given by:

    $$
        \hat(g)
        =
            \frac{1}{|D|} \sum_{\tau \in D} \sum_{t = 0}^{T}
            \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) R(\tau).
    $$

    We aim to maximize the expected return return of the policy, given by

    $$
    J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}} [R(\tau)].
    $$

    We do this by computing the gradient of the expected returns with respect
    to our parameters $\theta$, which tells us how to update said parameters.
    This quantity is known as the policy gradient, denoted

    $$
    \nabla_{\theta} J(\pi_{\theta}).
    $$

    Libraries like torch allow us to perform stochastic gradient descent (or a
    similar optimization algorithm) on an arbitrary loss function, which
    minimizes the value of the function by updating the trainable parameters
    involved in the computation of that function via backpropagation.

    So we use as our loss function the expression being differentiated in our
    formula for the gradient estimator. Note that the sum of gradients is the
    gradient of the sum, and so we can take $\nabla_{\theta}$ out of the double
    sum. The resulting expression is

    $$
            \frac{1}{|D|} \sum_{\tau \in D} \sum_{t = 0}^{T}
            \log \pi_{\theta}(a_t|s_t) R(\tau).
    $$

    If we were to use this as our loss function and call ``.backward()`` on its
    value using torch, we would be minimizing its value, hence minimizing
    expected returns. But since we wish to maximize expected rewards, we want
    our policy gradient to remain positive (the function for which the gradient
    of the above is an estimator). Thus we can use the negative of the above as
    our loss function.

    Assuming we run one policy update per trajectory, we don't need the outer
    sum, and hence the loss function is given by

    $$
        L(\pi_{\theta})
        =
            - \sum_{t = 0}^{T} \log \pi_{\theta}(a_t|s_t) R(\tau).
    $$

    This is what is implemented below.
    """


@typechecked
def compute_loss(
    policy: nn.Module,
    obs: Tensor[float, (-1, *OBS_SHAPE)],
    actions: Tensor[float, (-1, *ACT_SHAPE)],
    weights: Tensor[float, -1],
) -> Tensor[float, ()]:
    """ ^^^COMPUTE_LOSS_DOCSTRING^^^ """
    policy_distribution = get_policy_distribution(policy, obs)
    logp: Tensor[float, (-1, *ACT_SHAPE)] = policy_distribution.log_prob(actions)
    loss = -(logp * weights).mean()
    return loss
