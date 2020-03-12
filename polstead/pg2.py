""" Functions for a simple policy gradient. """
import torch.nn as nn
from torch.distributions.categorical import Categorical

# import asta.check
from asta import dims, Tensor, typechecked

OB = dims.OBS_SHAPE
N_ACTS = dims.NUM_ACTIONS


@typechecked
def get_policy_distribution(
    policy: nn.Module, obs: Tensor[float, (*OB,)]
) -> Categorical:
    """ Computes the policy distribution for the state given by ``ob``. """
    logits = policy(obs)
    return Categorical(logits=logits)


@typechecked
def get_batch_policy_distribution(
    policy: nn.Module, obs: Tensor[float, (-1, *OB)]
) -> Categorical:
    """ Computes the policy distribution for a batch of observations. """
    logits = policy(obs)
    return Categorical(logits=logits)


@typechecked
def get_action(policy: nn.Module, obs: Tensor[float, (*OB,)]) -> int:
    """ Sample action from policy. """
    distribution = get_policy_distribution(policy, obs)
    sample: Tensor[int, ()] = distribution.sample()
    act: int = sample.item()
    return act


@typechecked
def compute_loss(
    policy: nn.Module,
    obs: Tensor[float, (-1, *OB)],
    act: Tensor[int, -1],
    weights: Tensor[float, -1],
) -> Tensor[float, ()]:
    """ See ``LOSS.txt``. """
    # TODO: It would be useful to have variable shape elements which have to
    # all be the same within a single function call, but can be anything.
    logp = get_batch_policy_distribution(policy, obs).log_prob(act)
    return -(logp * weights).mean()


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

    @typechecked
    def forward(self, ob: Tensor[float, (..., *OB)]) -> Tensor[float, ..., N_ACTS]:
        r""" Actor which implements the policy $\pi_{\theta}$. """
        logits = self._policy(ob)
        return logits
