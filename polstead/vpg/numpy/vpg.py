""" Vanilla policy gradient classes and functions. """
from torch import nn
from asta import Tensor, shapes, dims, typechecked

# pylint: disable=too-few-public-methods


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
class RolloutStorage:
    """ A buffer for trajectory data. """

    def __init__(self, batch_size: int, ob_shape: Tuple[int, ...]):
        self.obs: Array[float] = np.zeros((batch_size, *ob_shape))
        self.acts: Array[int] = np.zeros((batch_size,))
        self.vals: Array[float] = np.zeros((batch_size,))
        self.rews: Array[float] = np.zeros((batch_size,))
        self.batch_len = 0
        self.ep_len = 0

    def add(
        self,
        ob: Array[float, shapes.OB],
        act: Array[int, ()],
        val: Array[float, ()],
        rew: int,
    ) -> None:
        """ Add data for timestep to the buffer. """
        t = self.batch_len
        self.obs[t] = ob
        self.acts[t] = act
        self.batch_len += 1

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
