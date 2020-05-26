""" Vanilla policy gradient classes and functions. """
from typing import Tuple, List
import torch
import numpy as np
from asta import Array, Tensor, shapes, dims, symbols, typechecked

# pylint: disable=too-few-public-methods

T = symbols.T


@typechecked
class RolloutStorage:
    """ A buffer for trajectory data. """

    def __init__(self, batch_size: int, ob_shape: Tuple[int, ...]):
        self.obs: Array[float] = np.zeros((batch_size, *ob_shape))
        self.acts: Array[int] = np.zeros((batch_size,), dtype=np.int64)
        self.vals: Array[float] = np.zeros((batch_size,))
        self.rews: Array[float] = np.zeros((batch_size,))
        self.advs: Array[float] = np.zeros((batch_size,))
        self.rtgs: Array[float] = np.zeros((batch_size,))
        self.rets: List[float] = []
        self.lens: List[int] = []
        self.batch_len = 0
        self.ep_len = 0
        self.ep_start = 0

        self.ob_shape = ob_shape
        self.batch_size = batch_size

    def add(
        self,
        ob: Array[float, shapes.OB],
        act: Array[int, ()],
        val: Array[float, ()],
        rew: float,
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
        acts = torch.Tensor(self.acts).int()
        advs = torch.Tensor(self.advs)
        rtgs = torch.Tensor(self.rtgs)

        # Reset buffer.
        self.batch_len = 0
        self.obs = np.zeros((self.batch_size, *self.ob_shape))
        self.acts = np.zeros((self.batch_size,), dtype=np.int64)
        self.vals = np.zeros((self.batch_size,))
        self.rews = np.zeros((self.batch_size,))

        return obs, acts, advs, rtgs
