""" An example trainer for a simply policy gradient implementation. """
import gym

import torch
from torch.optim import Adam

from asta import dims
from oxentiel import Oxentiel
from pg import Trajectory


def train(ox: Oxentiel) -> None:
    """ Training loop. """
    env: gym.Env = gym.make(ox.env_name)

    # Set asta dims.
    dims.OBS_SHAPE: Tuple[int, ...] = env.observation_space.shape
    dims.ACT_SHAPE: Tuple[int] = (env.action_space.n,)
    dims.NUM_ACTIONS: int = env.action_space.n

    optimizer = Adam(policy.parameters(), lr=ox.lr)
    policy = Policy(dims.OBS_SHAPE[0], dims.NUM_ACTIONS, ox.hidden_dim)
    trajectory = Trajectory()

    ob = env.reset()
    done = False
    weights = []

    for i in range(ox.iterations):

        ob_t: Tensor[float, dims.OBS_SHAPE] = torch.as_tensor(ob)

        act = get_action(ob_t)
        ob, rew, done, _ = env.step(act)

        trajectory.add(ob, act, rew)

        if done:
            ep_ret = sum(trajectory.rews)
            ep_len = len(trajectory.rews)

            weights.extend([ep_ret] * ep_len)
