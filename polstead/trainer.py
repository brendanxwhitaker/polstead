""" An example trainer for a simply policy gradient implementation. """
import gym
from oxentiel import Oxentiel
from asta import dims


def train(ox: Oxentiel) -> None:
    """ Training loop. """
    env = gym.make(ox.env_name)

    # Set asta dims.
    dims.OBS_SHAPE = env.observation_space.shape
    dims.ACT_SHAPE = (env.action_space.n,)
    dims.NUM_ACTIONS = env.action_space.n
