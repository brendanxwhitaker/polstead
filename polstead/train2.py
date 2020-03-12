""" An example trainer for a simply policy gradient implementation. """
import json
from typing import List

import torch
from torch.optim import Adam

import gym
import numpy as np

from oxentiel import Oxentiel

from pg2 import get_action, compute_loss, Policy
from asta import dims


SETTINGS_PATH = "settings/settings.json"


def train(ox: Oxentiel) -> None:
    """ Training loop. """

    env: gym.Env = gym.make(ox.env_name)

    dims.OBS_SHAPE = env.observation_space.shape
    dims.NUM_ACTIONS = env.action_space.n

    policy = Policy(dims.OBS_SHAPE[0], dims.NUM_ACTIONS, ox.hidden_dim)

    optimizer = Adam(policy.parameters(), lr=ox.lr)

    # make some empty lists for logging.
    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_weights: List[float] = []  # for R(tau) weighting in policy gradient
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    for i in range(ox.iterations):

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done or (i > 0 and i % ox.batch_size == 0):
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

        if i > 0 and i % ox.batch_size == 0:

            # take a single policy gradient update step
            optimizer.zero_grad()
            batch_loss = compute_loss(
                policy,
                obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                act=torch.as_tensor(batch_acts, dtype=torch.int32),
                weights=torch.as_tensor(batch_weights, dtype=torch.float32),
            )
            batch_loss.backward()
            optimizer.step()

            print(
                "Iteration: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
                % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
            )

            # make some empty lists for logging.
            batch_obs = []  # for observations
            batch_acts = []  # for actions
            batch_weights = []  # for R(tau) weighting in policy gradient
            batch_rets = []  # for measuring episode returns
            batch_lens = []  # for measuring episode lengths


def main() -> None:
    """ Run the trainer. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    train(ox)


if __name__ == "__main__":
    main()
