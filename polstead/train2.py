""" An example trainer for a simply policy gradient implementation. """
import json
from typing import List

import torch
from torch.optim import Adam

import gym
import numpy as np
from numpy.testing import assert_almost_equal

from oxentiel import Oxentiel

from pg2 import get_action, compute_loss, Policy, Trajectories
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

    trajectories = Trajectories()

    # reset episode-specific variables
    ob = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    for i in range(ox.iterations):

        # save obs
        batch_obs.append(ob.copy())

        # Critical to add the previous observation to trajectories buffer.
        prev_ob = ob

        # act in the environment
        act = get_action(policy, torch.as_tensor(ob, dtype=torch.float32))
        ob, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        trajectories.add(prev_ob, act, rew)

        if done or (i > 0 and i % ox.batch_size == 0):
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            trajectories.finish()

            # reset episode-specific variables
            ob, done, ep_rews = env.reset(), False, []

        if i > 0 and i % ox.batch_size == 0:

            assert_almost_equal(batch_obs, trajectories.obs)

            obs, acts, weights, mean_ret, mean_len = trajectories.get()

            oai_obs = torch.Tensor(batch_obs)
            oai_acts = torch.Tensor(batch_acts)
            oai_weights = torch.Tensor(batch_weights)

            assert_almost_equal(np.array(obs), np.array(oai_obs))

            optimizer.zero_grad()

            # batch_loss = compute_loss(policy, obs, acts, weights)
            batch_loss = compute_loss(policy, oai_obs, oai_acts, oai_weights)

            batch_loss.backward()
            optimizer.step()

            print(
                "Iteration: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
                % (i, batch_loss, mean_ret, mean_len)
            )

            del obs
            del acts
            del weights
            del mean_ret
            del mean_len

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
