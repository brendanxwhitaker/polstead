""" An example trainer for a simply policy gradient implementation. """
import json

import torch
from torch.optim import Adam

import gym
import numpy as np

from oxentiel import Oxentiel

from pg import get_action, compute_loss, Policy, Trajectories
from asta import dims

SETTINGS_PATH = "settings/settings.json"


def train(ox: Oxentiel) -> None:
    """ Training loop. """

    env: gym.Env = gym.make(ox.env_name)

    dims.OBS_SHAPE = env.observation_space.shape
    dims.NUM_ACTIONS = env.action_space.n

    policy = Policy(dims.OBS_SHAPE[0], dims.NUM_ACTIONS, ox.hidden_dim)
    optimizer = Adam(policy.parameters(), lr=ox.lr)
    trajectories = Trajectories()

    ob = env.reset()
    done = False

    for i in range(ox.iterations):

        # Critical: to add prev ob to trajectories buffer.
        prev_ob = ob

        ob_t = torch.Tensor(ob)
        act = get_action(policy, ob_t)
        ob, rew, done, _ = env.step(act)

        trajectories.add(prev_ob, act, rew)

        if done or (i > 0 and i % ox.batch_size == 0):
            trajectories.finish()
            ob, done = env.reset(), False

        if i > 0 and i % ox.batch_size == 0:
            mean_ret, mean_len = trajectories.stats()
            obs, acts, weights = trajectories.get()

            optimizer.zero_grad()
            batch_loss = compute_loss(policy, obs, acts, weights)
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


def main() -> None:
    """ Run the trainer. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    train(ox)


if __name__ == "__main__":
    main()
