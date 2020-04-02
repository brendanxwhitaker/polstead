""" An example trainer for a simply policy gradient implementation. """
import json

import torch
from torch.optim import Adam

import gym
import numpy as np

from oxentiel import Oxentiel

from pg import get_action, compute_loss, Policy, RolloutStorage
from asta import dims, shapes

SETTINGS_PATH = "settings/settings.json"


def train(ox: Oxentiel) -> None:
    """ Training loop. """

    env: gym.Env = gym.make(ox.env_name)

    shapes.OB = env.observation_space.shape
    dims.N_ACTS = env.action_space.n

    policy = Policy(shapes.OB[0], dims.N_ACTS, ox.hidden_dim)
    optimizer = Adam(policy.parameters(), lr=ox.lr)
    rollouts = RolloutStorage()

    ob = env.reset()
    done = False

    for i in range(ox.iterations):

        # Critical: to add prev ob to rollouts buffer.
        prev_ob = ob

        ob_t = torch.Tensor(ob)
        act = get_action(policy, ob_t)
        ob, rew, done, _ = env.step(act)

        rollouts.add(prev_ob, act, rew)

        if done or (i > 0 and i % ox.batch_size == 0):
            rollouts.finish()
            ob, done = env.reset(), False

        if i > 0 and i % ox.batch_size == 0:
            mean_ret, mean_ep_len = rollouts.stats()
            obs, acts, weights = rollouts.get()

            optimizer.zero_grad()
            batch_loss = compute_loss(policy, obs, acts, weights)
            batch_loss.backward()
            optimizer.step()

            print(f"Iteration: {i} \t ", end="")
            print(f"Loss: {batch_loss:.3f} \t ", end="")
            print(f"Mean return: {mean_ret:.3f} \t ", end="")
            print(f"Mean episode length: {mean_ep_len:.3f} \t ", end="\n")


def main() -> None:
    """ Run the trainer. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    train(ox)


if __name__ == "__main__":
    main()
