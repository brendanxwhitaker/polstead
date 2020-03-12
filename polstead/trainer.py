""" An example trainer for a simply policy gradient implementation. """
import json

import gym
from oxentiel import Oxentiel

import torch
from torch.optim import Adam

from pg import Trajectory, Policy, get_action, compute_loss
from asta import dims, Tensor

SETTINGS_PATH = "settings/settings.json"


def train(ox: Oxentiel) -> None:
    """ Training loop. """
    env: gym.Env = gym.make(ox.env_name)

    # Set asta dims.
    dims.OBS_SHAPE = env.observation_space.shape
    dims.NUM_ACTIONS = env.action_space.n

    policy = Policy(dims.OBS_SHAPE[0], dims.NUM_ACTIONS, ox.hidden_dim)
    optimizer = Adam(policy.parameters(), lr=ox.lr)

    ob = env.reset()
    done = False
    ema_ret = 0

    trajectory = Trajectory()

    for i in range(ox.iterations):

        ob_t: Tensor[float, dims.OBS_SHAPE] = torch.as_tensor(ob, dtype=torch.float32)
        act_t = get_action(policy, ob_t)

        act: int = act_t.item()
        ob, rew, done, _ = env.step(act)

        trajectory.add(ob, act, rew)

        if done or i % ox.batch_size == 0:
            ema_ret = trajectory.finish()
            ob = env.reset()
            done = False

        if i > 0 and i % ox.batch_size == 0:
            optimizer.zero_grad()
            obs, acts, weights = trajectory.get()

            batch_loss = compute_loss(policy, obs, acts, weights)
            batch_loss.backward()
            optimizer.step()

            print("Avg return:", sum(trajectory.ep_rets) / len(trajectory.ep_rets))
            # print("Batch loss:", batch_loss)
            # print("Avg length:", sum(trajectory.ep_lens) / len(trajectory.ep_lens))
            # print(f"EMA Return: {ema_ret}||||||", end="\n")

    print("")


def main() -> None:
    """ Run the trainer. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    train(ox)


if __name__ == "__main__":
    main()
