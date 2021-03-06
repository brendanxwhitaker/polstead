import json
import itertools
import gym
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from asta import shapes, dims
from oxentiel import Oxentiel
from polstead.vanilla import VPG
from polstead.autoregressors import LSTM


SETTINGS_PATH = "settings/settings.json"


def main() -> None:
    """ Test. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    env = gym.make(ox.env_name)

    # Set shapes and dimensions for use in type hints.
    dims.BATCH = ox.batch_size
    dims.ACTS = env.action_space.n
    shapes.OB = env.observation_space.shape

    # Create actor and critic networks.
    actor = nn.Sequential(
        nn.Linear(shapes.OB[0], ox.hidden_size),
        nn.RReLU(),
        nn.Linear(ox.hidden_size, ox.hidden_size),
        nn.RReLU(),
        LSTM(ox.hidden_size, ox.hidden_size, ox.hidden_size, 3, 0.0, False),
        nn.Linear(ox.hidden_size, dims.ACTS),
    )
    critic = nn.Sequential(
        nn.Linear(shapes.OB[0], ox.hidden_size),
        nn.RReLU(),
        nn.Linear(ox.hidden_size, ox.hidden_size),
        nn.RReLU(),
        nn.Linear(ox.hidden_size, 1),
    )

    # Create optimizer and learning rate scheduler.
    parameters = itertools.chain(actor.parameters(), critic.parameters())
    optimizer = Adam(parameters, lr=ox.lr)
    scheduler = OneCycleLR(optimizer, ox.lr, ox.cycle_steps)

    # Create agent.
    agent = VPG(ox, env, actor, critic, optimizer, scheduler)

    ob = env.reset()
    rew = 0
    done = False

    for i in range(ox.iterations):

        if i > 0 and i % ox.batch_size == 0:
            mean_ret = np.mean(agent.rollouts.rets)
            num_rets = len(agent.rollouts.rets)
            lr = scheduler.get_lr()
            print(f"Iteration: {agent.i} | Mean return: {mean_ret:.3f} | ", end="")
            print(f"LR: {lr}")

        act = agent(ob, rew, done)
        ob, rew, done, info = env.step(int(act))

        if done:
            ob = env.reset()


if __name__ == "__main__":
    main()
