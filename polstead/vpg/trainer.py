""" An example trainer for a simply policy gradient implementation. """
import json

import torch
from torch.optim import Adam

import gym

from oxentiel import Oxentiel

from asta import dims, shapes
from vpg import (
    get_action,
    compute_policy_loss,
    compute_value_loss,
    finish,
    ActorCritic,
    RolloutStorage,
)

SETTINGS_PATH = "settings_vpg.json"


def train(ox: Oxentiel) -> None:
    """ Training loop. """

    env: gym.Env = gym.make(ox.env_name)

    shapes.OB = env.observation_space.shape
    dims.N_ACTS = env.action_space.n

    ac = ActorCritic(shapes.OB[0], dims.N_ACTS, ox.hidden_dim)
    actor_optimizer = Adam(ac.pi.parameters(), lr=ox.lr)
    critic_optimizer = Adam(ac.v.parameters(), lr=ox.lr)
    rollouts = RolloutStorage()

    ob = env.reset()
    done = False

    for i in range(ox.iterations):

        # Critical: to add prev ob to rollouts buffer.
        prev_ob = ob

        ob_t = torch.Tensor(ob)
        act, val = get_action(ac, ob_t)
        ob, rew, done, _ = env.step(act)

        rollouts.add(prev_ob, act, val, rew)

        # If we're done, or we finished a batch.
        if done or (i > 0 and i % ox.batch_size == 0):
            rews = rollouts.rews
            vals = rollouts.vals
            last_val = 0 if done else vals[-1]
            ep_weights = finish(ox, rews, vals, last_val)
            rollouts.weights.extend(ep_weights)
            ob, done = env.reset(), False

        if i > 0 and i % ox.batch_size == 0:
            mean_ret, mean_ep_len = rollouts.stats()
            obs, acts, weights, rets = rollouts.get()

            actor_optimizer.zero_grad()
            policy_loss = compute_policy_loss(ac, obs, acts, weights)
            policy_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            value_loss = compute_value_loss(ac, obs, rets)
            value_loss.backward()
            critic_optimizer.step()

            print(f"Iteration: {i} \t ", end="")
            print(f"Loss: {policy_loss:.3f} \t ", end="")
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
