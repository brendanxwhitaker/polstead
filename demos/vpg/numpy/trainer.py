""" A vanilla policy gradient implementation (numpy). """
import time
import json
import gym
import numpy as np
from torch.optim import Adam
from asta import Array, Tensor, shapes, dims
from oxentiel import Oxentiel

from vpg import ActorCritic, RolloutStorage
from vpg import (
    get_action,
    get_advantages,
    get_rewards_to_go,
    get_policy_loss,
    get_value_loss,
)

SETTINGS_PATH = "settings_vpg.json"


def train(ox: Oxentiel) -> None:
    """ Trains a policy gradient model with hyperparams from ``ox``. """

    # Make the environment.
    env: gym.Env = gym.make(ox.env_name)

    # Set shapes and dimensions for use in type hints.
    dims.BATCH = ox.batch_size
    dims.ACTS = env.action_space.n
    shapes.OB = env.observation_space.shape

    # Make the policy object.
    ac = ActorCritic(shapes.OB[0], ox.hidden_dim, dims.ACTS)

    # Make optimizers.
    policy_optimizer = Adam(ac.pi.parameters(), lr=ox.lr)
    value_optimizer = Adam(ac.v.parameters(), lr=ox.lr)

    # Create a buffer object to store trajectories.
    rollouts = RolloutStorage(ox.batch_size, shapes.OB)

    # Get the initial observation.
    ob: Array[float, shapes.OB]
    ob = env.reset()

    t_start = time.time()

    for i in range(ox.iterations):

        # Sample an action from the policy and estimate the value of current state.
        act: Array[int, ()]
        val: Array[float, ()]
        act, val = get_action(ac, ob)

        # Step the environment to get new observation, reward, done status, and info.
        next_ob: Array[float, shapes.OB]
        rew: int
        done: bool
        next_ob, rew, done, _ = env.step(int(act))

        # Add data for a timestep to the buffer.
        rollouts.add(ob, act, val, rew)

        # Don't forget to update the observation.
        ob = next_ob

        # If we reached a terminal state, or we completed a batch.
        if done or rollouts.batch_len == ox.batch_size:

            # Step 1: Compute advantages and critic targets.

            # Get episode length.
            ep_len = rollouts.ep_len
            dims.EP_LEN = ep_len

            # Retrieve values and rewards for the current episode.
            vals: Array[float, ep_len]
            rews: Array[float, ep_len]
            vals, rews = rollouts.get_episode_values_and_rewards()

            # The last value should be zero if this is the end of an episode.
            last_val: float = 0.0 if done else vals[-1]

            # Compute advantages and rewards-to-go.
            advs: Array[float, ep_len] = get_advantages(ox, rews, vals, last_val)
            rtgs: Array[float, ep_len] = get_rewards_to_go(ox, rews)

            # Record the episode length.
            if done:
                rollouts.lens.append(len(advs))
                rollouts.rets.append(np.sum(rews))

            # Step 2: Reset vals and rews in buffer and record computed quantities.
            rollouts.vals[:] = 0
            rollouts.rews[:] = 0

            # Record advantages and rewards-to-go.
            j = rollouts.ep_start
            assert j + ep_len <= ox.batch_size
            rollouts.advs[j : j + ep_len] = advs
            rollouts.rtgs[j : j + ep_len] = rtgs
            rollouts.ep_start = j + ep_len
            rollouts.ep_len = 0

            # Step 3: Reset the environment.
            ob = env.reset()

        # If we completed a batch.
        if rollouts.batch_len == ox.batch_size:

            # Get batch data from the buffer.
            obs: Tensor[float, (ox.batch_size, *shapes.OB)]
            acts: Tensor[int, (ox.batch_size)]
            obs, acts, advs, rtgs = rollouts.get_batch()

            # Run a backward pass on the policy (actor).
            policy_optimizer.zero_grad()
            policy_loss = get_policy_loss(ac.pi, obs, acts, advs)
            policy_loss.backward()
            policy_optimizer.step()

            # Run a backward pass on the value function (critic).
            value_optimizer.zero_grad()
            value_loss = get_value_loss(ac.v, obs, rtgs)
            value_loss.backward()
            value_optimizer.step()

            # Reset pointers.
            rollouts.batch_len = 0
            rollouts.ep_start = 0

            # Print statistics.
            mean_ret = np.mean(rollouts.rets)
            mean_len = np.mean(rollouts.lens)
            print(f"Iteration: {i + 1} | ", end="")
            print(f"Time: {time.time() - t_start:.5f} | ", end="")
            print(f"Mean episode return: {mean_ret:.5f} | ", end="")
            print(f"Mean episode length: {mean_len:.5f}")
            t_start = time.time()
            rollouts.rets = []
            rollouts.lens = []


def main() -> None:
    """ Just loads the settings file and calls ``train()``. """
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = json.load(settings_file)
    ox = Oxentiel(settings)
    train(ox)


if __name__ == "__main__":
    main()
