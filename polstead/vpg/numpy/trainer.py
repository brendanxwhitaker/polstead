""" A vanilla policy gradient implementation (numpy). """
import gym
from torch.optim import Adam
from asta import Array, Tensor, shapes
from oxentiel import Oxentiel

from vpg import ActorCritic, RolloutStorage
from vpg import (
    get_action,
    get_advantages,
    get_rewards_to_go,
    get_policy_loss,
    get_value_loss,
)


def train(ox: Oxentiel) -> None:
    """ Trains a policy gradient model with hyperparams from ``ox``. """

    # Make the environment.
    env: gym.Env = gym.make(ox.env_name)

    # Make the policy object.
    ac = ActorCritic(ox)

    # Make optimizers.
    policy_optimizer = Adam(ac.pi.parameters(), lr=ox.lr)
    value_optimizer = Adam(ac.v.parameters(), lr=ox.lr)

    # Create a buffer object to store trajectories.
    rollouts = RolloutStorage(ox)

    # Get the initial observation.
    ob: Array[float, shapes.OB]
    ob = env.reset()

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
        if done or (i > 0 and i % ox.batch_size == 0):

            # Step 1: Compute advantages and critic targets.

            # Get episode length.
            ep_len = rollouts.episode_length

            # Retrieve values and rewards for the current episode.
            vals: Array[float, ep_len]
            rews: Array[float, ep_len]
            # TODO: This should do some slicing to only get up to ``ep_len``.
            vals, rews = rollouts.get_episode_rewards_and_values()

            # The last value should be zero if this is the end of an episode.
            last_val: int = 0 if done else vals[-1]

            # Compute advantages and rewards-to-go.
            advs: Array[float, ep_len] = get_advantages(ox, rews, vals, last_val)
            rtgs: Array[float, ep_len] = get_rewards_to_go(ox, rews, vals, last_val)

            # Step 2: Reset vals and rews in buffer and record computed quantities.
            rollouts.vals[:] = 0
            rollouts.rews[:] = 0

            # Record the episode length.
            rollouts.lens.append(len(advs))

            # Record advantages and rewards-to-go.
            j = rollouts.ptr
            rollouts.advs[j : j + ep_len] = advs
            rollouts.rtgs[j : j + ep_len] = rtgs

            # Step 3: Reset the environment.
            ob = env.reset()

        # If we completed a batch.
        if i > 0 and i % ox.batch_size == 0:

            # Get batch data from the buffer.
            obs: Tensor[float, (ox.batch_size, *shapes.OB)]
            acts: Tensor[int, (ox.batch_size)]
            advs: Tensor[float, (ox.batch_size)]
            rtgs: Tensor[float, (ox.batch_size)]
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
