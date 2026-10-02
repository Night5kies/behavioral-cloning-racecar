import os
from multiprocessing import Pool, cpu_count

import numpy as np

from grand_prix.env.car_racing import CarRacing
from grand_prix.expert import PurePursuitStanley


def collect_single_episode(env, episode_num=None, num_episodes=None):
    """
    Collect data from a single episode.

    Runs the expert, perturbs its actions with a noise model, and records
    (observation, action) pairs. Both sequential and parallel collection use this same function.

    Args:
        env: CarRacing environment instance
        episode_num: Episode number (for printing progress, None if not needed)
        num_episodes: Total number of episodes (for printing progress, None if not needed)

    Returns:
        obs_list: List of observations
        action_list: List of actions
    """
    obs_list = []
    action_list = []

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    # Expert controller
    mpc = PurePursuitStanley()
    done = False

    while not done:
        # Save observation BEFORE taking action (for imitation learning)
        obs_list.append(obs)

        # Expert action
        action = mpc.control(obs)

        # Noise model: independent zero-mean Gaussian noise per action dimension.
        # Action format: [steer, gas, brake]
        # Action ranges: steer in [-1, 1], gas in [0, 1], brake in [0, 1]
        noisy_action = np.array(action, dtype=np.float32, copy=True)

        # Add Gaussian noise to each action dimension
        steer_noise = np.random.normal(0.0, 0.05)
        gas_noise = np.random.normal(0.0, 0.02)
        brake_noise = np.random.normal(0.0, 0.02)
        noisy_action += np.array([steer_noise, gas_noise, brake_noise], dtype=np.float32)

        # Clip actions to valid ranges
        noisy_action[0] = np.clip(noisy_action[0], -1.0, 1.0)  # steer: [-1, 1]
        noisy_action[1] = np.clip(noisy_action[1], 0.0, 1.0)  # gas: [0, 1]
        noisy_action[2] = np.clip(noisy_action[2], 0.0, 1.0)  # brake: [0, 1]

        # Save action corresponding to the observation we just saved
        action_list.append(noisy_action)

        # Take step in environment
        obs, r, terminated, truncated, _ = env.step(noisy_action)

        total_reward += r
        if episode_num is not None and (steps % 200 == 0 or terminated or truncated):
            print(f"\nepisode {episode_num+1}/{num_episodes}")
            print("action " + str([f"{x:+0.2f}" for x in noisy_action]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        done = terminated or truncated

    return obs_list, action_list


def collect_expert_data_sequential():
    """Collect expert data sequentially with display enabled."""
    env = CarRacing(render_mode="human", camera_center="follow_car")
    obs_list = []
    action_list = []
    num_episodes = 10

    for episode in range(num_episodes):
        episode_obs_list, episode_action_list = collect_single_episode(
            env, episode_num=episode, num_episodes=num_episodes
        )
        obs_list.extend(episode_obs_list)
        action_list.extend(episode_action_list)

    env.close()

    # Validate data consistency for imitation learning
    if len(obs_list) != len(action_list):
        raise ValueError(
            f"Mismatch: {len(obs_list)} observations but {len(action_list)} actions. "
            "Each observation must have a corresponding action."
        )

    # Serialize expert data to numpy arrays
    os.makedirs("expert_data", exist_ok=True)
    np.save("expert_data/obs.npy", obs_list)
    np.save("expert_data/action.npy", action_list)
    print(
        f"\nSaved {len(action_list)} expert (observation, action) pairs for imitation learning."
    )


def _collect_episode_data_parallel(episode_idx):
    """Collect data from a single episode (worker function for parallel processing)."""
    # Set unique random seed for each worker based on episode index
    # This ensures diversity in the collected data across parallel workers
    np.random.seed(episode_idx)

    env = CarRacing(render_mode=None)  # Disable rendering for faster collection
    obs_list, action_list = collect_single_episode(env)
    env.close()
    return obs_list, action_list


def collect_expert_data_parallel():
    """Collect expert data in parallel across CPU cores with display disabled."""
    num_episodes = 10
    num_workers = cpu_count()

    print(f"Collecting expert data in parallel using {num_workers} CPU cores...")
    print("Display disabled for faster collection.")

    # Collect episodes in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(_collect_episode_data_parallel, range(num_episodes))

    # Combine results from all episodes
    obs_list = []
    action_list = []
    for episode_obs_list, episode_action_list in results:
        obs_list.extend(episode_obs_list)
        action_list.extend(episode_action_list)

    # Validate data consistency
    if len(obs_list) != len(action_list):
        raise ValueError(
            f"Mismatch: {len(obs_list)} observations but {len(action_list)} actions. "
            "Each observation must have a corresponding action."
        )

    # Serialize expert data to numpy arrays
    os.makedirs("expert_data", exist_ok=True)
    np.save("expert_data/obs.npy", obs_list)
    np.save("expert_data/action.npy", action_list)
    print(
        f"\nSaved {len(action_list)} expert (observation, action) pairs for imitation learning."
    )
