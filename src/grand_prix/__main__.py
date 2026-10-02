from argparse import ArgumentParser

from grand_prix.env.car_racing import CarRacing
from grand_prix.env.collect_data import (
    collect_expert_data_parallel,
    collect_expert_data_sequential,
)
from grand_prix.neural_net import NeuralNetPolicy, load_policy, train_policy


def main():
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--evaluate-policy", type=str, help="Path to a saved policy checkpoint."
    )
    group.add_argument(
        "--evaluate-neural-net-policy",
        type=str,
        help="Path to a saved neural net policy checkpoint.",
    )
    group.add_argument(
        "--train-neural-net-policy",
        type=str,
        help="Path to save neural net policy during training.",
    )
    group.add_argument(
        "--collect-expert-data",
        action="store_true",
        help="Collect expert data for imitation learning.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Disable display and parallelize data collection across CPU cores (faster).",
    )

    args = parser.parse_args()

    if args.collect_expert_data:
        if args.fast:
            collect_expert_data_parallel()
        else:
            collect_expert_data_sequential()
    elif args.train_neural_net_policy:
        train_policy(
            data_dir="expert_data",
            save_path=args.train_neural_net_policy,
        )
    elif args.evaluate_neural_net_policy:
        # Load neural network policy
        print(
            f"Loading neural network policy from {args.evaluate_neural_net_policy}..."
        )
        model, feature_mean, feature_std = load_policy(
            model_path=args.evaluate_neural_net_policy,
            data_dir="expert_data",
        )
        policy = NeuralNetPolicy(model, feature_mean, feature_std)

        # Run evaluation
        env = CarRacing(render_mode="human", camera_center="follow_car")
        num_episodes = 1

        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = policy.control(obs)
                obs, r, terminated, truncated, _ = env.step(action)

                total_reward += r
                if steps % 200 == 0 or terminated or truncated:
                    print(f"\nepisode {episode+1}/{num_episodes}")
                    print("action " + str([f"{x:+0.2f}" for x in action]))
                    print(f"step {steps} total_reward {total_reward:+0.2f}")
                steps += 1
                done = terminated or truncated

        env.close()
    else:
        raise RuntimeError(
            "Either --evaluate-policy, --evaluate-neural-net-policy, --train-neural-net-policy, or --collect-expert-data must be specified."
        )


if __name__ == "__main__":
    main()