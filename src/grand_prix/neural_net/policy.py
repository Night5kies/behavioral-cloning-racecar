import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grand_prix.neural_net.dataset import ObservationDataset, create_data_loaders


class PolicyNetwork(nn.Module):
    """
    MLP policy network that maps observations to actions (behavioral cloning).
    """

    def __init__(self):
        super(PolicyNetwork, self).__init__()

        input_dim = 10  # Observation features from ObservationDataset._extract_features
        hidden_dim = 128
        output_dim = 3  # [steer, gas, brake]

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, observation_dim)

        Returns:
            Action tensor of shape (batch_size, action_dim)
            Actions: steer in [-1,1], gas/brake in [0,1]
        """
        single_input = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True

        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        raw = self.fc_out(x)
        steer = torch.tanh(raw[..., :1])  # [-1, 1]
        gas_brake = torch.sigmoid(raw[..., 1:])  # [0, 1]
        actions = torch.cat([steer, gas_brake], dim=-1)

        if single_input:
            actions = actions.squeeze(0)
        return actions


def train_policy(
    data_dir="expert_data",
    save_path="policy.pt",
):
    """
    Train a neural network policy using imitation learning on expert demonstrations.
    """
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100
    train_split = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=train_split,
    )

    train_size = len(train_loader.dataset)  # type: ignore
    val_size = len(val_loader.dataset)  # type: ignore

    model = PolicyNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nTraining policy network...")
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    best_val_loss = float("inf")
    best_train_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        for observations, actions in train_loader:
            observations = observations.to(device)
            actions = actions.to(device)

            model.train()
            optimizer.zero_grad()

            predictions = model(observations)
            loss = criterion(predictions, actions)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * observations.size(0)

        model.eval()
        with torch.no_grad():
            for observations, actions in val_loader:
                observations = observations.to(device)
                actions = actions.to(device)
                preds = model(observations)
                loss = criterion(preds, actions)
                val_loss += loss.item() * observations.size(0)

        if train_size > 0:
            train_loss /= train_size
        if val_size > 0:
            val_loss /= val_size

        best_train_loss = min(best_train_loss, train_loss)
        best_val_loss = min(best_val_loss, val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

    print(
        f"\nTraining complete! Best train loss: {best_train_loss:.6f}, Best val loss: {best_val_loss:.6f}"
    )

    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )
    torch.save(model.state_dict(), save_path)
    print(f"Saved final model to {save_path}")


def load_policy(
    model_path,
    data_dir="expert_data",
    device=None,
):
    """
    Load a trained policy network (behavioral cloning).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_path = Path(data_dir) / "obs.npy"
    if obs_path.exists():
        obs_list = np.load(obs_path, allow_pickle=True)
        all_features = [ObservationDataset._extract_features(obs) for obs in obs_list]
        all_features = np.array(all_features, dtype=np.float32)
        feature_mean = all_features.mean(axis=0)
        feature_std = all_features.std(axis=0) + 1e-8
    else:
        print(f"Warning: {obs_path} not found. Using zero mean and unit std.")
        feature_mean = np.zeros(10)
        feature_std = np.ones(10)

    model = PolicyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, feature_mean, feature_std


class NeuralNetPolicy:
    """
    Wrapper class for neural network policy that handles feature extraction and normalization.
    """

    def __init__(self, model, feature_mean, feature_std, device=None):
        self.model = model
        self.model.eval()
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def control(self, obs):
        """
        Get action from neural network policy (imitation learning).
        """
        features = ObservationDataset._extract_features(obs)
        features = (features - self.feature_mean) / self.feature_std

        features_tensor = torch.FloatTensor(features).to(self.device)
        with torch.no_grad():
            action_tensor = self.model(features_tensor)

        action = action_tensor.cpu().numpy()
        return action.astype(np.float32)


# ========================== PPO IMPLEMENTATION ==========================


class PPOActorCritic(nn.Module):
    """
    Shared trunk with separate policy (mean, log_std) and value heads.
    Gaussian policy; actions squashed to env ranges.
    """

    def __init__(self, input_dim=10, hidden_dim=128, output_dim=3, log_std_init=-0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.ones(output_dim) * log_std_init)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        x = x.float()
        h = self.trunk(x)
        mean = self.mean_head(h)
        value = self.value_head(h).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        if single:
            mean = mean.squeeze(0)
            value = value.squeeze(0)
            log_std = log_std.squeeze(0)
        return mean, log_std, value

    def sample_action(self, obs_tensor):
        mean, log_std, value = self.forward(obs_tensor)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(-1)

        steer = torch.tanh(raw_action[..., :1])  # [-1, 1]
        gas_brake = torch.sigmoid(raw_action[..., 1:])  # [0, 1]
        action = torch.cat([steer, gas_brake], dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs_tensor, actions):
        mean, log_std, value = self.forward(obs_tensor)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        # Approximate log prob using squashed actions directly (simplified)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, gamma, lam):
        advantages = []
        gae = 0.0
        values = self.values + [0.0]
        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, self.values)]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
            returns, dtype=torch.float32
        )

    def to_tensors(self, device):
        obs = torch.stack(self.obs).to(device)
        actions = torch.stack(self.actions).to(device)
        log_probs = torch.stack(self.log_probs).to(device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        return obs, actions, log_probs, values

    def clear(self):
        self.__init__()


def train_ppo(
    env,
    feature_mean,
    feature_std,
    total_timesteps=50_000,
    rollout_steps=2048,
    epochs=10,
    minibatch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    learning_rate=3e-4,
):
    """
    Minimal PPO training loop using on-policy rollouts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset()
    model = PPOActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    buffer = RolloutBuffer()

    def preprocess(obs_dict):
        feats = ObservationDataset._extract_features(obs_dict)
        feats = (feats - feature_mean) / feature_std
        return torch.tensor(feats, dtype=torch.float32, device=device)

    global_step = 0
    while global_step < total_timesteps:
        buffer.clear()
        for _ in range(rollout_steps):
            obs_tensor = preprocess(obs)
            with torch.no_grad():
                action_tensor, log_prob, value = model.sample_action(obs_tensor)
            action_np = action_tensor.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                obs_tensor.cpu(),
                torch.tensor(action_np, dtype=torch.float32),
                log_prob.cpu(),
                reward,
                done,
                float(value.cpu().item()),
            )
            obs = next_obs
            global_step += 1
            if done:
                obs, _ = env.reset()
                buffer.values.append(0.0)
                break

        advantages, returns = buffer.compute_gae(gamma, gae_lambda)
        obs_batch, act_batch, old_logp_batch, _ = buffer.to_tensors(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs_batch.size(0)
        for _ in range(epochs):
            idx = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = idx[start:end]
                b_obs = obs_batch[mb_idx]
                b_actions = act_batch[mb_idx]
                b_old_logp = old_logp_batch[mb_idx]
                b_adv = advantages[mb_idx]
                b_returns = returns[mb_idx]

                new_logp, entropy, values = model.evaluate_actions(b_obs, b_actions)
                ratio = (new_logp - b_old_logp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (b_returns - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    return model
