from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ObservationDataset(Dataset):
    """Dataset for loading expert observations and actions."""

    def __init__(
        self, obs_list, action_list, normalize=True, feature_mean=None, feature_std=None
    ):
        self.obs_list = obs_list
        self.action_list = action_list
        self.normalize = normalize

        # Compute normalization statistics if not provided
        if normalize and feature_mean is None:
            all_features = []
            for obs in obs_list:
                features = self._extract_features(obs)
                all_features.append(features)
            all_features = np.array(all_features, dtype=np.float32)
            self.feature_mean = all_features.mean(axis=0)
            self.feature_std = (
                all_features.std(axis=0) + 1e-8
            )  # Add small epsilon to avoid division by zero
        else:
            self.feature_mean = (
                feature_mean if feature_mean is not None else np.zeros(10)
            )
            self.feature_std = feature_std if feature_std is not None else np.ones(10)

    def __len__(self):
        return len(self.action_list)

    def __getitem__(self, idx):
        obs = self.obs_list[idx]
        action = self.action_list[idx]

        # Extract relevant features from observation dict
        features = self._extract_features(obs)

        # Normalize features
        if self.normalize:
            features = (features - self.feature_mean) / self.feature_std

        return torch.FloatTensor(features), torch.FloatTensor(action)

    @staticmethod
    def _extract_features(obs):
        """Extract a flat feature vector from observation dict."""
        features = []

        # Car state features
        features.append(float(obs["car"]["position"][0]))
        features.append(float(obs["car"]["position"][1]))
        features.append(float(obs["car"]["heading"][0]))
        features.append(float(obs["car"]["speed"][0]))

        # Path-relative features
        features.append(float(obs["path"]["ey"][0]))  # lateral error
        features.append(float(obs["path"]["epsi"][0]))  # heading error
        features.append(float(obs["path"]["s"][0]))  # arclength
        features.append(float(obs["path"]["kappa"][0]))  # curvature
        features.append(float(obs["path"]["v_long"][0]))  # longitudinal velocity
        features.append(float(obs["path"]["v_lat"][0]))  # lateral velocity

        return np.array(features, dtype=np.float32)


def create_data_loaders(
    batch_size,
    train_split,
    data_dir="expert_data",
):
    """
    Load expert data and create train/validation DataLoaders.

    This function handles:
    - Loading observations and actions from .npy files
    - Data validation
    - Feature normalization statistics computation
    - Train/validation split
    - DataLoader creation

    Args:
        data_dir: Directory containing obs.npy and action.npy files
        batch_size: Batch size for DataLoaders
        train_split: Fraction of data to use for training (rest for validation)

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        feature_mean: Mean of normalized features (for inference)
        feature_std: Std of normalized features (for inference)
    """
    # Load expert data
    obs_path = Path(data_dir) / "obs.npy"
    action_path = Path(data_dir) / "action.npy"

    if not obs_path.exists() or not action_path.exists():
        raise FileNotFoundError(
            f"Expert data not found. Expected {obs_path} and {action_path}"
        )

    print(f"Loading expert data from {data_dir}...")
    obs_list = np.load(obs_path, allow_pickle=True)
    action_list = np.load(action_path, allow_pickle=True)

    # Validate data for imitation learning
    if len(obs_list) != len(action_list):
        raise ValueError(
            f"Mismatch: {len(obs_list)} observations but {len(action_list)} actions. "
            "Each observation must have a corresponding action for imitation learning."
        )

    print(
        f"Loaded {len(action_list)} expert (observation, action) pairs for imitation learning"
    )

    # Create full dataset first to compute normalization stats
    full_dataset = ObservationDataset(obs_list, action_list, normalize=True)

    # Split into train/validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_split_obj, val_split_obj = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Extract indices and create separate datasets with same normalization stats
    train_indices = train_split_obj.indices
    val_indices = val_split_obj.indices

    train_dataset = ObservationDataset(
        [obs_list[i] for i in train_indices],
        [action_list[i] for i in train_indices],
        normalize=True,
        feature_mean=full_dataset.feature_mean,
        feature_std=full_dataset.feature_std,
    )
    val_dataset = ObservationDataset(
        [obs_list[i] for i in val_indices],
        [action_list[i] for i in val_indices],
        normalize=True,
        feature_mean=full_dataset.feature_mean,
        feature_std=full_dataset.feature_std,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader