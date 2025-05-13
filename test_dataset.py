import sys
import os
import torch
from diffusion_policy.dataset.track_image_dataset import TrackImageDataset
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

zarr_path = os.path.expanduser('data/pusht/pusht_cchi_v7_replay.zarr')
dataset = PushTImageDataset(zarr_path, horizon=16)
zarr_path = os.path.expanduser('data/track/track_icemao_test_replay.zarr')

# parameters
pred_horizon = 4
obs_horizon = 2
action_horizon = 3
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = TrackImageDataset(
    zarr_path=zarr_path,
    horizon=pred_horizon,
)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['obs']['camera_image'].shape:", batch['obs']['camera_image'].shape)
print("batch['obs']['sonar_image'].shape:", batch['obs']['sonar_image'].shape)
print("batch['action'].shape", batch['action'].shape)