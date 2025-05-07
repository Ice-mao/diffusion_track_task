import sys
import os
import torch
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

zarr_path = os.path.expanduser('data/pusht/pusht_cchi_v7_replay.zarr')

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = PushTImageDataset(
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
print("batch['obs']['image'].shape:", batch['obs']['image'].shape)
print("batch['obs']['agent_pos'].shape:", batch['obs']['agent_pos'].shape)
print("batch['action'].shape", batch['action'].shape)