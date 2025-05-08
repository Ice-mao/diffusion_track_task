if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import numpy as np
import zarr
import argparse
from diffusion_policy.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import time
import math

import datasets
from imitation.data import huggingface_utils, rollout

left_camera_shape = (3, 224, 224)
right_camera_shape = (3, 224, 224)
sonar_shape = (1, 128, 128)
state = (1)
def get_imitation_episode(transition):
    obs = transition.obs[:-1] # in imitation, obs is longer 1 step than actions
    assert obs.shape[0] == transition.acts.shape[0], "obs and acts shape mismatch"
    assert obs.shape[1] == math.prod(left_camera_shape) + math.prod(right_camera_shape) + \
        math.prod(sonar_shape) + state, "obs self-mismatch"
    
    left_camera_img_size = math.prod(left_camera_shape)
    left_camera_img = obs[:, :left_camera_img_size].reshape(-1, *left_camera_shape)
    
    right_camera_img_size = math.prod(right_camera_shape)
    right_camera_img = obs[:, left_camera_img_size:left_camera_img_size + left_camera_img_size].reshape(-1, *right_camera_shape)

    sonar_size = math.prod(sonar_shape)
    sonar_offset = left_camera_img_size + right_camera_img_size
    sonar_img = obs[:, sonar_offset:sonar_offset+sonar_size].reshape(-1, *sonar_shape)

    return {
        'left_camera_img': left_camera_img,
        'right_camera_img': right_camera_img,
        'sonar_img': sonar_img,
        'state': state,
        'action': transition.acts
    }

def create_dataset(output_path, raw_dataset):
    """创建包含多个episode的数据集并保存为zarr格式"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建ReplayBuffer
    replay_buffer = ReplayBuffer.create_empty_zarr()
    
    for ep_idx in tqdm(range(len(raw_dataset)), desc="episodes traversal"):
        # 生成一个轨迹
        episode_data = get_imitation_episode(raw_dataset[ep_idx])
        # 添加到replay buffer
        replay_buffer.add_episode(
            data=episode_data,
            compressors='disk'  # 使用适合磁盘存储的压缩方式
        )
        
    # 保存到指定路径
    replay_buffer.save_to_path(output_path)
    print(f"数据集已保存至: {output_path}")
    print(f"包含 {replay_buffer.n_episodes} 个episodes，共 {replay_buffer.n_steps} 步")
    
    # 返回统计信息
    return {
        'n_episodes': replay_buffer.n_episodes,
        'n_steps': replay_buffer.n_steps,
        'episode_lengths': replay_buffer.episode_lengths
    }

def add_episode_existed_dataset(output_path, raw_dataset):
    """添加新的episode到已存在的数据集"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建ReplayBuffer
    replay_buffer = ReplayBuffer.copy_from_path(output_path, keys=['left_camera_img', 'right_camera_img', 'sonar_img', 'state', 'action'])
    print(f"原数据集包含 {replay_buffer.n_episodes} 个episodes，共 {replay_buffer.n_steps} 步")

    for ep_idx in tqdm(range(len(raw_dataset)), desc="episodes traversal"):
        # 生成一个轨迹
        episode_data = get_imitation_episode(raw_dataset[ep_idx])
        # 添加到replay buffer
        replay_buffer.add_episode(
            data=episode_data,
            compressors='disk'  # 使用适合磁盘存储的压缩方式
        )
        
    # 保存到指定路径
    replay_buffer.save_to_path(output_path)
    print(f"数据集已保存至: {output_path}")
    print(f"包含 {replay_buffer.n_episodes} 个episodes，共 {replay_buffer.n_steps} 步")
    
    # 返回统计信息
    return {
        'n_episodes': replay_buffer.n_episodes,
        'n_steps': replay_buffer.n_steps,
        'episode_lengths': replay_buffer.episode_lengths
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模拟数据并保存为zarr格式")
    parser.add_argument('--mode', '-m', default='create', choices=['create', 'add'],
                        help='输出的zarr文件路径')
    args = parser.parse_args()
    output_path = "data/test/sim_data_replay.zarr"
    
    dataset_path = os.path.join("/data/RL/log/sample/trajs_dam_v2/")
    dataset_0 = datasets.load_from_disk(dataset_path+"traj_0")
    dataset_1 = datasets.load_from_disk(dataset_path+"traj_1")
    dataset = datasets.concatenate_datasets([dataset_0, dataset_1])
    transitions = huggingface_utils.TrajectoryDatasetSequence(dataset)
    # 创建数据集
    if args.mode == 'create':
        stats = create_dataset(output_path, transitions)
    elif args.mode == 'add':
        stats = add_episode_existed_dataset(output_path, transitions)
    else:
        assert False, "mode must be create or add"
    print(stats)