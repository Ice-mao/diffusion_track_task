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
import glob

# left_camera_shape = (3, 224, 224)
# right_camera_shape = (3, 224, 224)
# sonar_shape = (1, 128, 128)
# state_shape = (1)
def get_imitation_episode(folder_path):
    # 创建路径
    action_path = os.path.join(folder_path, "action")
    left_camera_path = os.path.join(folder_path, "left_camera")
    right_camera_path = os.path.join(folder_path, "right_camera")
    sonar_path = os.path.join(folder_path, "sonar")
    state_path = os.path.join(folder_path, "state")

    action_files = sorted(glob.glob(os.path.join(action_path, "step_*.npy")))
    left_camera_files = sorted(glob.glob(os.path.join(left_camera_path, "step_*.jpg")))
    right_camera_files = sorted(glob.glob(os.path.join(right_camera_path, "step_*.jpg")))
    sonar_files = sorted(glob.glob(os.path.join(sonar_path, "step_*.png")))
    state_files = sorted(glob.glob(os.path.join(state_path, "step_*.npy")))
    
    n_steps = len(action_files)
    assert len(left_camera_files) == n_steps, "文件数量不一致"
    assert len(right_camera_files) == n_steps, "文件数量不一致"
    assert len(sonar_files) == n_steps, "文件数量不一致"
    assert len(state_files) == n_steps, "文件数量不一致"

    # 创建空数组来存放数据
    left_camera_imgs = []
    right_camera_imgs = []
    sonar_imgs = []
    states = []
    actions = []

    print(f"正在读取轨迹：{folder_path}")
    for i in range(n_steps):
        # 读取动作数据
        action = np.load(action_files[i])
        actions.append(action)
        
        # 读取相机图像
        left_camera = cv2.imread(left_camera_files[i])
        right_camera = cv2.imread(right_camera_files[i])
        sonar = cv2.imread(sonar_files[i], cv2.IMREAD_GRAYSCALE)
        
        left_camera = cv2.resize(left_camera, (224, 224))
        right_camera = cv2.resize(right_camera, (224, 224))
        sonar = cv2.resize(sonar, (128, 128))
        
        left_camera = cv2.cvtColor(left_camera, cv2.COLOR_BGR2RGB)
        right_camera = cv2.cvtColor(right_camera, cv2.COLOR_BGR2RGB)
        
        left_camera = left_camera.transpose(2, 0, 1)  # (3, 224, 224)
        right_camera = right_camera.transpose(2, 0, 1)  # (3, 224, 224)
        sonar = sonar.reshape(1, 128, 128) 

        state = np.load(state_files[i])
        
        # 将数据添加到列表中
        left_camera_imgs.append(left_camera)
        right_camera_imgs.append(right_camera)
        sonar_imgs.append(sonar)
        states.append(state)

        # 将列表转换为numpy数组
    left_camera_imgs = np.array(left_camera_imgs)
    right_camera_imgs = np.array(right_camera_imgs)
    sonar_imgs = np.array(sonar_imgs)
    states = np.array(states)
    actions = np.array(actions)

    return {
        'left_camera_img': left_camera_imgs,
        'right_camera_img': right_camera_imgs,
        'sonar_img': sonar_imgs,
        'state': states,
        'action': actions
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模拟数据并保存为zarr格式")
    parser.add_argument('--mode', '-m', default='add', choices=['create', 'add'],
                        help='输出的zarr文件路径')
    args = parser.parse_args()
    output_path = "data/track/track_dam_test_replay.zarr"
    
    dataset_path = os.path.join("/data/log/sample/trajs_manual/")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if args.mode == 'create':
        replay_buffer = ReplayBuffer.create_empty_zarr()
    elif args.mode == 'add':
        replay_buffer = ReplayBuffer.copy_from_path(output_path, keys=['left_camera_img', 'right_camera_img', 'sonar_img', 'state', 'action'])
        print(f"原数据集包含 {replay_buffer.n_episodes} 个episodes，共 {replay_buffer.n_steps} 步")
    else:
        assert False, "mode must be create or add"
    
    folders = sorted(glob.glob(os.path.join(dataset_path, "traj_*")))
    for traj_idx in tqdm(folders, desc="episodes traversal"):
        # dataset_path_idx = os.path.join(dataset_path, f"traj_{ep_idx}")
        episode_data = get_imitation_episode(traj_idx)
        replay_buffer.add_episode(
            data=episode_data,
            compressors='disk'
        )
        
    # 保存到指定路径
    replay_buffer.save_to_path(output_path)
    print(f"数据集已保存至: {output_path}")
    print(f"包含 {replay_buffer.n_episodes} 个episodes，共 {replay_buffer.n_steps} 步")