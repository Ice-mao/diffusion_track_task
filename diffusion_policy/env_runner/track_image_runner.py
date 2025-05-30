if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
# import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
# from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

import auv_env
import gymnasium as gym

class TrackImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            max_steps=200,
            n_obs_steps=2,
            n_action_steps=8,
        ):
        super().__init__(output_dir)

        self.env = gym.make("v2-sample-student")
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        obs_history = []

        # start rollout
        raw_obs, _ = env.reset()
        policy.reset()

        # 将初始观测添加到历史队列
        parsed_obs = parse_observation(raw_obs)
        obs_history.append(parsed_obs)
        
        # 初始填充历史队列至所需长度
        for _ in range(self.n_obs_steps - 1):
            obs_history.append(parsed_obs)  # 重复使用第一帧填充
        
        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval Visualization")
        done = False
        step_count = 0
        total_reward = 0

        while not done and step_count < self.max_steps:
            # 构建模型输入格式
            np_obs_dict = {
                'camera_image': np.stack([o['camera_image'] for o in obs_history]),
                'sonar_image': np.stack([o['sonar_image'] for o in obs_history]),
                'state': np.stack([o['state'] for o in obs_history])
            }
            
            # 转为tensor，添加batch维度
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=dtype))

            # 运行策略
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # 从device转回CPU
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action'][0]  # 去掉batch维度

            # 执行环境步骤
            for i in range(self.n_action_steps):
                raw_obs, reward, done, _, info = env.step(action[i,:])

                # 更新观测历史队列
                if not done:
                    parsed_obs = parse_observation(raw_obs)
                    obs_history.pop(0)  # 移除最旧的观测
                    obs_history.append(parsed_obs)  # 添加新观测
            
            print('reward:', reward)
            total_reward += reward
            # 更新进度条
            pbar.update(1)
            step_count += 1
        pbar.close()

        log_data = {
            'total_reward': reward,
            'steps': step_count
        }

        return log_data
    
def parse_observation(obs):
    left_camera_shape = (3, 224, 224)
    right_camera_shape = (3, 224, 224)
    sonar_shape = (1, 128, 128)
    state_shape = (1)
    
    left_camera_img_size = math.prod(left_camera_shape)
    left_camera_img = obs[:left_camera_img_size].reshape(*left_camera_shape)
    
    right_camera_img_size = math.prod(right_camera_shape)
    right_camera_img = obs[left_camera_img_size:left_camera_img_size + left_camera_img_size].reshape(*right_camera_shape)

    sonar_size = math.prod(sonar_shape)
    sonar_offset = left_camera_img_size + right_camera_img_size
    sonar_img = obs[sonar_offset:sonar_offset+sonar_size].reshape(*sonar_shape)

    state = obs[-1:]

    # 归一化图像数据
    left_camera_img = left_camera_img/255
    # right_camera_img = right_camera_img/255
    sonar_img = sonar_img/255
    
    return {
        'camera_image': left_camera_img,
        'sonar_image': np.repeat(sonar_img, 3, axis=0),  # (3, 128, 128)
        'state': state.reshape(state_shape)
    }
