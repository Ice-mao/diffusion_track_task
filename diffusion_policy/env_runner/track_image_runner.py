import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class TrackImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
        ):
        super().__init__(output_dir)

        self.env = env
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        # n_envs = len(self.env_fns)
        # n_inits = len(self.env_init_fn_dills)
        # n_chunks = math.ceil(n_inits / n_envs)

        # start rollout
        obs, _ = env.reset()
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False
        while not done:
            # create obs dict
            np_obs_dict = dict(obs)
            # if self.past_action and (past_action is not None):
            #     # TODO: not tested
            #     np_obs_dict['past_action'] = past_action[
            #         :,-(self.n_obs_steps-1):].astype(np.float32)
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']

            # step env
            obs, reward, done, info = env.step(action)
            done = np.all(done)
            past_action = action

            # update pbar
            pbar.update(action.shape[1])
        pbar.close()

        all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
