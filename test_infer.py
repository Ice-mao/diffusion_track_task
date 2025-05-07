import torch
import torch.nn as nn
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
import numpy as np
import gdown
import os
from tqdm import tqdm
import collections
import imageio

# 添加设备定义
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 添加缺失的变量定义
pred_horizon = 16  # 预测的动作步数
action_horizon = 1  # 每次执行的动作步数
num_diffusion_iters = 100  # 扩散模型的迭代次数

# 定义数据标准化和反标准化函数
def normalize_data(data, stats):
    return (data - stats['mean']) / stats['std']

def unnormalize_data(data, stats):
    return data * stats['std'] + stats['mean']

# 定义数据统计信息
stats = {
    'action': {
        'mean': np.array([0.0, 0.0]),
        'std': np.array([1.0, 1.0])
    },
    'agent_pos': {
        'mean': np.array([0.0, 0.0]),
        'std': np.array([1.0, 1.0])
    }
}

# 初始化噪声调度器
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=True,
    prediction_type="epsilon"
)

# 视频写入函数
def vwrite(filename, images):
    imageio.mimsave(filename, images, fps=30)

vision_encoder = get_resnet('resnet18', "IMAGENET1K_V1")

# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder = replace_submodules(
    root_module=vision_encoder,
    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
    func=lambda x: nn.GroupNorm(
        num_groups=x.num_features//16, 
        num_channels=x.num_features)
)

# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2
obs_horizon = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

load_pretrained = True
if load_pretrained:
  ckpt_path = "pusht_vision_100ep.ckpt"
  if not os.path.isfile(ckpt_path):
      id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
      gdown.download(id=id, output=ckpt_path, quiet=False)

  state_dict = torch.load(ckpt_path, map_location='cuda')
  ema_nets = nets
  ema_nets.load_state_dict(state_dict)
  print('Pretrained weights loaded.')
else:
  print("Skipped pretrained weight loading.")

ema_nets.to(device)
# limit enviornment interaction to 200 steps before termination
max_steps = 200
env = PushTImageEnv()
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(100000)

# get first observation
obs = env.reset()

# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        images = np.stack([x['image'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

        # normalize observation
        nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
        # images are already normalized to [0,1]
        nimages = images

        # device transfer
        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
        # (2,3,96,96)
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
        # (2,2)

        # infer action
        with torch.no_grad():
            # get image features
            image_features = ema_nets['vision_encoder'](nimages)
            # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, info = env.step(action[i])
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print('Score: ', max(rewards))

# visualize
from IPython.display import Video
vwrite('vis.mp4', imgs)
Video('vis.mp4', embed=True, width=256, height=256)