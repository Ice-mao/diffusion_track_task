import torch
import torch.nn as nn
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np
import gdown
import os

# 定义观察和预测的时间步长
obs_horizon = 2  # 根据代码上下文，这里使用2个观察步骤
pred_horizon = 16  # 根据配置文件中的horizon设置


# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
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

# demo
with torch.no_grad():
    # example inputs
    image = torch.zeros((1, obs_horizon,3,96,96))
    agent_pos = torch.zeros((1, obs_horizon, 2))
    # vision encoder
    image_features = nets['vision_encoder'](
        image.flatten(end_dim=1))
    # (2,512)
    image_features = image_features.reshape(*image.shape[:2],-1)
    # (1,2,512)
    obs = torch.cat([image_features, agent_pos],dim=-1)
    # (1,2,514)

    noised_action = torch.randn((1, pred_horizon, action_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = nets['noise_pred_net'](
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)



########### Training ###########
from tqdm import tqdm
# from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusers.training_utils import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from test_dataset import dataloader
num_epochs = 100

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
# ema = EMAModel(
#     model=nets,
#     power=0.75)

ema = EMAModel(
    nets,
    inv_gamma=1.0,                 # 控制预热期的长度
    power=0.75,                    # 与原代码一致
    # 不需要指定model_cls和model_config，因为我们不使用内部模型复制功能
)
# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

for param in nets.parameters():
    param.requires_grad = True
nets.train()
with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nimage = nbatch['obs']['image'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['obs']['agent_pos'][:,:obs_horizon].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                # encoder vision features
                image_features = nets['vision_encoder'](
                    nimage.flatten(end_dim=1))
                image_features = image_features.reshape(
                    *nimage.shape[:2],-1)
                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = nets['noise_pred_net'](
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)


                # 检查损失是否有梯度函数
                if not hasattr(loss, 'grad_fn'):
                    print("警告：损失没有梯度函数！")
                    # 尝试显式创建一个需要梯度的新变量
                    dummy = noise_pred.sum()
                    loss = nn.functional.mse_loss(noise_pred.detach(), noise) + 0 * dummy
                
                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(nets)

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# 保存EMA模型的权重
ckpt_path = "pusht_vision_100ep.ckpt"
torch.save(nets.state_dict(), ckpt_path)
print(f'模型已保存至 {ckpt_path}')

# Weights of the EMA model
# is used for inference
# ema_nets = nets
# ema.copy_to(ema_nets.parameters())
# 创建一个与原始模型结构相同的新模型用于推理
import copy

ema_nets = copy.deepcopy(nets)

# 应用EMA权重进行推理
ema.copy_to(ema_nets.parameters())

# 保存EMA模型
ema_ckpt_path = "pusht_vision_100ep_ema.ckpt"
torch.save(ema_nets.state_dict(), ema_ckpt_path)
print(f'EMA模型已保存至 {ema_ckpt_path}')
