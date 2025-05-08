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

def get_imitation_trajectory(num_steps=100, img_size=96):
    """生成模拟的轨迹数据，包含图像和坐标"""
    # 生成随机的智能体位置轨迹
    agent_pos = np.zeros((num_steps, 2))
    # 起点为随机位置
    agent_pos[0] = np.random.uniform(-0.5, 0.5, size=2)
    
    # 生成随机的连续轨迹
    for i in range(1, num_steps):
        # 随机步长，但保证平滑性
        step = np.random.normal(0, 0.03, size=2)
        agent_pos[i] = agent_pos[i-1] + step
        # 确保位置在合理范围内
        agent_pos[i] = np.clip(agent_pos[i], -1, 1)
    
    # 生成动作（这里简化为位置差分）
    actions = np.zeros((num_steps, 2))
    actions[:-1] = agent_pos[1:] - agent_pos[:-1]
    actions[-1] = actions[-2]  # 最后一个动作重复倒数第二个
    
    # 生成模拟的图像数据
    images = np.zeros((num_steps, img_size, img_size, 3), dtype=np.uint8)
    for i in range(num_steps):
        # 创建简单的彩色背景
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        
        # 将智能体绘制为一个彩色圆点
        x_px = int((agent_pos[i, 0] + 1) / 2 * img_size)
        y_px = int((agent_pos[i, 1] + 1) / 2 * img_size)
        cv2.circle(img, (x_px, y_px), 5, (0, 0, 255), -1)
        
        # 添加一个目标物体（绿色方块）
        target_x = int(img_size * 0.7)
        target_y = int(img_size * 0.7)
        cv2.rectangle(img, (target_x-7, target_y-7), (target_x+7, target_y+7), (0, 255, 0), -1)
        
        images[i] = img
    
    return {
        'camera_img': images,           # RGB图像
        'sonar_img': images,           # RGB图像
        'state': agent_pos,    # 智能体位置
        'action': actions          # 动作
    }

def create_dataset(output_path, num_episodes=10, min_steps=80, max_steps=120, img_size=96):
    """创建包含多个episode的数据集并保存为zarr格式"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建ReplayBuffer
    replay_buffer = ReplayBuffer.create_empty_zarr()
    
    for ep_idx in tqdm(range(num_episodes), desc="生成episodes"):
        # 随机确定这个episode的步数
        num_steps = np.random.randint(min_steps, max_steps+1)
        
        # 生成一个轨迹
        episode_data = generate_random_trajectory(num_steps=num_steps, img_size=img_size)
        
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

def visualize_episode(replay_buffer, episode_idx=0):
    """可视化一个episode的内容"""
    episode = replay_buffer.get_episode(episode_idx)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 显示第一帧图像
    axes[0, 0].imshow(episode['image'][0])
    axes[0, 0].set_title("First Frame")
    axes[0, 0].axis('off')
    
    # 显示最后一帧图像
    axes[0, 1].imshow(episode['image'][-1])
    axes[0, 1].set_title("Last Frame")
    axes[0, 1].axis('off')
    
    # 绘制智能体轨迹
    axes[1, 0].plot(episode['agent_pos'][:, 0], episode['agent_pos'][:, 1], 'b-')
    axes[1, 0].plot(episode['agent_pos'][0, 0], episode['agent_pos'][0, 1], 'go', label='Start')
    axes[1, 0].plot(episode['agent_pos'][-1, 0], episode['agent_pos'][-1, 1], 'ro', label='End')
    axes[1, 0].set_title("Agent Trajectory")
    axes[1, 0].set_xlim(-1, 1)
    axes[1, 0].set_ylim(-1, 1)
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 绘制动作序列
    time = np.arange(len(episode['action']))
    axes[1, 1].plot(time, episode['action'][:, 0], 'r-', label='x action')
    axes[1, 1].plot(time, episode['action'][:, 1], 'g-', label='y action')
    axes[1, 1].set_title("Actions over time")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("Action value")
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模拟数据并保存为zarr格式")
    parser.add_argument('--output', '-o', default='data/test/sim_data_replay.zarr',
                        help='输出的zarr文件路径')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                        help='要生成的episode数量')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='是否可视化生成的数据')
    args = parser.parse_args()
    
    # 创建数据集
    stats = create_dataset(args.output, num_episodes=args.episodes)
    
    # 可视化数据
    if args.visualize:
        replay_buffer = ReplayBuffer.copy_from_path(args.output)
        fig = visualize_episode(replay_buffer)
        plt.show()
        
        # 保存可视化结果
        vis_dir = os.path.join(os.path.dirname(args.output), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        fig.savefig(os.path.join(vis_dir, 'episode_0_visualization.png'))
        plt.close(fig)