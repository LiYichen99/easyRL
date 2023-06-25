import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import PPOAgent
from agent.configuration import PPOConfig
from agent.model import LinearActor, LinearCritic
import gym
from loguru import logger


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    env = gym.make('CartPole-v1')
    actor_model = LinearActor(
        obs_dim=4,
        hidden_dim=256,
        action_dim=2
    )
    critic_model = LinearCritic(
        obs_dim=4,
        hidden_dim=256
    )
    config = PPOConfig(
        actor_model=actor_model,
        critic_model=critic_model,
        ckpt_path='./PPO_CartPole_v1.ckpt',
        actor_lr=0.001,
        critic_lr=0.001,
        gamma=0.99,
        clip_eps=0.2,
        batch_size=5,
        learn_epochs=4
    )
    agent = PPOAgent(config)

    episode_reward_sum_list = []
    epochs = 500
    for i in range(epochs):
        loss_list, reward_sum = agent.train(env)
        episode_reward_sum_list.append(np.array([i, reward_sum]))
        logger.info(f'Episode {i}/{epochs}, average loss: {np.mean(loss_list) if loss_list else None}, reward sum: {reward_sum}')

    env.close()

    episode_reward_sum_list = np.array(episode_reward_sum_list)

