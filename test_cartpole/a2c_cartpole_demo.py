import numpy as np
from agent import A2CAgent
from agent.configuration import A2CConfig
from agent.model import ActorCritic
import gym
from loguru import logger


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    actor_critic_model = ActorCritic(
        obs_dim=4,
        hidden_dim=256,
        action_dim=2
    )
    config = A2CConfig(
        actor_critic_model=actor_critic_model,
        ckpt_path='./PPO_CartPole_v1.ckpt',
        lr=0.0005,
        gamma=0.95
    )
    agent = A2CAgent(config)

    epochs = 500
    for i in range(epochs):
        loss_list, reward_sum = agent.train(env)
        logger.info(f'Episode {i}/{epochs}, average loss: {np.mean(loss_list) if loss_list else None}, reward sum: {reward_sum}')

    env.close()
