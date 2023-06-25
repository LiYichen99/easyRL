import numpy as np
from agent import ReinforceAgent
from agent.configuration import PGConfig
from agent.model import LinearPolicyModel
import gym
from loguru import logger


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    policy_model = LinearPolicyModel(
        input_dim=4,
        output_dim=2,
        hidden_dim=64
    )
    config = PGConfig(
        policy_model=policy_model,
        ckpt_path='./PG_CartPole_v1.ckpt',
        lr=0.01,
        gamma=0.95
    )
    agent = ReinforceAgent(config)

    epochs = 500
    for i in range(epochs):
        loss_list, reward_sum = agent.train(env)
        logger.info(f'Episode {i}/{epochs}, average loss: {np.mean(loss_list) if loss_list else -1}, reward sum: {reward_sum}')

    env.close()
