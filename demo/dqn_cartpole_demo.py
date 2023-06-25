import numpy as np
from agent import DQNAgent
from agent.configuration import DQNConfig
from agent.model import LinearQModel
from agent.memory import QueueMemory
from agent.exploration import LinearExploration
import gymnasium as gym
from loguru import logger


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    eval_model = LinearQModel(
        input_dim=4,
        output_dim=2,
        hidden_dim=64,
        dueling=True
    )
    target_model = LinearQModel(
        input_dim=4,
        output_dim=2,
        hidden_dim=64,
        dueling=True
    )
    exploration = LinearExploration(
        init_epsilon=1.0,
        min_epsilon=0.02,
        epsilon_decay=0.995
    )
    config = DQNConfig(
        eval_model=eval_model,
        target_model=target_model,
        ckpt_path='./DQN_CartPole_v1.ckpt',
        batch_size=32,
        lr=0.01,
        gamma=0.99,
        target_replace_frequency=100,
        capacity=5000,
        max_grad_norm=1.0,
        exploration=exploration,
        ddqn=True,
    )
    agent = DQNAgent(config)

    epochs = 500
    for i in range(epochs):
        loss_list, reward_sum = agent.train(env)
        logger.info(f'Episode {i}/{epochs}, average loss: {np.mean(loss_list) if loss_list else -1}, reward sum: {reward_sum}')

    env.close()