import random

from environment.gobang import WrapperGoBang
from agent.model import ConvQModel
from agent.configuration import DQNConfig
from agent.exploration import LinearExploration
from agent import DQNAgent, Agent
from loguru import logger
import numpy as np



class GoBangRandomAgent(Agent):
    def step(self, obs, explore=True, legal_actions=None):
        return random.choice(legal_actions)


if __name__ == '__main__':
    env = WrapperGoBang(board_size=15)
    eval_model = ConvQModel(
        in_channels=2,
        in_h=15,
        in_w=15,
        hidden_channels=64,
        output_dim=225,
        dueling=False
    )
    target_model = ConvQModel(
        in_channels=2,
        in_h=15,
        in_w=15,
        hidden_channels=64,
        output_dim=225,
        dueling=False
    )
    exploration = LinearExploration(
        init_epsilon=1.0,
        min_epsilon=0.1,
        epsilon_decay=0.995
    )
    config = DQNConfig(
        eval_model=eval_model,
        target_model=target_model,
        ckpt_path='./DQN_gobang.ckpt',
        batch_size=32,
        lr=0.001,
        gamma=0.99,
        target_replace_frequency=100,
        capacity=5000,
        max_grad_norm=1.0,
        exploration=exploration
    )
    agent = DQNAgent(config)

    epochs = 100000
    for i in range(epochs):
        loss_list, reward_sum = agent.train(env)
        logger.info(
            f'Episode {i}/{epochs}, average loss: {np.mean(loss_list) if loss_list else -1}, reward sum: {reward_sum}')
        if i % 40 == 0:
            win = 0
            for _ in range(10):
                win += agent.test(env, GoBangRandomAgent(None), display=False)
            logger.info(
                f'Episode {i}/{epochs}, win {win}/{10}')
        if i % 50 == 0:
            agent.test(env, GoBangRandomAgent(None), display=True)
