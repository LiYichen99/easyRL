import random

from environment.gobang import WrapperGoBang
from agent.configuration import DQNConfig
from agent.exploration import LinearExploration
from agent import DDQNAgent, Agent
from agent.model import ConvQModel
from loguru import logger
import numpy as np


class GoBangRandomAgent(Agent):
    def step(self, obs, explore=True, legal_actions=None):
        return random.choice(legal_actions)


if __name__ == '__main__':
    env = WrapperGoBang(board_size=7)
    eval_model = ConvQModel(
        in_channels=2,
        in_h=7,
        in_w=7,
        hidden_channels=16,
        output_dim=49,
        dueling=True
    )
    target_model = ConvQModel(
        in_channels=2,
        in_h=7,
        in_w=7,
        hidden_channels=16,
        output_dim=49,
        dueling=True
    )
    exploration = LinearExploration(
        init_epsilon=1.0,
        min_epsilon=0.1,
        epsilon_decay=0.995
    )
    config = DQNConfig(
        eval_model=eval_model,
        target_model=target_model,
        ckpt_path='./DDQN_gobang.ckpt',
        batch_size=32,
        lr=0.0001,
        gamma=0.95,
        target_replace_frequency=100,
        capacity=5000,
        max_grad_norm=1.0,
        exploration=exploration
    )
    agent = DDQNAgent(config)

    epochs = 100000
    for i in range(epochs):
        loss_list, reward_sum = agent.train(env)
        logger.info(
            f'Episode {i}/{epochs}, average loss: {np.mean(loss_list) if loss_list else -1}, reward sum: {reward_sum}')
        if i % 100 == 0:
            win = 0
            draw = 0
            lose = 0
            for _ in range(50):
                done = agent.test(env, GoBangRandomAgent(None), display=False)
                if done == 1:
                    win += 1
                elif done == -1:
                    draw += 1
                else:
                    lose += 1
            logger.info(
                f'Episode {i}/{epochs}, win {win}/{50}, draw {draw}/{50}, lose {lose}/{50}')
        if i % 50 == 0:
            agent.test(env, GoBangRandomAgent(None), display=True)
