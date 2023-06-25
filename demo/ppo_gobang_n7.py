import random

from environment.gobang import WrapperGoBang
from agent.configuration import PPOConfig
from agent.exploration import LinearExploration
from agent import PPOAgent, Agent
from agent.model.ac_model import ConvActor, ConvCritic
from loguru import logger
import numpy as np

import torch.nn as nn


class GoBangRandomAgent(Agent):
    def step(self, obs, explore=True, legal_actions=None):
        return {'action': random.choice(legal_actions)}


if __name__ == '__main__':
    env = WrapperGoBang(board_size=7)
    actor_model = ConvActor(
        in_channels=2,
        in_h=7,
        in_w=7,
        hidden_channels=16,
        action_dim=49
    )
    critic_model = ConvCritic(
        in_channels=2,
        in_h=7,
        in_w=7,
        hidden_channels=16
    )
    exploration = LinearExploration(
        init_epsilon=1.0,
        min_epsilon=0.1,
        epsilon_decay=0.995
    )
    config = PPOConfig(
        actor_model=actor_model,
        critic_model=critic_model,
        ckpt_path='./PPO_gobang_n7',
    )
    agent = PPOAgent(config)

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
