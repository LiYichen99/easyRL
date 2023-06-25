import os

import numpy as np
import torch
from loguru import logger

from agent import RandomAgent
from environment.gobang import WrapperGoBang

random_agent = RandomAgent()

def do_exp(agent, n, epochs, test_freq, test_n, result_path):
    # torch.manual_seed(42)
    # np.random.seed(42)

    loss_episode = []
    reward_episode = []
    win_ratio = []
    best_win_ratio = -np.inf
    env = WrapperGoBang(n)
    for i in range(epochs):
        loss_list, reward_sum = agent.train(env)
        logger.info(
            f'Episode {i}/{epochs}, average loss: {np.mean(loss_list) if loss_list else -1}, reward sum: {reward_sum}')
        if loss_list:
            loss_episode.append(np.array([i, np.mean(loss_list)]))
            reward_episode.append(np.array([i, reward_sum]))

        if i % test_freq == 0:
            win = 0
            for _ in range(test_n):
                win += agent.test(env, random_agent, display=False)
            logger.info(f'Episode {i}/{epochs}, win {win}/{test_n}')
            ratio = 1.0 * win / test_n
            win_ratio.append(np.array([i, ratio]))
            if ratio >= best_win_ratio:
                agent.save()
    np.save(os.path.join(result_path, 'loss.npy'), np.array(loss_episode))
    np.save(os.path.join(result_path, 'reward.npy'), np.array(reward_episode))
    np.save(os.path.join(result_path, 'ratio.npy'), np.array(win_ratio))
