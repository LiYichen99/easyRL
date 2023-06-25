import numpy as np
from agent import A3CAgent
from agent.configuration import A3CConfig
from agent.model import A3CNet
import gymnasium as gym
import multiprocessing as mp


if __name__ == '__main__':
    env_cls = gym.make
    env_kwargs = dict(id='CartPole-v1', render_mode='human')
    model_cls = A3CNet
    model_kwargs = dict(obs_dim=4, action_dim=2)

    config = A3CConfig(
        actor_critic_model_cls=model_cls,
        actor_critic_model_kwargs=model_kwargs,
        ckpt_path='./A3C_cartpole.ckpt',
        env_cls=env_cls,
        env_kwargs=env_kwargs
    )

    agent_list = [A3CAgent(config) for _ in range(4)]

    p_list = []

    for agent in agent_list:
        p = mp.Process(
            target=agent.train, args=()
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()
