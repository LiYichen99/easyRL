from agent.configuration import PPOConfig
from experiment_gobang.code.experiment import do_exp
from agent import PPOAgent
from agent.model import *

actor_model = LinearActor(
    obs_dim=450,
    action_dim=225,
    hidden_dim=512
)
critic_model = LinearCritic(
    obs_dim=450,
    hidden_dim=512
)
config = PPOConfig(
    actor_model=actor_model,
    critic_model=critic_model,
    ckpt_path='../checkpoint/ppo',
    actor_lr=0.0003,
    critic_lr=0.0003,
    gamma=0.95,
    clip_eps=0.2,
    gae_lambda=0.95,
    batch_size=32,
    learn_epochs=10,
    update_steps=1024
)
agent = PPOAgent(config)
do_exp(agent, 15, 1000, 50, 50, '../data/experiment/ppo')
