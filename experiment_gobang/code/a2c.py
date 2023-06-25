from agent.configuration import A2CConfig
from agent.model import ActorCritic

actor_critic_model = ActorCritic(
    obs_dim=450,
    hidden_dim=512,
    action_dim=225
)
config = A2CConfig(
    actor_critic_model=actor_critic_model,
    ckpt_path='../checkpoint/ppo',

)