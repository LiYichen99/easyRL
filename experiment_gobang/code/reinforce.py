from experiment_gobang.code.experiment import do_exp
from agent import ReinforceAgent
from agent.model import *
from agent.configuration import PGConfig

model = LinearPolicyModel(
    input_dim=450,
    output_dim=225,
    hidden_dim=512
)
config = PGConfig(
    policy_model=model,
    ckpt_path='../checkpoint/reinforce',
    lr=0.001,
    gamma=0.99
)
agent = ReinforceAgent(config)
do_exp(agent, 15, 1000, 50, 50, '../data/reinforce')
