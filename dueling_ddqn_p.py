from agent import DQNAgent
from agent.configuration import DQNConfig
from agent.exploration import LinearExploration
from agent.memory import PriorityMemory
from agent.model import LinearQModel
from experiment_gobang.code.experiment import do_exp

config = DQNConfig(
    eval_model=LinearQModel(450, 255, 512, dueling=True),
    target_model=LinearQModel(450, 255, 512, dueling=True),
    ckpt_path='../checkpoint/dueling_ddqn_p.ckpt',
    batch_size=32,
    lr=0.00001,
    gamma=0.99,
    target_replace_frequency=100,
    capacity=50000,
    max_grad_norm=1.0,
    exploration=LinearExploration(1.0, 0.1, 0.995),
    ddqn=True,
    memory=PriorityMemory(50000)
)
agent = DQNAgent(config)
do_exp(agent, 15, 1000, 50, 50, '../data/dueling_ddqn_p')
