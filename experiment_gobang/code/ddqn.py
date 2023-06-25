from agent import DQNAgent
from agent.model import ConvQModel, LinearQModel
from agent.configuration import DQNConfig
from agent.exploration import LinearExploration
from experiment_gobang.code.experiment import do_exp

if __name__ == '__main__':
    eval_model = LinearQModel(
        input_dim=450,
        output_dim=225,
        hidden_dim=512,
        dueling=False
    )
    target_model = LinearQModel(
        input_dim=450,
        output_dim=225,
        hidden_dim=512,
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
        ckpt_path='../checkpoint/ddqn.ckpt',
        batch_size=32,
        lr=0.00005,
        gamma=0.99,
        target_replace_frequency=100,
        capacity=5000,
        max_grad_norm=1.0,
        exploration=exploration,
        ddqn=True
    )
    agent = DQNAgent(config)
    do_exp(agent, 15, 1000, 50, 50, '../data/ddqn')
