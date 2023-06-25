import ml_collections
import torch.nn
from agent.memory import *
from agent.exploration import *
from agent.optimizer import SharedAdam


class Config(object):

    def __init__(self):
        self.config = ml_collections.ConfigDict()
        self.available_keys = []

    def set(self, key, value):
        self.config[key] = value

    def get(self, key):
        return self.config.get(key, None)

    def to_json(self):
        return self.config.to_json()

    def to_yaml(self):
        return self.config.to_yaml()

    def to_dict(self):
        return self.config.to_dict()

    def __str__(self):
        return str(self.config)

    def __repr__(self):
        return str(self.config)


class DQNConfig(Config):

    def __init__(self,
                 eval_model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 ckpt_path: str,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 gamma: float = 0.95,
                 target_replace_frequency: int = 100,
                 capacity: int = 20000,
                 max_grad_norm: float = 1.0,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs
                 ):
        super(DQNConfig, self).__init__()
        self.available_keys = [
            'memory',
            'optimizer',
            'eval_model',
            'target_model',
            'exploration',
            'gamma',
            'lr'
            'target_replace_frequency',
            'criterion',
            'ckpt_path',
            'batch_size'
            'device',
            'max_grad_norm',
            'ddqn',
        ]

        self.set('eval_model', eval_model.to(device))
        self.set('target_model', target_model.to(device))
        self.set('ckpt_path', ckpt_path)
        self.set('batch_size', batch_size)
        self.set('lr', lr)
        self.set('gamma', gamma)
        self.set('target_replace_frequency', target_replace_frequency)
        self.set('capacity', capacity)
        self.set('device', device)
        self.set('max_grad_norm', max_grad_norm)

        self.set('optimizer', torch.optim.Adam(eval_model.parameters(), lr=lr) if 'optimizer' not in kwargs.keys() else kwargs.get('optimizer'))
        self.set('criterion', torch.nn.MSELoss() if 'criterion' not in kwargs.keys() else kwargs.get('criterion'))
        self.set('exploration', ConstExploration(0.1) if 'exploration' not in kwargs.keys() else kwargs.get('exploration'))
        self.set('memory', QueueMemory(capacity) if 'memory' not in kwargs.keys() else kwargs.get('memory'))
        self.set('ddqn', False)


class PGConfig(Config):

    def __init__(self,
                 policy_model: torch.nn.Module,
                 ckpt_path: str,
                 lr: float = 1e-3,
                 gamma: float = 0.95,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs):
        super(PGConfig, self).__init__()
        self.available_keys = [
            'optimizer',
            'policy_model',
            'gamma',
            'lr'
            'ckpt_path',
            'device',
        ]
        self.set('policy_model', policy_model.to(device))
        self.set('lr', lr)
        self.set('gamma', gamma)
        self.set('ckpt_path', ckpt_path)
        self.set('device', device)

        self.set('optimizer', torch.optim.Adam(policy_model.parameters(), lr=lr) if 'optimizer' not in kwargs.keys() else kwargs.get('optimizer'))
        self.set('memory', EpisodeMemory())


class PPOConfig(Config):

    def __init__(self,
                 actor_model: torch.nn.Module,
                 critic_model: torch.nn.Module,
                 ckpt_path: str,
                 actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.95,
                 clip_eps: float = 0.2,
                 batch_size: int = 5,
                 learn_epochs: int = 4,
                 update_steps: int = 20,
                 gae_lambda: float = 0.95,
                 actor_loss_factor: float = 1,
                 critic_loss_factor: float = 0.5,
                 entropy_loss_factor: float = 0.01,
                 actor_max_norm: float = 0.5,
                 critic_max_norm: float = 0.5,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs):
        super(PPOConfig, self).__init__()
        self.available_keys = [
            'actor_model',
            'critic_model',
            'ckpt_path',
            'actor_lr',
            'critic_lr',
            'clip_eps',
            'gae_lambda',
            'batch_size',
            'learn_epochs',
            'gamma',
            'actor_optimizer',
            'critic_optimizer'
            'device',
            'actor_loss_factor',
            'critic_loss_factor',
            'entropy_loss_factor',
            'actor_max_norm',
            'critic_max_norm',
            'update_steps'
        ]

        self.set('actor_model', actor_model.to(device))
        self.set('critic_model', critic_model.to(device))
        self.set('actor_lr', actor_lr)
        self.set('critic_lr', critic_lr)
        self.set('ckpt_path', ckpt_path)
        self.set('clip_eps', clip_eps)
        self.set('gamma', gamma)
        self.set('batch_size', batch_size)
        self.set('learn_epochs', learn_epochs)
        self.set('device', device)
        self.set('gae_lambda', gae_lambda)
        self.set('actor_loss_factor', actor_loss_factor)
        self.set('critic_loss_factor', critic_loss_factor)
        self.set('entropy_loss_factor', entropy_loss_factor)
        self.set('actor_max_norm', actor_max_norm)
        self.set('critic_max_norm', critic_max_norm)
        self.set('update_steps', update_steps)

        self.set('actor_optimizer', torch.optim.Adam(actor_model.parameters(), lr=actor_lr) if 'actor_optimizer' not in kwargs.keys() else kwargs.get('actor_optimizer'))
        self.set('critic_optimizer', torch.optim.Adam(critic_model.parameters(), lr=critic_lr) if 'critic_optimizer' not in kwargs.keys() else kwargs.get('critic_optimizer'))
        self.set('memory', EpisodeMemory())


class A2CConfig(Config):
    def __init__(self,
                 actor_critic_model: torch.nn.Module,
                 ckpt_path: str,
                 lr: float = 1e-3,
                 gamma: float = 0.95,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs):
        super(A2CConfig, self).__init__()
        self.available_keys = [
            'actor_critic_model',
            'ckpt_path',
            'lr',
            'gamma',
            'optimizer',
            'device'
        ]
        self.set('actor_critic_model', actor_critic_model.to(device))
        self.set('ckpt_path', ckpt_path)
        self.set('lr', lr)
        self.set('gamma', gamma)
        self.set('device', device)

        self.set('optimizer', torch.optim.Adam(actor_critic_model.parameters(), lr=lr) if 'optimizer' not in kwargs.keys() else kwargs.get('optimizer'))


class A3CConfig(Config):
    def __init__(self,
                 actor_critic_model_cls: torch.nn.Module,
                 actor_critic_model_kwargs: dict,
                 ckpt_path: str,
                 env_cls,
                 env_kwargs: dict,
                 train_epochs: int = 300,
                 lr: float = 0.01,
                 gamma: float = 0.95,
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        super(A3CConfig, self).__init__()
        self.available_keys = [
            'actor_critic_model_cls',
            'actor_critic_model_kwargs',
            'ckpt_path',
            'lr',
            'gamma',
            'optimizer',
            'device'
        ]
        global_actor_critic_model = actor_critic_model_cls(**actor_critic_model_kwargs).share_memory()
        self.set('actor_critic_model_cls', actor_critic_model_cls)
        self.set('actor_critic_model_kwargs', actor_critic_model_kwargs)
        self.set('global_actor_critic_model', global_actor_critic_model)
        global_optimizer = SharedAdam(global_actor_critic_model.parameters(), lr=lr)
        global_optimizer.share_memory()
        self.set('global_optimizer', global_optimizer)
        self.set('ckpt_path', ckpt_path)
        self.set('gamma', gamma)
        self.set('device', device)
        self.set('train_epochs', train_epochs)
        self.set('env_cls', env_cls)
        self.set('env_kwargs', env_kwargs)
        self.set('memory', EpisodeMemory())