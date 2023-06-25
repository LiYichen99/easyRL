import random

from agent.configuration import Config
from agent.algorithm import *
from loguru import logger


class Agent(object):

    def __init__(self, config: Config):
        super(Agent, self).__init__()
        self.config = config

    def step(self, obs, explore=True, legal_actions=None):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, env, another_agent, display=False):
        obs, info = env.reset()
        i = 1
        while True:
            if i == 1:
                result = self.step(obs, explore=False, legal_actions=env.get_legal_actions() if callable(
                    getattr(env, 'get_legal_actions', None)) else None)
                action = result.get('action')
                if display:
                    print(f'{str(self)}: {action}')
            else:
                result = another_agent.step(obs, explore=False, legal_actions=env.get_legal_actions() if callable(
                    getattr(env, 'get_legal_actions', None)) else None)
                action = result.get('action')
                if display:
                    print(f'{str(another_agent)}: {action}')
            next_obs, reward, terminate, truncated, info = env.step(action)
            obs = next_obs
            if display:
                env.render()
            if terminate or truncated:
                break
            i = info['current_player']
        if info['winner'] == 0:
            if display:
                print('Draw Done')
            return 0
        elif info['winner'] == 1:
            if display:
                print(f'{str(self)} Done')
            return 1
        else:
            if display:
                print(f'{str(another_agent)} Done')
            return 0

    def save(self):
        pass

    def load(self):
        pass


class RandomAgent(Agent):
    def __init__(self, config=None):
        super(RandomAgent, self).__init__(config)

    def step(self, obs, explore=True, legal_actions=None):
        return {'action': random.choice(legal_actions)}

    def __str__(self):
        return 'Random Agent'


class RuleAgent(Agent):
    def __init__(self, config=None):
        super(RuleAgent, self).__init__(config)
        self.directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    def step(self, obs, explore=True, legal_actions=None):
        for action in legal_actions:
            cx, cy = action // 15, action % 15
            for dx, dy in self.directions:
                x, y = cx + dx, cy + dy
                if 0 <= x < 15 and 0 <= y < 15 and obs[0, x, y] == 1:
                    return {'action': action}
        return {'action': random.choice(legal_actions)}


class DQNAgent(Agent):

    def __init__(self, config: Config):
        super(DQNAgent, self).__init__(config)
        self.dqn = DQN(config)

    def step(self, obs, explore=True, legal_actions=None):
        return self.dqn.get_action(obs, explore, legal_actions)

    def train(self, env):
        loss_list = []
        reward_sum = 0
        obs, info = env.reset()
        while True:
            result = self.step(obs, explore=True, legal_actions=env.get_legal_actions() if callable(
                getattr(env, 'get_legal_actions', None)) else None)
            action = result.get('action')
            next_obs, reward, terminate, truncated, info = env.step(action)
            self.dqn.store_sample([obs, action, reward, next_obs, terminate or truncated])
            obs = next_obs
            reward_sum += reward
            if self.dqn.can_learn():
                loss = self.dqn.learn()
                loss_list.append(loss)
            if terminate or truncated:
                return loss_list, reward_sum

    def save(self):
        self.dqn.save_checkpoint()

    def load(self):
        self.dqn.load_checkpoint()

    def __str__(self):
        return 'DQN Agent'


class ReinforceAgent(Agent):

    def __init__(self, config: Config):
        super(ReinforceAgent, self).__init__(config)
        self.reinforce = Reinforce(config)

    def step(self, obs, explore=True, legal_actions=None):
        return self.reinforce.get_action(obs, explore, legal_actions)

    def train(self, env):
        reward_sum = 0
        obs, info = env.reset()
        while True:
            result = self.step(obs, explore=True, legal_actions=env.get_legal_actions() if callable(
                getattr(env, 'get_legal_actions', None)) else None)
            action = result.get('action')
            action_log_prob = result.get('action_log_prob')
            next_obs, reward, terminate, truncated, info = env.step(action)
            self.reinforce.store_sample([action_log_prob, reward])
            obs = next_obs
            reward_sum += reward
            if terminate or truncated:
                break
        loss = self.reinforce.learn()

        return loss, reward_sum

    def save(self):
        self.reinforce.save_checkpoint()

    def load(self):
        self.reinforce.load_checkpoint()

    def __str__(self):
        return 'REINFORCE Agent'


class PPOAgent(Agent):

    def __init__(self, config: Config):
        super(PPOAgent, self).__init__(config)
        self.ppo = PPO(config)
        self.update_steps = config.get('update_steps')
        self.steps = 0

    def step(self, obs, explore=True, legal_actions=None):
        return self.ppo.get_action(obs, explore, legal_actions)

    def train(self, env):
        reward_sum = 0
        obs, info = env.reset()
        loss_list = []
        while True:
            self.steps += 1
            result = self.step(obs, explore=True, legal_actions=env.get_legal_actions() if callable(
                getattr(env, 'get_legal_actions', None)) else None)
            action = result.get('action')
            action_log_prob = result.get('action_log_prob')
            value = result.get('value')
            next_obs, reward, terminate, truncated, info = env.step(action)
            done = terminate or truncated
            self.ppo.store_sample([obs, action, action_log_prob, value, reward, done])
            obs = next_obs
            reward_sum += reward
            if self.steps % self.update_steps == 0:
                loss_list.append(self.ppo.learn())
            if done:
                break
        return loss_list, reward_sum

    def save(self):
        self.ppo.save_checkpoint()

    def load(self):
        self.ppo.load_checkpoint()

    def __str__(self):
        return 'PPO Agent'


class A2CAgent(Agent):

    def __init__(self, config: Config):
        super(A2CAgent, self).__init__(config)
        self.a2c = A2C(config)

    def step(self, obs, explore=True, legal_actions=None):
        return self.a2c.get_action(obs, explore, legal_actions)

    def train(self, env):
        reward_sum = 0
        obs, info = env.reset()
        loss_list = []
        while True:
            result = self.step(obs, explore=True, legal_actions=env.get_legal_actions() if callable(
                getattr(env, 'get_legal_actions', None)) else None)
            action = result.get('action')
            next_obs, reward, terminate, truncated, info = env.step(action)
            done = terminate or truncated
            reward_sum += reward
            loss = self.a2c.learn(obs, reward, next_obs, done)
            loss_list.append(loss)
            obs = next_obs
            if done:
                break

        return loss_list, reward_sum

    def save(self):
        self.a2c.save_checkpoint()

    def load(self):
        self.a2c.load_checkpoint()

    def __str__(self):
        return 'A2C Agent'


class A3CAgent(Agent):
    def __init__(self, config: Config):
        super().__init__(config)
        self.a3c = A3C(config)

    def step(self, obs, explore=True, legal_actions=None):
        return self.a3c.get_action(obs, explore, legal_actions)

    def train(self, env):
        reward_sum = 0
        obs, info = env.reset()
        while True:
            result = self.step(obs, explore=True, legal_actions=env.get_legal_actions() if callable(
                getattr(env, 'get_legal_actions', None)) else None)
            action = result.get('action')
            action_log_prob = result.get('action_log_prob')
            value = result.get('value')
            next_obs, reward, terminate, truncated, info = env.step(action)
            done = terminate or truncated
            reward_sum += reward
            self.a3c.store_sample([action_log_prob, value, reward])
            obs = next_obs
            if done:
                break
        loss = self.a3c.learn()
        return [loss], reward_sum

    def save(self):
        self.a3c.save_checkpoint()

    def load(self):
        self.a3c.load_checkpoint()

    def __str__(self):
        return 'A3C Agent'
