import numpy as np
import torch
from agent.configuration import Config


class Algorithm(object):
    def __init__(self,
                 configuration: Config):
        self.config = configuration

    def get_action(self, obs, explore=True, legal_actions=None):
        pass

    def store_sample(self, sample, **kwargs):
        pass

    def can_learn(self):
        pass

    def learn(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass


class DQN(Algorithm):

    def __init__(self, configuration: Config):
        super(DQN, self).__init__(configuration)
        self.eval_model = configuration.get('eval_model')
        self.target_model = configuration.get('target_model')
        self.batch_size = configuration.get('batch_size')
        self.gamma = configuration.get('gamma')
        self.memory = configuration.get('memory')
        self.learn_step_counter = 0
        self.target_replace_frequency = configuration.get('target_replace_frequency')
        self.criterion = configuration.get('criterion')
        self.optimizer = configuration.get('optimizer')
        self.device = configuration.get('device')
        self.exploration = configuration.get('exploration')
        self.max_grad_norm = configuration.get('max_grad_norm')
        self.ckpt_path = configuration.get('ckpt_path')
        self.memory_counter = 0
        self.ddqn = configuration.get('ddqn')

    def get_action(self, obs, explore=True, legal_actions=None):
        q_value = self.eval_model(
            torch.unsqueeze(
                torch.tensor(obs, dtype=torch.float32, device=self.device),
                dim=0
            )
        )[0]
        epsilon = self.exploration.get_epsilon()

        if not explore or np.random.uniform() > epsilon:
            if legal_actions:
                best_action = legal_actions[0]
                best_action_q = q_value[best_action]
                for action in legal_actions:
                    if q_value[action] > best_action_q:
                        best_action = action
                        best_action_q = q_value[action]
            else:
                best_action = torch.argmax(q_value).item()
        else:
            if legal_actions:
                best_action = legal_actions[np.random.randint(len(legal_actions))]
            else:
                best_action = np.random.randint(len(q_value))
        return {'action': best_action}

    def store_sample(self, sample, **kwargs):
        self.memory.store(sample, **kwargs)
        self.memory_counter += 1

    def can_learn(self):
        return self.memory_counter >= self.batch_size

    def learn(self):
        self.learn_step_counter += 1
        batch = self.memory.sample(self.batch_size)   
        batch_samples = batch.get('samples')
        tree_idx = batch.get('tree_idx', None)
        if tree_idx:
            tree_idx = np.array(tree_idx)
        samples_weight = batch.get('samples_weight', None)
        if samples_weight:
            samples_weight = np.array(samples_weight)
        else:
            samples_weight = np.ones(len(batch_samples))
        obs, action, reward, next_obs, dones = zip(*batch_samples)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.int64, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        q_eval = self.eval_model(obs).gather(1, action.reshape(-1, 1)).reshape(-1)

        if self.ddqn:
            q_eval_max_action = torch.argmax(self.eval_model(next_obs), dim=1)
            q_target = self.target_model(next_obs).detach()
            y = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.batch_size):
                y[i] = reward[i] if dones[i] else reward[i] + self.gamma * q_target[i, q_eval_max_action[i]].item()
        else:
            q_target = self.target_model(next_obs).max(1)[0].detach()
            y = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.batch_size):
                y[i] = reward[i] if dones[i] else reward[i] + self.gamma * q_target[i].item()

        abs_errors = torch.abs(y - q_eval).detach().cpu().numpy()
        self.memory.update(tree_idx, abs_errors)
        loss = torch.mean(torch.tensor(samples_weight, device=self.device) * torch.square(y - q_eval))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.exploration.step()
        if self.learn_step_counter % self.target_replace_frequency == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        return loss.item()


    def save_checkpoint(self):
        ckpt = {
            'eval_model_state_dict': self.eval_model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict()
        }
        torch.save(ckpt, self.ckpt_path)

    def load_checkpoint(self):
        ckpt = torch.load(self.ckpt_path, map_location='cpu')
        self.eval_model.load_state_dict(ckpt['eval_model_state_dict'])
        self.target_model.load_state_dict(ckpt['target_model_state_dict'])


# https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient
class Reinforce(Algorithm):

    def __init__(self, configuration: Config):
        super(Reinforce, self).__init__(configuration)
        self.policy_model = configuration.get('policy_model')
        self.gamma = configuration.get('gamma')
        self.device = configuration.get('device')
        self.optimizer = configuration.get('optimizer')
        self.ckpt_path = configuration.get('ckpt_path')
        self.eps = np.finfo(np.float32).eps.item()
        self.memory = configuration.get('memory')

    def get_action(self, obs, explore=False, legal_actions=None):
        action_probs = self.policy_model(
            torch.unsqueeze(torch.tensor(obs, device=self.device, dtype=torch.float32), dim=0)
        )[0]
        mask = torch.zeros(len(action_probs), device=self.device)
        if legal_actions:
            mask[legal_actions] = 1
            action_probs = mask * action_probs
        if explore:
            action = torch.multinomial(action_probs, num_samples=1)
        else:
            action = torch.argmax(action_probs)
        log_prob = torch.log(action_probs[action])
        return {'action': action.item(), 'action_log_prob': log_prob}

    def store_sample(self, sample, **kwargs):
        self.memory.store(sample)

    def learn(self, **kwargs):
        R = 0
        batches = self.memory.sample()
        samples = batches.get('samples')
        action_log_probs, rewards = zip(*samples)
        returns = np.zeros_like(rewards)
        for i in reversed(range(0, len(rewards))):
            R = rewards[i] + self.gamma * R
            returns[i] = R
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)  # 归一化
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        loss = -torch.sum(torch.cat(action_log_probs) * returns)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 10)
        self.optimizer.step()
        self.memory.clear()
        return loss.item()

    def save_checkpoint(self):
        check_point = {
            'policy_net': self.policy_model.state_dict(),
        }
        torch.save(check_point, self.ckpt_path)

    def load_checkpoint(self):
        check_point = torch.load(self.ckpt_path, map_location='cpu')
        self.policy_model.load_state_dict(check_point['policy_model'])


class PPO(Algorithm):

    def __init__(self, configuration: Config):
        super(PPO, self).__init__(configuration)
        self.actor_model = configuration.get('actor_model')
        self.critic_model = configuration.get('critic_model')
        self.gamma = configuration.get('gamma')
        self.actor_optimizer = configuration.get('actor_optimizer')
        self.critic_optimizer = configuration.get('critic_optimizer')
        self.gae_lambda = configuration.get('gae_lambda')
        self.ckpt_path = configuration.get('ckpt_path')
        self.batch_size = configuration.get('batch_size')
        self.learn_epochs = configuration.get('learn_epochs')
        self.clip_eps = configuration.get('clip_eps')
        self.device = configuration.get('device')
        self.memory = configuration.get('memory')
        self.actor_loss_factor = configuration.get('actor_loss_factor')
        self.critic_loss_factor = configuration.get('critic_loss_factor')
        self.entropy_loss_factor = configuration.get('entropy_loss_factor')
        self.actor_max_norm = configuration.get('actor_max_norm')
        self.critic_max_norm = configuration.get('critic_max_norm')

    def get_action(self, obs, explore=False, legal_actions=None):
        obs = torch.unsqueeze(torch.tensor(obs, device=self.device, dtype=torch.float32), dim=0)
        action_probs = self.actor_model(obs)[0]
        mask = torch.zeros(len(action_probs), device=self.device)
        if legal_actions:
            mask[legal_actions] = 1
            action_probs = mask * action_probs
        dist = torch.distributions.Categorical(action_probs)
        value = self.critic_model(obs).item()
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs)
        action_log_prob = dist.log_prob(action).item()
        action = action.item()

        return {'action': action, 'action_log_prob': action_log_prob, 'value': value}

    def store_sample(self, sample, **kwargs):
        self.memory.store(sample)

    def learn(self):
        batch_samples = self.memory.sample()
        samples = batch_samples.get('samples')
        obs, actions, action_log_probs, values, rewards, dones = zip(*samples)
        obs = np.array(obs)
        actions = np.array(actions)
        action_log_probs = np.array(action_log_probs)
        values = np.array(values)
        rewards = np.array(rewards)
        dones = np.array(dones)
        advantage = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        n = len(samples)
        loss_list = []
        for _ in range(self.learn_epochs):
            batch_starts = np.arange(0, n, self.batch_size)
            indices = np.arange(n, dtype=np.int64)
            np.random.shuffle(indices)
            batches_idx = [indices[i:i + self.batch_size] for i in batch_starts]
            for batch_idx in batches_idx:
                obs_batch = torch.tensor(obs[batch_idx], device=self.device, dtype=torch.float32)
                actions_batch = torch.tensor(actions[batch_idx], device=self.device, dtype=torch.int64)
                values_batch = torch.tensor(values[batch_idx], device=self.device, dtype=torch.float32)
                old_action_log_probs_batch = torch.tensor(action_log_probs[batch_idx], device=self.device, dtype=torch.float32)
                advantage_batch = torch.tensor(advantage[batch_idx], device=self.device, dtype=torch.float32)
                targets_batch = advantage_batch + values_batch

                action_probs_batch = self.actor_model(obs_batch)
                dist = torch.distributions.Categorical(action_probs_batch)
                entropy = dist.entropy()
                new_values_batch = self.critic_model(obs_batch).reshape(-1)
                new_action_log_probs_batch = dist.log_prob(actions_batch)
                ratio = new_action_log_probs_batch.exp() / old_action_log_probs_batch.exp()
                surr = ratio * advantage_batch
                actor_loss = -torch.mean(
                    torch.min(surr,
                              torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage_batch
                              )
                )
                critic_loss = torch.mean(torch.square(new_values_batch - targets_batch))
                entropy_loss = entropy.mean()
                total_loss = self.actor_loss_factor * actor_loss + self.critic_loss_factor * critic_loss - self.entropy_loss_factor * entropy_loss
                loss_list.append(total_loss.item())
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.actor_max_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.critic_max_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear()
        return np.mean(loss_list)

    def save_checkpoint(self):
        check_point = {
            'actor_model': self.actor_model.state_dict(),
            'critic_model': self.critic_model.state_dict()
        }
        torch.save(check_point, self.ckpt_path)

    def load_checkpoint(self):
        check_point = torch.load(self.ckpt_path, map_location='cpu')
        self.actor_model.load_state_dict(check_point['actor_model'])
        self.critic_model.load_state_dict(check_point['critic_model'])


class A2C(Algorithm):
    def __init__(self, configuration: Config):
        super(A2C, self).__init__(configuration)
        self.actor_critic_model = configuration.get('actor_critic_model')
        self.ckpt_path = configuration.get('ckpt_path')
        self.gamma = configuration.get('gamma')
        self.optimizer = configuration.get('optimizer')
        self.device = configuration.get('device')
        self.action_log_prob = None

    def get_action(self, obs, explore=False, legal_actions=None):
        obs = torch.unsqueeze(torch.tensor(obs, device=self.device), dim=0)
        action_probs, _ = self.actor_critic_model(obs)

        if legal_actions:
            mask = torch.zeros(len(action_probs[0]), device=self.device)
            mask[legal_actions] = 1
            action_probs = mask * action_probs
        dist = torch.distributions.Categorical(action_probs)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs)
        action_log_prob = dist.log_prob(action)
        self.action_log_prob = action_log_prob
        action = action.item()

        return {'action': action}

    def learn(self, obs, reward, next_obs, done):
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)

        _, value = self.actor_critic_model(obs)
        _, next_value = self.actor_critic_model(next_obs)

        delta = reward + self.gamma * next_value.detach() * (1 - int(done)) - value

        actor_loss = -self.action_log_prob * delta.detach()
        critic_loss = delta ** 2

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        self.optimizer.step()

        return (actor_loss + critic_loss).item()


class A3C(Algorithm):

    def __init__(self, configuration: Config):
        super().__init__(configuration)
        self.device = configuration.get('device')
        actor_critic_model_cls = configuration.get('actor_critic_model_cls')
        actor_critic_model_kwargs = configuration.get('actor_critic_model_kwargs')
        local_actor_critic_model = actor_critic_model_cls(**actor_critic_model_kwargs).to(self.device)
        self.local_actor_critic_model = local_actor_critic_model
        self.global_actor_critic_model = configuration.get('global_actor_critic_model')
        self.global_optimizer = configuration.get('global_optimizer')
        self.gamma = configuration.get('gamma')
        self.eps = np.finfo(np.float32).eps.item()
        self.memory = configuration.get('memory')

    
    def get_action(self, obs, explore=False, legal_actions=None):
        obs = torch.unsqueeze(torch.tensor(obs, device=self.device), dim=0)
        if explore:
            action_probs, value = self.local_actor_critic_model(obs)
        else:
            action_probs, value = self.global_actor_critic_model(obs)

        if legal_actions:
            mask = torch.zeros(len(action_probs[0]), device=self.device)
            mask[legal_actions] = 1
            action_probs = mask * action_probs
        dist = torch.distributions.Categorical(action_probs)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs)
        action_log_prob = dist.log_prob(action)
        action = action.item()

        return {'action': action, 'action_log_prob': action_log_prob, 'value': value[0]}
    
    def store_sample(self, sample, **kwargs):
        self.memory.store(sample)
    
    def ensure_shared_grads(self):
        for param, shared_param in zip(self.local_actor_critic_model.parameters(), self.global_actor_critic_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
    
    def learn(self):
        batch = self.memory.sample().get('samples')
        action_log_probs, values, rewards = zip(*batch)
        action_log_probs = torch.cat(action_log_probs)
        values = torch.cat(values)
        returns = np.zeros(len(rewards))
        R = 0
        for i in reversed(range(0, len(rewards))):
            R = rewards[i] + self.gamma * R
            returns[i] = R
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        value_loss = torch.mean((returns - values) ** 2)
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)
        policy_loss = -torch.sum(action_log_probs * returns)
        total_loss = policy_loss + value_loss
        self.global_optimizer.zero_grad()
        total_loss.backward()
        print(self.local_actor_critic_model.actor_model[0].weight.grad.mean())
        torch.nn.utils.clip_grad_norm_(self.local_actor_critic_model.parameters(), 50)
        self.ensure_shared_grads()
        self.global_optimizer.step()
        self.memory.clear()
        self.local_actor_critic_model.load_state_dict(self.global_actor_critic_model.state_dict())

        return total_loss.item()

        