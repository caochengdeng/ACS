import os
import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn

from ACS.common.buffer import SafetyBuffer
from ACS.models.actor_critic import PolicyNet, ValueNet
from ACS.models.safety import SafetyNet
from ACS.envs.highway_env import HighwayEnv


class ACSADP:
    def __init__(self, n_states, n_actions,
                 actor_lr, critic_lr, safety_lr, lmbda, epochs, eps, gamma, vf_coef, ent_coef, max_grad_norm,
                 batch_size, max_step,
                 write, device):
        # Instantiate policy network
        self.actor = PolicyNet(n_states, n_actions).to(device)
        # Instantiate value network
        self.critic = ValueNet(n_states).to(device)
        # Instantiate safety network
        self.safety = SafetyNet(n_states).to(device)

        # Optimizer for policy network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.994)

        # Optimizer for value network
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.994)

        # Optimizer for safety network
        self.safety_optimizer = optim.Adam(self.safety.parameters(), lr=safety_lr)
        self.scheduler_safety = torch.optim.lr_scheduler.ExponentialLR(self.safety_optimizer, gamma=0.994)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.max_step = max_step
        self.device = device
        self.write = write
        self.steps = 0

        # This means safe within the next 5 seconds, initialized first step
        self.s_criteria = -50 * self.gamma ** 50
        self.safety_buffer = SafetyBuffer()

    def take_action(self, state, action_rule, If_updata=False):
        self.steps += 1
        # Dimension change [n_state]-->tensor[1,n_states]
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # 转化为二维张量

        # Probability distribution of each action under the current state [2,]
        probs = self.actor(state)
        actions = []

        for i in range(len(probs)):
            action_list = torch.distributions.Categorical(probs[i])
            action = action_list.sample().item()
            actions.append(action)

        safety_value_rule = self.safety(
            torch.tensor(np.array(action_rule), dtype=torch.float).unsqueeze(0).to(self.device), state)
        safety_value_actor = self.safety(
            torch.tensor(np.array(actions), dtype=torch.float).unsqueeze(0).to(self.device), state)

        if not If_updata:
            self.safety_buffer.safety_list.append(safety_value_actor.tolist())
        else:
            self.safety_buffer.safety_list_spare.append(safety_value_actor.tolist())
        # self.safety_buffer.safetybuffer.append(safety_value_rule)

        action_make = actions if self.ExplorationFunction(safety_value_actor, safety_value_rule, self.steps,
                                                          If_update=If_updata) else action_rule

        return action_make, actions

    def ExplorationFunction(self, s_actor, s_rule, step, If_update=False):
        if not If_update:
            self.write.add_scalar('.', s_actor, self.steps)
            self.write.add_scalar('.', s_rule, self.steps)
        if s_actor >= self.s_criteria: return True
        if s_actor >= s_rule:
            return True
        else:
            if If_update:
                return False
            p = math.exp(
                (s_actor - s_rule) / (30 * ((1.1 * self.max_step - step) / self.max_step) ** 2))  # 需要给予一定的探索能力,可能5还不够
            if not If_update:
                self.write.add_scalar('.', p, self.steps)
            if np.random.rand() < p:
                return True
            else:
                return False

    def Update_Criteria(self):
        print('update s_criteria...')
        env = HighwayEnv()
        reward_all = []
        success_rate = []
        s_criteria_list = []
        k = random.randint(1, 100)

        for u in range(0, 11):
            self.s_criteria = self.safety_buffer.quantile(u).tolist()
            s_criteria_list.append(self.s_criteria)
            reward_ = []  # Store values under each standard
            success_rate_episode = []
            for i in range(1, 11):
                done = False
                reward_episode = 0
                step = 0
                action_rule = [1, 3]

                state = env.reset(If_update=True, Up_seed=i * k)  # Reset

                while not done:
                    step += 1
                    action, _ = self.take_action(state, action_rule, If_updata=True)
                    self.steps -= 1

                    next_state, reward, reward_safety, done, _, info, action_next_rule = env.step(action)
                    # Update state
                    state = next_state
                    action_rule = action_next_rule
                    reward_episode += reward

                success_rate_episode.append(1 if step >= 600 else 0)
                reward_.append(reward_episode)

            success_rate.append(sum(success_rate_episode) / 10)
            reward_all.append(sum(reward_) / 10)

        index = list(np.argsort(np.array(reward_all)))
        self.s_criteria = s_criteria_list[index[-1]]
        self.safety_buffer.clear()
        self.write.add_scalar('S/s_criteria', s_criteria_list[index[-1]], self.steps)

        return s_criteria_list[index[-1]], reward_all, success_rate, s_criteria_list

    # Training
    def learn(self, buffer):
        states = torch.tensor(buffer.states, dtype=torch.float).to(self.device)
        actions = torch.tensor(buffer.actions).to(self.device)
        rewards = torch.tensor(buffer.rewards, dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(buffer.next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(buffer.is_terminals, dtype=torch.float).to(self.device).view(-1, 1)
        rewards_safety = torch.tensor(buffer.rewards_safety, dtype=torch.float).to(self.device).view(-1, 1)
        next_actions_safety = torch.tensor(buffer.next_actions_safety, dtype=torch.float).to(self.device)
        actions_safety = torch.tensor(buffer.actions_safety, dtype=torch.float).to(self.device)

        next_q_target = self.critic(next_states)
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        td_value = self.critic(states)
        td_delta = td_target - td_value
        td_delta = td_delta.cpu().detach().numpy()

        next_s_target = self.safety(next_actions_safety, next_states[:-1])

        td_safety_target = rewards_safety[:-1] + self.gamma * next_s_target * (1 - dones[:-1])

        advantage = 0
        advantage_list = []

        # Calculate advantage function
        for delta in td_delta[::-1]:  # Reverse temporal difference value
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)

        advantage_list.reverse()

        advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)

        old_probs = self.actor(states)
        old_log_probs_1 = torch.log(old_probs[0].gather(1, actions[:, 0].unsqueeze(0).T)).detach()
        old_log_probs_2 = torch.log(old_probs[1].gather(1, actions[:, 1].unsqueeze(0).T)).detach()
        old_log_probs = old_log_probs_2 + old_log_probs_1

        buffer_capacity = len(buffer.rewards)

        for u in range(self.epochs):
            for index in BatchSampler(SubsetRandomSampler(range(buffer_capacity)), self.batch_size, True):
                probs = self.actor(states[index])
                log_probs_1 = torch.log(probs[0].gather(1, actions[index, 0].unsqueeze(0).T))
                log_probs_2 = torch.log(probs[1].gather(1, actions[index, 1].unsqueeze(0).T))
                log_probs = log_probs_2 + log_probs_1

                # Ratio between new and old policies
                ratio = torch.exp(log_probs - old_log_probs[index])

                surr1 = ratio * advantage[index]

                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage[index]

                actor_loss = torch.mean(-torch.min(surr1, surr2))
                # print(actor_loss)

                critic_loss = torch.mean(F.mse_loss(self.critic(states[index]), td_target[index].detach()))

                if buffer_capacity - 1 in index: index.remove(buffer_capacity - 1)
                safety_loss = torch.mean(
                    F.mse_loss(self.safety(actions_safety[index], states[index]), td_safety_target[index].detach()))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.safety_optimizer.zero_grad()

                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                safety_loss.backward()
                nn.utils.clip_grad_norm_(self.safety.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.safety_optimizer.step()
            if u == self.epochs - 1:
                self.write.add_scalar('.', actor_loss, self.steps)
                self.write.add_scalar('.', critic_loss, self.steps)
                self.write.add_scalar('.', safety_loss, self.steps)
                self.write.add_scalar('.', self.actor_optimizer.state_dict()['param_groups'][0]['lr'],
                                      self.steps)
                self.write.add_scalar('.', self.critic_optimizer.state_dict()['param_groups'][0]['lr'],
                                      self.steps)
                self.write.add_scalar('.', self.safety_optimizer.state_dict()['param_groups'][0]['lr'],
                                      self.steps)
                self.scheduler_actor.step()
                self.scheduler_critic.step()
                self.scheduler_safety.step()

    def save_model(self, path, index):
        """
        save model
        :param path:
        :param index: Which time to save the model
        """
        path_actor = os.path.join(path, 'actor_model_{}'.format(index))
        path_critic = os.path.join(path, 'critic_model_{}'.format(index))
        path_safety = os.path.join(path, 'safety_model_{}'.format(index))
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)
        torch.save(self.safety.state_dict(), path_safety)

    def load_model(self, path, index):
        """
        load model
        :param index:
        :param path:
        """
        path_actor = os.path.join(path, 'actor_model_{}'.format(index))
        path_safety = os.path.join(path, 'safety_model_{}'.format(index))
        self.actor.load_state_dict(torch.load(path_actor, map_location=lambda storage, loc: storage))
        self.safety.load_state_dict(torch.load(path_safety, map_location=lambda storage, loc: storage))