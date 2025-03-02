# Code for discrete environment model
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time
import math
import highway_env


class SafetyBuffer:
    def __init__(self):
        self.safety_list = []
        self.safety_list_spare = []

    def clear(self):
        print("Safety Buffer length:{}".format(len(self.safety_list)))
        del self.safety_list[:]
        self.safety_list = copy.deepcopy(self.safety_list_spare)
        del self.safety_list_spare[:]

    def quantile(self, Q):
        S_safety = torch.tensor(self.safety_list)
        return torch.quantile(S_safety, Q / 10)  # 传入小数


# Buffer
class RolloutBuffer:
    """
    Buffer for storing episode data
    """

    def __init__(self):
        self.actions = []  # Actions
        self.states = []  # States
        self.next_states = []
        self.rewards = []  # Rewards
        self.is_terminals = []  # Terminal
        self.rewards_safety = []  # Collision penalties
        self.next_actions_safety = []
        self.actions_safety = []

    def clear(self):
        """
        Clear the buffer
        """
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.rewards_safety[:]
        del self.next_actions_safety[:]
        del self.actions_safety[:]


def orthogonal_init(layer, gain=1.0):
    """
    Orthogonal initialization
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


# ----------------------------------- #
# Build policy network -- actor
# ----------------------------------- #
class PolicyNet(nn.Module):
    def __init__(self, n_states, nvecs_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.ModuleList([nn.Linear(64, n) for n in nvecs_actions])

        # self._initialize_weights()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        for layer in self.fc3:
            orthogonal_init(layer, gain=0.01)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x1 = F.leaky_relu(x)
        x = [F.softmax(layer(x1), dim=-1) for layer in self.fc3]  # Calculate probability of each action
        if torch.isnan(x[0]).any():
            print(x)  # 寻找是否传入空
        return x


# ----------------------------------- #
# Build value network -- critic
# ----------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        # self._initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)  # Evaluate current state value
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x


# ----------------------------------- #
# Build safety network -- safety
# ----------------------------------- #

class SafetyNet(nn.Module):
    def __init__(self, n_states):
        super(SafetyNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 8)

        self.f1 = nn.Linear(n_states, 64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 1)

        # orthogonal_init(self.fc1)
        # orthogonal_init(self.fc2)
        # orthogonal_init(self.f1)
        # orthogonal_init(self.f2)
        # orthogonal_init(self.f3)
        # orthogonal_init(self.fc3)

    def forward(self, x1, x2):
        x1 = x1.to(dtype=torch.float)
        x1 = self.fc1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.fc2(x1)
        x1 = F.leaky_relu(x1)

        x2 = self.f1(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.f2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.f3(x2)
        x2 = F.leaky_relu(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = torch.clamp(-self.fc4(x), max=0)
        return x


# ----------------------------------- #
# 构建模型
# ----------------------------------- #

class PPO:
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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.994)

        # Optimizer for value network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.994)

        # Optimizer for safety network
        self.safety_optimizer = torch.optim.Adam(self.safety.parameters(), lr=safety_lr)
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

    # 动作选择
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
            self.safety_buffer.safety_list_spare.append(safety_value_actor.tolist())  # 在进行优化时,将优化过程中的输出记录下来,不浪费
        # self.safety_buffer.safetybuffer.append(safety_value_rule)

        action_make = actions if self.ExplorationFunction(safety_value_actor, safety_value_rule, self.steps,
                                                          If_update=If_updata) else action_rule

        # print(safety_value_actor, safety_value_rule)
        # print(torch.tensor(np.array(actions)).to(self.device))
        # print(torch.tensor(np.array(action_rule)).to(self.device))
        # action_make = actions if (safety_value_actor + 0.1) >= safety_value_rule else action_rule
        # action_make = actions
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
        env = highway_env_im.HighwayEnv()
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

        # print(next_actions)
        # print(next_actions.shape)

        next_q_target = self.critic(next_states)
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        td_value = self.critic(states)
        td_delta = td_target - td_value
        td_delta = td_delta.cpu().detach().numpy()

        next_s_target = self.safety(next_actions_safety, next_states[:-1])  # 使用2047步更新
        # print(next_s_target.shape)
        # print(rewards_safety[:-1].shape)
        td_safety_target = rewards_safety[:-1] + self.gamma * next_s_target * (1 - dones[:-1])  # 2047
        # td_safety_value = self.safety(actions[:-1], states[:-1])  # 其实此处的actions相当于第一步的动作，还需要求取下一步的动作

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

                # if buffer_capacity - 1 in index: index.remove(buffer_capacity - 1)
                # safety_loss = torch.max(
                #     F.mse_loss(self.safety(actions_safety[index], states[index]), td_safety_target[index].detach()))
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
                self.write.add_scalar('.', actor_loss, self.steps)  # 全局步数
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
