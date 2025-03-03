import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from ACS_Agent import ACSAgent, RolloutBuffer
import highway_env
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_episodes = 5000
gamma = 0.96
actor_lr = 0.001
critic_lr = 0.001
safety_lr = 0.001
return_list = []
vf_coef = 0.1
ent_coef = 0.01
max_grad_norm = 1.5
batch_szie = 256
max_step = 1000000
writer = SummaryWriter(r'...')

env = highway_env.HighwayEnv()
n_states = env.observation_space.shape[1]
nvecs_actions = env.action_space.nvec
max_simulation_step = env.simulation_time * 10


# set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(12)

agent = ACSAgent(n_states=n_states,
            n_actions=nvecs_actions,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            safety_lr=safety_lr,
            lmbda=0.95,
            epochs=10,
            eps=0.2,
            gamma=gamma,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            batch_size=batch_szie,
            max_step=max_step,
            write=writer,
            device=device
            )

buffer = RolloutBuffer()
capacity = 2048

episode = 0
reward_update = []
steps_list = []

step_num = 0
velocity_episode = []
velocity_step = []
acceleration_episode = []
acceleration_step = []
if_update = False
while step_num < max_step:
    state = env.reset()
    done = False
    episode_return = 0
    action_rule = [1, 3]
    next_action = [1, 3]
    step = 0
    while True:
        step += 1
        step_num += 1
        velocity_step.append(state[0])
        velocity_episode.append(state[0])
        acceleration_step.append(state[1])
        acceleration_episode.append(state[1])

        action, action_agent = agent.take_action(state, action_rule)
        action_copy = copy.deepcopy(action_agent)

        next_state, reward, reward_safety, done, _, _, action_next_rule = env.step(action)
        action_rule = action_next_rule

        buffer.states.append(state)
        buffer.rewards_safety.append(reward_safety)
        buffer.actions.append(action_copy)
        buffer.next_states.append(next_state)
        buffer.rewards.append(reward)
        buffer.is_terminals.append(done)
        if (step_num + 1) % capacity != 1:
            buffer.next_actions_safety.append(action_copy)
        buffer.actions_safety.append(action_copy)

        state = next_state

        episode_return += reward
        if len(buffer.states) >= capacity:
            agent.learn(buffer)
            buffer.clear()
            writer.add_scalar('reward/average_reward', np.mean(reward_update), step_num)

            writer.add_scalar('done/success_step', np.mean(steps_list), step_num)
            writer.add_scalar('done/success_rate', np.mean([1 if i >= max_simulation_step else 0 for i in steps_list]),
                              step_num)

            writer.add_scalar('velocity/step_velocity', np.mean(velocity_step), step_num)

            writer.add_scalar('acceleration/acceleration_step', np.mean(acceleration_step), step_num)

            reward_update, steps_list, velocity_step, acceleration_step = [], [], [], []

        if done:
            episode += 1
            steps_list.append(step)
            reward_update.append(episode_return)
            # return
            return_list.append(episode_return)
            writer.add_scalar('reward/episode_reward', episode_return, episode)  # 存储每局的奖励
            writer.add_scalar('reward/episode_average_reward', np.mean(return_list[-10:]), episode)  # 每局的平均奖励，移动平均

            writer.add_scalar('done/steps', step, episode)

            writer.add_scalar('velocity/episode_velocity', np.mean(velocity_episode), episode)

            writer.add_scalar('acceleration/acceleration_episode', np.mean(acceleration_episode), episode)

            velocity_episode, acceleration_episode = [], []
            step = 0

            if if_update:
                start_time = time.time()
                s_criteria, reward_list, success_list, s_criteria_list = agent.Update_Criteria()  # 更新相关指标
                end_time = time.time()
                if_update = False
                print(s_criteria)
                print(reward_list)
                print(success_list)
                print(s_criteria_list)
                print("耗时: {:.2f}秒".format(end_time - start_time))
            break

        if step_num % (max_step / 10) == 0:
            agent.save_model(path='.', index=step_num)
            if_update = True
            print(step_num)
writer.close()
