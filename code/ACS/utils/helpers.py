import time
import copy
import numpy as np
import torch


def set_seed(seed):
    """set seed"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_training_state(agent, step_num):
    """save model"""
    agent.save_model(path='.', index=step_num)


def log_metrics(writer, metrics_tracker, step_num, episode, max_simulation_step):
    """recorde TensorBoard"""
    writer.add_scalar('reward/average_reward', np.mean(metrics_tracker.reward_update), step_num)
    writer.add_scalar('done/success_step', np.mean(metrics_tracker.steps_list), step_num)
    writer.add_scalar('done/success_rate', metrics_tracker.get_success_rate(max_simulation_step), step_num)
    writer.add_scalar('velocity/step_velocity', np.mean(metrics_tracker.velocity_step), step_num)
    writer.add_scalar('acceleration/acceleration_step', np.mean(metrics_tracker.acceleration_step), step_num)
    writer.add_scalar('reward/episode_reward', metrics_tracker.return_list[-1], episode)
    writer.add_scalar('reward/episode_average_reward', metrics_tracker.get_recent_average_reward(), episode)
    writer.add_scalar('done/steps', metrics_tracker.steps_list[-1], episode)
    writer.add_scalar('velocity/episode_velocity', np.mean(metrics_tracker.velocity_episode), episode)
    writer.add_scalar('acceleration/acceleration_episode', np.mean(metrics_tracker.acceleration_episode), episode)


def update_criteria_periodically(agent, step_num, max_step, metrics_tracker):
    if step_num % (max_step / 10) == 0 and step_num > 0:
        start_time = time.time()
        s_criteria, reward_list, success_list, s_criteria_list = agent.Update_Criteria()
        end_time = time.time()

        print(f"s_criteria: {s_criteria}")
        print(f"reward_list: {reward_list}")
        print(f"success_list: {success_list}")
        print(f"s_criteria_list: {s_criteria_list}")
        print(f"time: {end_time - start_time:.2f}s")
        return True
    return False