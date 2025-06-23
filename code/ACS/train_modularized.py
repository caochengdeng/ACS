import torch
from config.training_config import TrainingConfig
from ACS.metrics.tracker import MetricTracker
from envs.highway_env import HighwayEnv
from ACS.algo.ACS_Agent import ACSADP
from models.actor_critic import PolicyNet, ValueNet
from models.safety import SafetyNet
from common.buffer import RolloutBuffer
from utils.helpers import set_seed, save_training_state, log_metrics, update_criteria_periodically
import copy


def main():
    set_seed(TrainingConfig.SEED)

    # env
    env_wrapper = HighwayEnv(TrainingConfig.ENV_NAME, TrainingConfig.SEED)

    # agent
    agent = ACSADP(n_states=env_wrapper.n_states,
                   n_actions=env_wrapper.nvecs_actions,
                   actor_lr=TrainingConfig.ACTOR_LR,
                   critic_lr=TrainingConfig.CRITIC_LR,
                   safety_lr=TrainingConfig.SAFETY_LR,
                   lmbda=TrainingConfig.LMBDA,
                   epochs=TrainingConfig.EPOCHS,
                   eps=TrainingConfig.EPS,
                   gamma=TrainingConfig.GAMMA,
                   vf_coef=TrainingConfig.VF_COEF,
                   ent_coef=TrainingConfig.ENT_COEF,
                   max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
                   batch_size=TrainingConfig.BATCH_SIZE,
                   max_step=TrainingConfig.MAX_STEP,
                   write=TrainingConfig.writer,
                   device=TrainingConfig.DEVICE)

    # buffer
    buffer = RolloutBuffer()
    capacity = TrainingConfig.CAPACITY

    metrics_tracker = MetricTracker()

    step_num = 0
    action_rule = [1, 3]

    while step_num < TrainingConfig.MAX_STEP:
        state = env_wrapper.reset()
        done = False
        episode_return = 0
        step = 0

        while not done:
            step += 1
            step_num += 1

            # 记录速度和加速度
            metrics_tracker.add_velocity(state[0])
            metrics_tracker.add_acceleration(state[1])

            action, action_agent = agent.take_action(state, action_rule)
            action_copy = copy.deepcopy(action_agent)

            next_state, reward, reward_safety, done, _, info, action_next_rule = env_wrapper.step(action)
            action_rule = copy.deepcopy(action_next_rule)

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

                log_metrics(TrainingConfig.writer, metrics_tracker, step_num,
                            len(metrics_tracker.return_list), env_wrapper.max_simulation_step)

                metrics_tracker.reset_velocity_and_acceleration()

            if done:
                metrics_tracker.add_return(episode_return)
                metrics_tracker.add_steps(step)

                if update_criteria_periodically(agent, step_num, TrainingConfig.MAX_STEP, metrics_tracker):
                    pass

                episode_return = 0
                step = 0
                metrics_tracker.reset_velocity_and_acceleration()
                break

    TrainingConfig.writer.close()
    env_wrapper.close()


if __name__ == "__main__":
    main()
