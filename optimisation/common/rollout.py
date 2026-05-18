import numpy as np

def store_data(scope_tag: str, logger, step, lens, rewards):
    #scope_tag= "test"/"train"
    data = {"lens_stat/max": np.max(lens),
            "lens_stat/min":   np.min(lens),
            "lens_stat/mean": np.mean(lens),
            "lens_stat/std": np.std(lens),
            "returns_stat/max": np.max(rewards),
            "returns_stat/min": np.min(rewards),
            "returns_stat/mean": np.mean(rewards),
            "returns_stat/std": np.std(rewards)}

    logger.write(scope_tag, step, data)
    # self.logger.write("tag",step,value)

def store_data_diff(scope_tag: str, logger, step, pop_rewards, candidates_rewards,best_reward):
    diff_rewards = pop_rewards - candidates_rewards
    data = {"diff_stat/max": np.max(diff_rewards),
            "diff_stat/min": np.min(diff_rewards),
            "diff_stat/mean": np.mean(diff_rewards),
            "diff_stat/std": np.std(diff_rewards),
            "diff_stat/pop2best_max": np.max(pop_rewards-best_reward),
            "diff_stat/pop2best_min": np.min(pop_rewards-best_reward),
            "diff_stat/pop2best_mean": np.mean(pop_rewards-best_reward),
            "diff_stat/pop2best_std": np.std(pop_rewards-best_reward),
            }
    logger.write(scope_tag, step, data)

def store_data_test(scope_tag: str, logger, step, results):

    for instnace_tag, reward, steps, time in results:
        data = {f"{instnace_tag}/ens_stat/max": np.max(steps),
                f"{instnace_tag}/ens_stat/min":   np.min(steps),
                f"{instnace_tag}/ens_stat/mean": np.mean(steps),
                f"{instnace_tag}/ens_stat/std": np.std(steps),
                f"{instnace_tag}/reward_stat/max": np.max(reward),
                f"{instnace_tag}/reward_stat/min": np.min(reward),
                f"{instnace_tag}/reward_stat/mean": np.mean(reward),
                f"{instnace_tag}/reward_stat/std": np.std(reward)}
        logger.write(scope_tag, step, data)


def test_episode(logger, step, prog, task_envs):
    rewards = []
    lens = []
    trajectories_list = []
    for task_env in task_envs:
        steps, reward, trajectories = task_env.evaluate_program_logged(prog)
        # print("steps, reward, trajectories", steps, reward, trajectories)
        rewards.append(reward)
        lens.append(steps)
        trajectories_list.append(trajectories)
    data = {"lens_stat/max": np.max(lens),
            "lens_stat/min":   np.min(lens),
            "lens_stat/mean": np.mean(lens),
            "lens_stat/std": np.std(lens),
            "returns_stat/max": np.max(rewards),
            "returns_stat/min": np.min(rewards),
            "returns_stat/mean": np.mean(rewards),
            "returns_stat/std": np.std(rewards)}

    logger.write("program_test/env_step", step, data)