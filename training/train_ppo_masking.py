#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import MlpPolicy, PPO

from gym_love_letter.agents import RandomAgent
from gym_love_letter.envs.base import LoveLetterMultiAgentEnv, Rewards


SEED = 721

# NUM_TIMESTEPS = int(2e7)
# EVAL_FREQ = 250000
# EVAL_EPISODES = 1000
NUM_TIMESTEPS = 100000
EVAL_FREQ = 10000
EVAL_EPISODES = 200


@click.command()
@click.argument("output_folder", type=click.Path())
@click.option("--load", "-l", "load_path")
def train(output_folder, load_path):
    base_output = Path(output_folder)
    full_output = base_output / datetime.datetime.now().isoformat(timespec="seconds")
    # latest = base_output / "latest"
    # latest.symlink_to(full_output)

    logger.configure(folder=str(full_output))

    env = LoveLetterMultiAgentEnv(
        num_players=4, reward_fn=Rewards.fast_elimination_reward
    )
    env.seed(SEED)

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
    # model = PPO(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
    #             optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)
    if load_path:
        model = PPO.load(load_path, env)
    else:
        def test_fn(env):
            return env.valid_action_mask()

        model = PPO(MlpPolicy, env, verbose=1, action_mask_fn=test_fn)

    other_agents = [RandomAgent(env, SEED + i) for i in range(3)]
    # other_agents = [
    #     PPO.load("zoo/ppo_reward_bugfix2/latest/best_model", env),
    #     PPO.load("zoo/ppo_reward_bugfix2/latest/best_model", env),
    #     PPO.load("zoo/ppo_reward_bugfix2/latest/best_model", env),
    # ]
    agents = [model, *other_agents]
    env.set_agents(agents)

    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(full_output),
        log_path=str(full_output),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
    )

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(str(full_output / "final_model"))

    env.close()


if __name__ == "__main__":
    train()
