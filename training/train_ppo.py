#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import os

import click
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import MlpPolicy, PPO

from gym_love_letter.agents import RandomAgent
from gym_love_letter.envs.base import LoveLetterMultiAgentEnv


LOGDIR = "ppo_tmp"  # moved to zoo afterwards.
logger.configure(folder=LOGDIR)

SEED = 721

# NUM_TIMESTEPS = int(2e7)
# EVAL_FREQ = 250000
# EVAL_EPISODES = 1000
NUM_TIMESTEPS = 300000
EVAL_FREQ = 5000
EVAL_EPISODES = 50



@click.command()
@click.option("--load", "-l", "load_path")
def train(load_path):
    env = LoveLetterMultiAgentEnv(num_players=4)
    env.seed(SEED)

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
    # model = PPO(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
    #             optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)
    if load_path:
        model = PPO.load(load_path, env)
    else:
        model = PPO(MlpPolicy, env)

    import ipdb; ipdb.set_trace()

    random_agents = [RandomAgent(env, SEED + i) for i in range(3)]
    agents = [model, *random_agents]
    env.set_agents(agents)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model"))  # probably never get to this point.

    env.close()


if __name__ == "__main__":
    train()
