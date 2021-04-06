import torch as th
from stable_baselines3.ppo import PPO

from gym_love_letter.agents import RandomAgent
from gym_love_letter.envs.base import InvalidPlayError, LoveLetterMultiAgentEnv


def make_agents(env):
    load_path = "zoo/ppo_masking/final_model"
    model = PPO.load(load_path, env)
    random1 = RandomAgent(env)
    random2 = RandomAgent(env)
    random3 = RandomAgent(env)

    return [model, random1, random2, random3]


env = LoveLetterMultiAgentEnv(num_players=4, make_agents_cb=make_agents)


GAME_LIMIT = 1000
STEP_LIMIT = 100


env.reset()
wins_by_pos = {p.position: 0 for p in env.players}
invalid_games = 0


for i in range(GAME_LIMIT):
    if i % 10 == 0:
        print(i)

    env.reset()
    move_count = 0

    while not env.game_over and move_count < STEP_LIMIT:
        if env.current_player.active:
            try:
                move_count += 1
                agent = env.agents[env.current_player.position]
                if type(agent) is PPO:
                    action_mask = th.as_tensor(env.valid_action_mask())
                    action_id, _state = agent.predict(env.observe().vector, action_masks=action_mask)
                else:
                    action_id, _state = agent.predict(env.observe().vector)
                env.protected_step(action_id, full_cycle=False)
            except InvalidPlayError:
                continue
        else:
            env._next_player()

    if move_count == STEP_LIMIT:
        print(env.current_player)
        invalid_games += 1
    else:
        for p in env.players:
            if p in env.winners:
                wins_by_pos[p.position] += 1


for k, v in wins_by_pos.items():
    print(f"{k}: {v}")
print(f"Invalid games: {invalid_games}")
