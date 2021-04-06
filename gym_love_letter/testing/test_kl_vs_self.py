import torch as th
from stable_baselines3.ppo import PPO

from gym_love_letter.agents import RandomAgent
from gym_love_letter.envs.base import InvalidPlayError, LoveLetterMultiAgentEnv


def make_agents(env):
    new_load_path = "zoo/ppo_kl/2020-12-27T16:28:42/final_model"
    new_model = PPO.load(new_load_path, env)

    old_load_path = "zoo/ppo_logging/2020-12-27T15:51:49/final_model"
    # old_load_path = "zoo/ppo_headsup/latest/best_model"
    old_model = PPO.load(old_load_path, env)

    # random1 = RandomAgent(env)
    # random2 = RandomAgent(env)

    return [new_model, old_model]


env = LoveLetterMultiAgentEnv(num_players=2, make_agents_cb=make_agents)


GAME_LIMIT = 20000
STEP_LIMIT = 100


wins_by_pos = {p.position: 0 for p in env.players}
starts_by_pos = {p.position: 0 for p in env.players}
invalid_games = 0


for i in range(GAME_LIMIT):
    if i % 100 == 0:
        print(i)
        print("Wins by position")
        for k, v in wins_by_pos.items():
            print(f"{k}: {v}")
        print(f"Invalid games: {invalid_games}")
        print("Starts by position")
        for k, v in starts_by_pos.items():
            print(f"{k}: {v}")

    try:
        env.reset(training=False)
    except InvalidPlayError:
        continue
    starts_by_pos[env.starting_player.position] += 1

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


print("Wins by position")
for k, v in wins_by_pos.items():
    print(f"{k}: {v}")
print(f"Invalid games: {invalid_games}")
print("Starts by position")
for k, v in starts_by_pos.items():
    print(f"{k}: {v}")
