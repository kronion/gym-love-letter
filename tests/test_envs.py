import pytest

from gym_love_letter.agents import HumanAgent, RandomAgent
from gym_love_letter.envs import LoveLetterBaseEnv


class TestBaseEnvInitialization:
    def test_default_initialization(self):
        env = LoveLetterBaseEnv()
        obs = env.reset()

    def test_initialize_agent_classes(self):
        env = LoveLetterBaseEnv(agent_classes=[HumanAgent, RandomAgent])
        obs = env.reset()

        assert isinstance(env.players[0].agent, HumanAgent)
        assert isinstance(env.players[1].agent, RandomAgent)

    def test_set_agent(self):
        env = LoveLetterBaseEnv()
        agents = [RandomAgent(env), HumanAgent(env)]
        env.set_agents(agents)
        obs = env.reset()

        assert isinstance(env.players[0].agent, RandomAgent)
        assert isinstance(env.players[1].agent, HumanAgent)

    def test_num_players_and_agents_must_be_equal(self):
        # Constructor approach
        with pytest.raises(ValueError):
            env = LoveLetterBaseEnv(num_players=1, agent_classes=[HumanAgent, RandomAgent])
        with pytest.raises(ValueError):
            env = LoveLetterBaseEnv(num_players=3, agent_classes=[HumanAgent, RandomAgent])

        # Post-constructor method approach
        env = LoveLetterBaseEnv(num_players=2)
        agents = [RandomAgent(env)]
        with pytest.raises(ValueError):
            env.set_agents(agents)
        agents = [RandomAgent(env), RandomAgent(env), RandomAgent(env)]
        with pytest.raises(ValueError):
            env.set_agents(agents)

class TestBaseEnvGameplay:
    def test_basic_game_completion(self):
        env = LoveLetterBaseEnv(num_players=2)

        # Seed the env and random agents for reproducability
        MAIN_SEED = 543210543210
        env.seed(MAIN_SEED)
        for player in env.players:
            assert isinstance(player.agent, RandomAgent)
            seed = env.np_random.randint(0, 2 ** 63 - 1)
            player.agent.seed(seed)

        obs = env.reset()
        done = False

        # Because of deck size, the game is guaranteed to finish within 16 moves
        move_count = 0
        MAX_GAME_DURATION = 16

        while not done and move_count < MAX_GAME_DURATION:
            player = env.current_player
            action_id, _ = player.agent.predict(obs)
            obs, reward, done, info = env.step(action_id)
            move_count += 1

        assert move_count < MAX_GAME_DURATION
