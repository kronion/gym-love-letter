from typing import SupportsFloat

import gymnasium as gym

from gym_love_letter.envs.base import LoveLetterBaseEnv
from gym_love_letter.envs.observations import DictObservation


class DictEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.unwrapped, LoveLetterBaseEnv)
        assert isinstance(self.action_space, gym.spaces.Discrete)

        self.observation_space = DictObservation.space(int(self.action_space.n))

    def reset(self, *args, **kwargs) -> tuple[dict, dict]:
        assert isinstance(self.unwrapped, LoveLetterBaseEnv)

        obs, info = super().reset(*args, **kwargs)

        dict_obs = {
            "observation": obs,
            "action_mask": self.unwrapped.valid_action_mask()
        }

        return dict_obs, info

    def step(self, *args, **kwargs) -> tuple[dict, SupportsFloat, bool, bool, dict]:
        assert isinstance(self.unwrapped, LoveLetterBaseEnv)

        obs, reward, terminated, truncated, info = super().step(*args, **kwargs)

        dict_obs = {
            "observation": obs,
            "action_mask": self.unwrapped.valid_action_mask()
        }

        return dict_obs, reward, terminated, truncated, info
