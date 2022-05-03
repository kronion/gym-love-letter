from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from gym_love_letter.envs.observations import Observation


class Agent(ABC):
    def __init__(self, env, *args, **kwargs):
        self.env = env

    @abstractmethod
    def predict(self, observation: Observation, action_masks: Optional[np.array] = None):
        raise RuntimeError("Unimplemented")
