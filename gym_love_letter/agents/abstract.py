from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from gym_love_letter.envs.observations import Observation


class Agent(ABC):
    # May be overridden to indicate that the agent is controlled by a human
    interactive: bool = False

    @abstractmethod
    def predict(self, observation: Observation, action_masks: Optional[np.array] = None):
        raise RuntimeError("Unimplemented")
