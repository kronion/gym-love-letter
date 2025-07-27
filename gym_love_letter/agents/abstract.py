from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

import numpy as np


if TYPE_CHECKING:
    from gym_love_letter.envs import LoveLetterBaseEnv
    from gym_love_letter.envs.observations import Observation


class Agent(Protocol):
    env: LoveLetterBaseEnv

    @abstractmethod
    def predict(self, observation: Observation, action_masks: np.array | None = None):
        raise RuntimeError("Unimplemented")
