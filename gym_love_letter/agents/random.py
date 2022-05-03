from typing import Tuple

from gym.utils import seeding

from gym_love_letter.agents.abstract import Agent


class RandomAgent(Agent):
    def __init__(self, env, seed: int = None):
        super().__init__(env)

        if seed is not None:
            self.seed(seed)

    def predict(self, *args, **kwargs) -> Tuple[int, None]:
        valid_action_ids = [a._id for a in self.env.valid_actions]
        return (self.np_random.choice(valid_action_ids), None)

    @property
    def np_random(self):
        """
        Lazily seed the rng if not set explicitly.
        """

        if not hasattr(self, "_np_random"):
            self.seed()

        return self._np_random

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
