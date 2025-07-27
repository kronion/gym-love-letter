from gym_love_letter.agents.abstract import Agent


class HumanAgent(Agent):
    def __init__(self, env):
        self.env = env

    def predict(self, *args, **kwargs) -> int:
        raise RuntimeError("Expected a human to make the decision")
