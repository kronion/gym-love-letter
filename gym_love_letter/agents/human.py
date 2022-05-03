from gym_love_letter.agents.abstract import Agent


class HumanAgent(Agent):
    def predict(self, *args, **kwargs) -> int:
        raise RuntimeError("Expected a human to make the decision")
