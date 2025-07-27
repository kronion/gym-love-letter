from .base import LoveLetterMultiAgentEnv
from .observations import DictObservation


class LoveLetterDictEnv(LoveLetterMultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = DictObservation.space()

    def observe(self) -> DictObservation:
        return DictObservation(
            self.num_players,
            self.players,
            self.current_player,
            self.deck,
            self.discard_pile,
            self.action_history,
            self.game_over,
            self.winners,
            self,
        )
