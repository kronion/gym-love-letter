from typing import List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_love_letter.engine import Card, Deck, Player
from gym_love_letter.envs.actions import ActionWrapper


class Observation:
    MAX_NUM_PLAYERS = 4

    CURRENT_HAND_SIZE = 2
    TARGET_HAND_SIZE = 1
    PRIEST_SLOTS = 3  # Two priests + a king for a max of three distinct observations
    PRIEST_SLOT_SIZE = 2  # Target + card
    _ACTIVE = 1
    _SAFE = 1
    STATUS_SIZE = _ACTIVE + _SAFE

    def __init__(
        self,
        num_players: int,
        players: List[Player],
        curr_player: Player,
        deck: Deck,
        discard: List[Card],
        plays: List[ActionWrapper],
        game_over: bool,
        winners: List[Player],
        env: gym.Env,
    ):
        self.num_players = num_players
        self.players = players
        self.curr_player = curr_player
        self.deck = deck
        self.discard = discard
        self.plays = plays
        self.game_over = game_over
        self.winners = winners
        self.valid_actions = env.valid_actions

        self._init_player_vector_internals()
        self._init_full_vector_internals()

    def _init_player_vector_internals(self):
        # Position of information in the "player" observation vector
        i = 0
        self._player_hand_pos = slice(i, i + self.CURRENT_HAND_SIZE)
        i += self.CURRENT_HAND_SIZE

        # Remembered information about other player hands (from playing a Priest or King).
        # NB: Non-current players have fewer cards.
        self._player_target_hand_pos = []
        for slot in range(self.PRIEST_SLOTS):
            self._player_target_hand_pos.append(slice(i, i + self.PRIEST_SLOT_SIZE))
            i += self.PRIEST_SLOT_SIZE

        self._player_status_pos = []
        for pos in range(self.MAX_NUM_PLAYERS):
            self._player_status_pos.append(slice(i, i + self.STATUS_SIZE))
            i += self.STATUS_SIZE

        self._player_deck_size_pos = i
        i += 1

        self._discard_size = self.deck.size() - 1
        self._player_discard_pos = slice(i, i + self._discard_size)
        i += self._discard_size

        # Cannot be more actions than there are played cards
        self._action_history_size = self._discard_size
        self._player_action_history_pos = slice(i, i + self._action_history_size)
        i += self._action_history_size

        self.player_vec_length = i

    def _init_full_vector_internals(self):
        # Position of information in the "full" observation vector
        i = 0

        self._num_players_pos = i
        i += 1

        self._full_hand_pos = []
        for pos in range(self.MAX_NUM_PLAYERS):
            self._full_hand_pos.append(slice(i, i + self.CURRENT_HAND_SIZE))
            i += self.CURRENT_HAND_SIZE

        # Remembered information about other player hands (from playing a Priest or King).
        # NB: Non-current players have fewer cards.
        self._full_target_hand_pos = []
        for pos in range(self.MAX_NUM_PLAYERS):
            slots = []
            for slot in range(self.PRIEST_SLOTS):
                slots.append(slice(i, i + self.PRIEST_SLOT_SIZE))
                i += self.PRIEST_SLOT_SIZE

            self._full_target_hand_pos.append(slots)

        self._full_status_pos = []
        for pos in range(self.MAX_NUM_PLAYERS):
            self._full_status_pos.append(slice(i, i + self.STATUS_SIZE))
            i += self.STATUS_SIZE

        self._full_deck_size_pos = i
        i += 1

        self._discard_size = self.deck.size() - 1
        self._full_discard_pos = slice(i, i + self._discard_size)
        i += self._discard_size

        # Cannot be more actions than there are played cards
        self._action_history_size = self._discard_size
        self._full_action_history_pos = slice(i, i + self._action_history_size)
        i += self._action_history_size

        self.full_vec_length = i

    @property
    def vector(self) -> np.ndarray:
        """
        Encodes the game state visible to the current player.
        """

        vec = np.zeros(self.player_vec_length, dtype=np.int64)

        # Start the vector with the current player's hand
        vec[self._player_hand_pos] = self.curr_player.hand.vector

        for i, target in enumerate(self.curr_player.priest_info()):
            vec[self._player_target_hand_pos[i]] = [
                (target.position - self.curr_player.position) % self.num_players,
                self.curr_player.priest_info()[target],
            ]

        # Iterate over all players starting from the position of the current player
        for i in range(self.num_players):
            pos = (self.curr_player.position + i) % self.num_players
            player = self.players[pos]
            vec[self._player_status_pos[i]] = player.status_vector

        vec[self._player_deck_size_pos] = self.deck.remaining()

        # The following sections should be padded with zeros if the data is smaller than the available space
        pad_length = self._discard_size - len(self.discard)
        discard = [d.value for d in self.discard] + [0] * pad_length
        vec[self._player_discard_pos] = discard

        pad_length = self._action_history_size - len(self.plays)
        actions = [a.action._id for a in self.plays] + [0] * pad_length
        vec[self._player_action_history_pos] = actions

        return vec

    @property
    def full_vector(self):
        """
        Encodes the entire game state, not just the state visible to the current player.
        """

        vec = np.zeros(self.full_vec_length, dtype=np.int64)

        for pos in range(self.num_players):
            player = self.players[pos]
            vec[self._full_hand_pos[pos]] = player.hand.vector

        for pos in range(self.num_players):
            player = self.players[pos]

            for i, target in enumerate(player.priest_info()):
                vec[self._full_target_hand_pos[pos][i]] = [
                    (target.position - player.position) % self.num_players,
                    player.priest_info()[target],
                ]

        for pos in range(self.num_players):
            player = self.players[pos]
            vec[self._full_status_pos[pos]] = player.status_vector

        vec[self._full_deck_size_pos] = self.deck.remaining()

        # The following sections should be padded with zeros if the data is smaller than the available space
        pad_length = self._discard_size - len(self.discard)
        discard = [d.value for d in self.discard] + [0] * pad_length
        vec[self._full_discard_pos] = discard

        pad_length = self._action_history_size - len(self.plays)
        actions = [a.action._id for a in self.plays] + [0] * pad_length
        vec[self._full_action_history_pos] = actions

        return vec

    @classmethod
    def space(cls, action_space_size: int) -> spaces.Space:
        space = []

        # Put current player's cards at the front of the observation state so
        # they're easy to find.
        CARD_ONE = len(Card)
        CARD_TWO = len(Card)

        hand = [CARD_ONE, CARD_TWO]
        space += hand

        # Priest information. Three spaces because of the number of priests + king in the deck
        hand = [CARD_ONE]
        for _ in range(cls.PRIEST_SLOTS):
            space += [cls.MAX_NUM_PLAYERS]  # Every other player is a target, or nobody
            space += hand

        for player in range(cls.MAX_NUM_PLAYERS):
            STILL_ACTIVE = 2
            SAFE = 2
            player_state = [STILL_ACTIVE, SAFE]
            space += player_state

        # Now add in global information: remaining deck size and discard
        DECK_SIZE = Deck.size()
        space += [DECK_SIZE]

        # Keep track of every single discarded card
        # -1 because one card is held out from the deck each game
        for card in range(DECK_SIZE - 1):
            space += [len(Card)]

        # Keep track of every action made as well. There can only
        # be as many actions as there are played cards.
        for card in range(DECK_SIZE - 1):
            space += [action_space_size]

        return spaces.MultiDiscrete(space)

    def __repr__(self):
        return f"Observation: {self.vector}"


class DictObservation(Observation):
    @classmethod
    def space(cls, action_space_size: int) -> spaces.Space:
        return spaces.Dict({
            "observation": super().space(action_space_size),
            "action_mask": spaces.MultiBinary(action_space_size),
        })
