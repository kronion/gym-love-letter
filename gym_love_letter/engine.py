from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from enum import IntEnum
from typing import Sequence, Type

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from gym_love_letter import constants, utils
from gym_love_letter.agents import Agent, HumanAgent


class Card(IntEnum):
    EMPTY = 0
    GUARD = 1
    PRIEST = 2
    BARON = 3
    HANDMAID = 4
    PRINCE = 5
    KING = 6
    COUNTESS = 7
    PRINCESS = 8

    @property
    def takes_target(self):
        return self in [self.GUARD, self.PRIEST, self.BARON, self.PRINCE, self.KING]

    @classmethod
    def ordered(cls):
        # Skips EMPTY, we only consider real cards
        return [cls(i) for i in range(cls.GUARD, cls.PRINCESS + 1)]

    @classmethod
    def space(cls) -> spaces.MultiBinary:
        log_size = math.ceil(math.log2(len(cls)))
        return spaces.MultiBinary(log_size)

    def serialize(self) -> np.ndarray:
        log_size = math.ceil(math.log2(len(self.__class__)))
        return utils.to_binary_array(self, log_size)


class Deck:
    card_frequency = {
        Card.GUARD: 5,
        Card.PRIEST: 2,
        Card.BARON: 2,
        Card.HANDMAID: 2,
        Card.PRINCE: 2,
        Card.KING: 1,
        Card.COUNTESS: 1,
        Card.PRINCESS: 1,
    }

    def __init__(self):
        self.cards = [
            card for card, freq in self.card_frequency.items() for i in range(freq)
        ]
        self.pointer = 0

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

    @classmethod
    def size(cls) -> int:
        return sum(cls.card_frequency.values())

    @classmethod
    def space(cls) -> spaces.MultiBinary:
        log_size = math.ceil(math.log2(cls.size()))
        return spaces.MultiBinary(log_size)

    def serialize(self) -> np.ndarray:
        log_size = math.ceil(math.log2(self.size()))
        return utils.to_binary_array(self.remaining(), log_size)

    def shuffle(self) -> None:
        self.pointer = 1  # Effectively discards one card so it's never observed
        self.np_random.shuffle(self.cards)

    def draw(self) -> Card:
        try:
            card = self.cards[self.pointer]
            self.pointer += 1
        except IndexError as e:
            raise IndexError("Deck has no more cards") from e
        return card

    def remaining(self) -> int:
        return len(self.cards) - self.pointer


class Discard:
    def __init__(self):
        self._discard: list[Card] = []

    @classmethod
    def space(cls) -> spaces.MultiBinary:
        return spaces.MultiBinary(Deck.size())

    @classmethod
    def size(cls) -> int:
        # One random card is removed from the deck each game
        return Deck.size() - 1

    def append(self, card: Card) -> None:
        self._discard.append(card)

    def reset(self) -> None:
        self._discard = []

    def serialize(self) -> np.ndarray:
        vector = [0] * Deck.size()

        card_counts = defaultdict(int)
        for card in self._discard:
            card_counts[card] += 1

        idx = 0
        for card in Card.ordered():
            freq = Deck.card_frequency[card]
            one_count = card_counts[card]

            for i in range(freq):
                if i < one_count:
                    vector[idx] = 1
                idx += 1

        return np.array(vector)


class Hand:
    MAX_SIZE = 2

    def __init__(self, cards: Sequence[Card] | None = None, max_size: int = MAX_SIZE):
        self._hand = [Card.EMPTY for _ in range(max_size)]

        if cards:
            if len(cards) > max_size:
                raise ValueError("Initialized hand with too many cards")

            for i in range(len(cards)):
                self._hand[i] = cards[i]

        self._index = 0

    def __iter__(self) -> Iterator[Card]:
        return iter(self._hand)

    @property
    def card(self) -> Card | None:
        """
        Shortcut to return the only card in the player's hand.
        """

        if all([c != Card.EMPTY for c in self]):
            raise ValueError("Expected player to have only one card, but found two")

        for c in self:
            if c != Card.EMPTY:
                return c

        return None

    @property
    def cards(self) -> list[Card]:
        return [card for card in self if card != Card.EMPTY]

    @property
    def full(self) -> bool:
        return Card.EMPTY not in self

    @property
    def vector(self) -> list[int]:
        return [card.value for card in self]

    @classmethod
    def parse(cls, vector: Sequence) -> Hand:
        cards = [Card[i] for i in vector]
        return cls(cards=cards, max_size=len(cards))

    @classmethod
    def space(cls) -> spaces.Tuple:
        return spaces.Tuple((Card.space(), Card.space()))

    def serialize(self) -> tuple:
        return (self._hand[0].serialize(), self._hand[1].serialize())

    def add(self, card: Card) -> None:
        """
        Add a card to the player's hand.
        """

        success = False
        for i in range(len(self._hand)):
            if self._hand[i] == Card.EMPTY:
                self._hand[i] = card
                success = True
                break

        if not success:
            raise ValueError("Player has a full hand and cannot accept more cards")

    def discard(self, card: Card) -> None:
        """
        Remove selected card from player's hand.
        """

        success = False
        for i in range(len(self._hand)):
            if self._hand[i] == card:
                self._hand[i] = Card.EMPTY
                success = True
                break

        if not success:
            raise ValueError(f"Player hand does not contain {card.name}")

    def draw(self, deck: Deck) -> None:
        if self.full:
            raise ValueError("Player has a full hand and cannot accept more cards")

        self.add(deck.draw())


class PriestInfo:
    def __init__(self, player: Player, num_players: int):
        self.num_players = num_players
        self._player = player
        self._priest_info: dict[Player, Card] = {}

    @classmethod
    def space(cls) -> spaces.Dict:
        return spaces.Dict({
            "target_1": Card.space(),
            "target_2": Card.space(),
            "target_3": Card.space(),
        })

    def serialize(self) -> dict:
        serialization = {f"target_{i}": Card.EMPTY.serialize() for i in range(1, constants.MAX_NUM_PLAYERS)}

        for player, card in self.priest_info().items():
            relative_pos = (player.position - self._player.position) % self.num_players
            serialization[f"target_{relative_pos}"] = card.serialize()

        return serialization

    def add(self, target: Player) -> None:
        if target == self._player:
            raise ValueError("Cannot play priest on oneself")

        if not target.active or target.safe:
            raise ValueError("Invalid priest target")

        if target.card is None:
            raise ValueError("Priest target does not have a card")

        if len(self._priest_info) == 3 and target not in self._priest_info:
            raise ValueError("Cannot add excess priest info")

        self._priest_info[target] = target.card

    def priest_info(self) -> dict[Player, Card]:
        """
        Use this method instead of directly accessing _priest_info in order
        to benefit from basic sanity checking.
        """

        if len(self._priest_info) > self.num_players - 1:
            raise ValueError("Excess priest info")

        to_remove = set()
        for player in self._priest_info:
            if not player.active:
                to_remove.add(player)

        for player in to_remove:
            del self._priest_info[player]

        return self._priest_info

    def remove(self, target: Player) -> None:
        del self._priest_info[target]

    def reset(self) -> None:
        self._priest_info = {}

    def swap(self, player1: Player, player2: Player) -> None:
        if player1 == self._player or player2 == self._player:
            raise ValueError("Card knowledge must be for other players")

        if (
            player1 in self._priest_info
            and player2 in self._priest_info
        ):
            cached = self._priest_info[player1]
            self._priest_info[
                player1
            ] = self._priest_info[player2]
            self._priest_info[player2] = cached
        elif player1 in self._priest_info:
            self._priest_info[
                player2
            ] = self._priest_info[player1]
            self.remove(player1)
        elif player2 in self._priest_info:
            self._priest_info[
                player1
            ] = self._priest_info[player2]
            self.remove(player2)

class Player:
    def __init__(self, position: int, num_players: int, name: str | None = None, hand: Hand | None = None):
        self.position = position
        self.num_players = num_players
        self.name = name if name is not None else f"Player {position + 1}"
        self.agent: Agent | None = None

        # State that resets each game via reset()
        self.active = True
        self.safe = False
        self.hand = hand if hand else Hand()
        self._priest_info = PriestInfo(self, num_players)
        self.play_history: list[Card] = []
        self.players_eliminated: set[Player] = set()

    @property
    def card(self) -> Card | None:
        return self.hand.card

    @property
    def interactive(self) -> bool:
        if self.agent is None:
            return False

        return isinstance(self.agent, HumanAgent)

    @property
    def last_played(self) -> Card:
        if not self.play_history:
            return Card.EMPTY

        return self.play_history[-1]

    @property
    def status_vector(self) -> tuple[int, int]:
        return (self.active, self.safe)

    @classmethod
    def space(cls) -> spaces.Dict:
        return spaces.Dict({
            "hand": Hand.space(),
            "priest_info": PriestInfo.space(),
            "player_state": cls.state_space(),
        })

    @classmethod
    def state_space(cls) -> spaces.Dict:
        return spaces.Dict({
            "active": spaces.Discrete(2),
            "safe": spaces.Discrete(2),
        })

    def serialize(self) -> dict:
        return {
            "hand": self.hand.serialize(),
            "priest_info": self._priest_info.serialize(),
            "player_state": self.serialize_state(),
        }

    def serialize_state(self) -> dict:
        return {
            "active": utils.to_binary_array(int(self.active), 1),
            "safe": utils.to_binary_array(int(self.safe), 1),
        }

    def reset(self):
        """
        Initialize player state for the beginning of a game.

        Note that the player's agent is not changed. It can be swapped with set_agent().
        """

        self.active = True
        self.safe = False
        self.hand = Hand()
        self.play_history: list[Card] = []
        self.players_eliminated: set[Player] = set()
        self._priest_info.reset()

    def set_agent(self, agent: Agent):
        self.agent = agent

    def eliminate(self) -> Card | None:
        """
        Mark the player as out of the game.

        Returns:
            The card in their hand. Note that a player can only have one card at
            elimination time.
        """

        self.active = False
        hand = self.hand
        self.hand = Hand()

        return hand.card

    def discard(self, card: Card) -> None:
        self.hand.discard(card)

    def draw(self, deck: Deck) -> None:
        self.hand.draw(deck)

    def play(self, card: Card) -> None:
        self.discard(card)
        self.play_history.append(card)

    def add_priest_target(self, target: Player) -> None:
        self._priest_info.add(target)

    def priest_info(self) -> dict[Player, Card]:
        return self._priest_info.priest_info()

    def remove_priest_target(self, target: Player) -> None:
        self._priest_info.remove(target)

    def swap_priest_knowledge(self, player1: Player, player2: Player) -> None:
        self._priest_info.swap(player1, player2)

    def __repr__(self):
        return f"Player: {self.name}"


class Game:
    def __init__(self, env_class: Type[gym.Env], agent_classes: list[Type[Agent]]):
        self.env = env_class(num_players=len(agent_classes))
        # TODO initialize player agents once env is initialized

        self.players_done = [False for _ in range(len(self.env.players))]
        self.obs_history: np.ndarray = []

        self.obs = np.array([])  # Necessary to provide a type hint
        self.player_done = False
        self.reward = 0
        self.info: dict = {}

    def step(self, action_id: int | None = None):
        if self.player_done:
            self.players_done[self.env.current_player.position] = True
            self.obs, self.reward, self.player_done, self.info = self.env.next_player()
            return

        if not action_id:
            try:
                action_id = self.env.current_player.agent.predict(self.obs)
            except AttributeError as e:
                raise ValueError("Must initialize player agent") from e

        self.obs, self.reward, self.player_done, self.info = self.env.step(action_id)

    def reset(self):
        self.obs_history = []
        self.obs = self.env.reset()
        self.player_done = False
        self.reward = 0
        self.info: dict = {}

    def run(self):
        self.reset()
        while not all(self.players_done):
            self.step()
