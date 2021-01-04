from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gym
import numpy as np
from gym.utils import seeding


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


class Hand:
    MAX_SIZE = 2

    def __init__(self, cards: Sequence[Card] = None, max_size: int = MAX_SIZE):
        self._hand = [Card.EMPTY for i in range(max_size)]

        if cards:
            if len(cards) > max_size:
                raise ValueError("Initialized hand with too many cards")

            for i in range(len(cards)):
                self._hand[i] = cards[i]

        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index == len(self._hand):
            raise StopIteration

        card = self._hand[self._index]
        self._index += 1
        return card

    @classmethod
    def parse(cls, vector: Sequence) -> Hand:
        cards = [Card[i] for i in vector]
        return cls(cards=cards, max_size=len(cards))

    @property
    def card(self) -> Optional[Card]:
        """
        Shortcut to return the only card in the player's hand.
        """

        if all([c != Card.EMPTY for c in self]):
            raise ValueError("Expected player to have only one card, but found two")

        for c in self:
            if c != Card.EMPTY:
                return c

        return None
        # raise ValueError("Expected player to have one card, but found empty hand")

    @property
    def full(self) -> bool:
        return Card.EMPTY not in self

    @property
    def vector(self) -> List[int]:
        return [card.value for card in self]

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


class Player:
    def __init__(self, position: int, name: str, hand: Hand = None):
        self.position = position
        self.name = name
        self.active = True
        self.safe = False
        self.hand = hand if hand else Hand()
        self.play_history: List[Card] = []
        self.players_eliminated: Set[Player] = set()

        self._priest_targets: Dict[Player, Card] = {}

    @property
    def card(self) -> Optional[Card]:
        return self.hand.card

    @property
    def last_played(self) -> Card:
        if not self.play_history:
            return Card.EMPTY

        return self.play_history[-1]

    @property
    def status_vector(self) -> Tuple[int, int]:
        return (self.active, self.safe)

    def eliminate(self) -> Optional[Card]:
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
        if target == self:
            raise ValueError("Cannot play priest on oneself")

        if not target.active or target.safe:
            raise ValueError("Invalid priest target")

        if target.card is None:
            raise ValueError("Priest target does not have a card")

        if len(self._priest_targets) == 3 and target not in self._priest_targets:
            raise ValueError("Cannot add excess priest info")

        self._priest_targets[target] = target.card

    def priest_info(self) -> Dict[Player, Card]:
        """
        Use this method instead of directly accessing _priest_targets in order
        to benefit from basic sanity checking.
        """

        if len(self._priest_targets) > 3:
            raise ValueError("Excess priest info")

        for player in self._priest_targets:
            if not player.active:
                import ipdb; ipdb.set_trace()
                raise ValueError("Inactive priest target")

        return self._priest_targets

    def remove_priest_target(self, target: Player) -> None:
        del self._priest_targets[target]

    def __repr__(self):
        return f"Player: {self.name}"


class Game:
    def __init__(self, env_class: gym.Env, agent_classes: List):
        self.env = env_class(num_players=len(agent_classes))
        self.agents = [cls(self.env) for cls in agent_classes]
        self.agents_done = [False for i in range(len(self.agents))]
        self.obs_history: np.array = []

        self.obs = np.array([])  # Necessary to provide a type hint
        self.player_done = False
        self.reward = 0
        self.info: Dict = {}

    @property
    def curr_agent(self):
        return self.agents[self.env.current_player.position]

    def step(self, action_id: int = None):
        if self.player_done:
            self.agents_done[self.env.current_player.position] = True
            self.obs, self.reward, self.player_done, self.info = self.env.next_player()
            return

        if not action_id:
            action_id = self.curr_agent.predict(self.obs)

        self.obs, self.reward, self.player_done, self.info = self.env.step(action_id)

    def reset(self):
        self.obs_history = []
        self.obs = self.env.reset()
        self.player_done = False
        self.reward = 0
        self.info: Dict = {}

    def run(self):
        self.reset()
        while not all(self.agents_done):
            self.step()
