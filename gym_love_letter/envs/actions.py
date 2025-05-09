from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from gymnasium import spaces

from gym_love_letter import utils
from gym_love_letter.engine import Card, Deck, Discard, Player


# Four targets or no target, so 3 bits needed to binary encode
LOG_TARGET = 3


@dataclass
class Action:
    card: Card
    target: Optional[int] = None
    guess: Optional[Card] = None
    _id: int = -1

    @classmethod
    def space(cls) -> spaces.MultiBinary:
        card_encoding = Card.space().n
        guess_encoding = Card.space().n - 1  # Minus 1 because GUARD isn't a valid guess
        total = card_encoding + LOG_TARGET + guess_encoding

        return spaces.MultiBinary(total)

    def serialize(self) -> np.ndarray:
        card_vec = self.card.serialize()

        # Convert None target to 0, pad int targets by 1 to deduplicate
        target = self.target + 1 if self.target is not None else 0
        target_vec = utils.to_binary_array(target, LOG_TARGET)

        guess = self.guess
        assert guess is not Card.GUARD

        if guess is None:
            guess = Card.EMPTY.value
        else:
            guess = guess - 1
        guess_vec = utils.to_binary_array(guess, Card.space().n - 1)

        return np.array(card_vec + target_vec + guess_vec)


@dataclass
class ActionWrapper:
    """
    Includes information not needed (or provided) for training.
    """

    action: Action
    player: Player
    discarding_player: Optional[Player] = None
    discard: Optional[Card] = None


class History:
    def __init__(self):
        self._history: list[ActionWrapper] = []

    @classmethod
    def space(cls) -> spaces.MultiBinary:
        return spaces.MultiBinary(Action.space().n * Discard.size())

    def __len__(self) -> int:
        return len(self._history)

    def append(self, action: ActionWrapper) -> None:
        self._history.append(action)

    def reset(self) -> None:
        self._history = []

    def serialize(self) -> np.ndarray:
        vector = []
        vector_parts = [action.action.serialize() for action in self._history]
        if len(vector_parts) > 0:
            vector = np.concatenate(vector_parts)

        remaining = self.space().n - len(vector)
        padding = np.zeros(remaining)
        vector = np.concatenate([vector, padding])

        return vector


def _generate_guard_actions(num_players: int) -> List[Action]:
    actions = []
    for player in range(num_players):
        for card in Card:
            if card not in [Card.EMPTY, Card.GUARD]:
                actions.append(Action(Card.GUARD, player, card))

    # Throw away card with no effect if there are no valid targets
    actions.append(Action(Card.GUARD))

    return actions


def _generate_priest_actions(num_players: int) -> List[Action]:
    actions = []
    for player in range(num_players):
        actions.append(Action(Card.PRIEST, player))

    # Throw away card with no effect if there are no valid targets
    actions.append(Action(Card.PRIEST))

    return actions


def _generate_baron_actions(num_players: int) -> List[Action]:
    actions = []
    for player in range(num_players):
        actions.append(Action(Card.BARON, player))

    # Throw away card with no effect if there are no valid targets
    actions.append(Action(Card.BARON))

    return actions


def _generate_prince_actions(num_players: int) -> List[Action]:
    actions = []
    for player in range(num_players):
        actions.append(Action(Card.PRINCE, player))

    return actions


def _generate_king_actions(num_players: int) -> List[Action]:
    actions = []
    for player in range(num_players):
        actions.append(Action(Card.KING, player))

    # Throw away card with no effect if there are no valid targets
    actions.append(Action(Card.KING))

    return actions


def generate_actions(num_players: int) -> List[Action]:
    # Empty action is used to represent actions yet to be taken in observation
    # space (i.e. future turns in action_history)
    # TODO: Decide if this can be deleted
    actions = [Action(Card.EMPTY)]

    actions += _generate_guard_actions(num_players)
    actions += _generate_priest_actions(num_players)
    actions += _generate_baron_actions(num_players)
    actions += [Action(Card.HANDMAID)]
    actions += _generate_prince_actions(num_players)
    actions += _generate_king_actions(num_players)
    actions += [Action(Card.COUNTESS)]

    # Use the action's position in the list as its id
    for i in range(len(actions)):
        actions[i]._id = i

    return actions
