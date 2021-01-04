from dataclasses import dataclass
from typing import List, Optional

from gym_love_letter.engine import Card, Player


@dataclass
class Action:
    card: Card
    target: Optional[int] = None
    guess: Optional[Card] = None
    _id: int = -1


@dataclass
class ActionWrapper:
    """
    Includes information not needed (or provided) for training.
    """

    action: Action
    player: Player
    discarding_player: Optional[Player] = None
    discard: Optional[Card] = None


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
