from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_love_letter.agents import RandomAgent
from gym_love_letter.engine import Card, Deck, Discard, Player
from gym_love_letter.envs.actions import (Action, ActionWrapper,
                                          generate_actions)
from gym_love_letter.envs.observations import Observation


if TYPE_CHECKING:
    from gym_love_letter.agents import Agent


class InvalidPlayError(ValueError):
    pass


class Rewards:
    @staticmethod
    def simple_turn_reward(env) -> float:
        if not env.current_player.active:
            return 0
        return 1

    @staticmethod
    def game_completion_reward(env) -> float:
        if not env.current_player.active:
            return 0

        reward = 1 / len(env.active_players)

        if env.game_over:
            reward += 5

        return reward

    @staticmethod
    def game_won_reward(env) -> float:
        if env.game_over and env.current_player.active:
            return 1

        return 0

    @staticmethod
    def fast_elimination_reward(env) -> float:
        player = env.current_player

        # Reward for surviving a round (after having made a move)
        reward = 1 if player.active and len(player.play_history) > 0 else 0

        # Extra reward for eliminating other players
        reward += len(player.players_eliminated) * 3
        player.players_eliminated = set()  # Reset cache. TODO use a counter for this instead

        # Extra reward for each future action that was prevented
        if env.game_over and player.active:
            reward += 10  # Always reward winning
            reward += env.deck.remaining()

        NORMALIZE_BY = 25

        return reward / NORMALIZE_BY


class LoveLetterBaseEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_players: int = 2,
        randomize_player_count: bool = False,
        agent_classes: Sequence[type[Agent]] | None = None,
        reward_fn: Callable[[LoveLetterBaseEnv], float] = Rewards.fast_elimination_reward,
        player_names: list[str] | None = None,
    ):
        # If we want to use stable_baselines, our action space cannot be a tuple or Dict
        self.actions = generate_actions(Observation.MAX_NUM_PLAYERS)
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.actions))

        self.observation_space = Observation.space(int(self.action_space.n))

        self.num_players = num_players
        self.randomize_player_count = randomize_player_count
        self.reward = reward_fn

        # Player names are auto-generated if not specified
        if player_names is None:
            player_names = []

        self.players = [
            Player(i, name=name) for i, name in itertools.zip_longest(range(self.num_players), player_names)
        ]
        self.current_player = self.players[0]
        self.starting_player = self.current_player
        self.winners: list[Player] = []

        # Make a new deck
        self.deck = Deck()

        # Clear action history & discard pile
        self.action_history: list[ActionWrapper] = []
        self.discard_pile = Discard()

        self.game_over = False

        # Now that the environment has been initialized, provide a reference
        # to each player agent. This allows agents to access the env's
        # valid action mask.
        self._agents: Sequence[Agent] = []
        if agent_classes is None:
            agent_classes = [RandomAgent] * self.num_players
        agents = [cls(self) for cls in agent_classes]

        self.set_agents(agents)

    def set_agents(self, agents: Sequence[Agent]) -> None:
        if len(agents) != self.num_players:
            raise ValueError("Must have same number of agents as players")
        self._agents = agents
        for agent, player in zip(self._agents, self.players):
            player.set_agent(agent)

    @property
    def active_players(self) -> list[Player]:
        return [player for player in self.players if player.active]


    def valid_action_mask(self) -> np.ndarray:
        mask = [self._valid_action(action) for action in self.actions]
        return np.array(mask, dtype=np.int8)

    @property
    def valid_actions(self) -> list[Action]:
        """
        The set of valid actions for the current player.

        Note that action target positions are indexed with respect to the
        current player's position, not the global indexes managed by the env.
        """

        return [a for a in self.actions if self._valid_action(a)]

    def _valid_targets(self, card: Card) -> list[int | None]:
        """
        Returns the positions of valid player targets for the given card.

        IMPORTANT: In this context, None means the card is valid to play with no target.
        Many cards require a target _unless_ there's no legal target, in which case they
        can be played with no target.
        """

        if not card.takes_target:
            return [None]

        # Never valid to target an inactive or safe player
        players = [p for p in self.active_players if not p.safe]

        # Only Prince can be used to target oneself
        if card != Card.PRINCE:
            players = [p for p in players if p is not self.current_player]

        # NB: All player positions are normalized relative to the current player's
        player_positions: list[int | None] = [
            (p.position - self.current_player.position) % self.num_players
            for p in players
        ]
        if len(players) == 0:
            player_positions = [None]

        return player_positions


    def _valid_action(self, action: Action) -> bool:
        # The empty card is a sentinel value for vectors. It cannot be played.
        if action.card == Card.EMPTY:
            return False

        # Cannot play a card you don't have in your hand
        if action.card not in self.current_player.hand:
            return False

        if Card.COUNTESS in self.current_player.hand and action.card in [
            Card.PRINCE,
            Card.KING,
        ]:
            return False

        if action.card.takes_target:
            return action.target in self._valid_targets(action.card)

        return True

    def decode_action(self, action_id: int) -> Action:
        """
        All actions are chosen as if the current player is in the 0th position
        at the table. However, from the perspective of the environment, all
        players have a fixed position. This method must be used to translate
        the agent's chosen action into the "global" equivalent.
        """

        action = self.actions[action_id]

        # Checks that action targets are active and unprotected, and that there
        # is a target when there ought to be.
        if action not in self.valid_actions:
            # I think this is just a hack for now: immediately end the game if play was invalid
            # import ipdb; ipdb.set_trace()
            raise InvalidPlayError(f"Invalid action {action} played")

        if action.target is not None:
            action_copy = copy.deepcopy(action)

            # Decode the action's target into the global index.
            action_copy.target = (action.target + self.current_player.position) % self.num_players
            return action_copy
        else:
            return action

    def discard(self, player: Player, card: Card) -> None:
        player.discard(card)
        self.discard_pile.append(card)

        # Update priest info for all other players
        for p in self.active_players:
            if p != player:
                if p.priest_info().get(player, None) == card:
                    p.remove_priest_target(player)

    def eliminate(self, player: Player) -> None:
        card = player.eliminate()
        if card is not None:
            self.discard_pile.append(card)

        # Add eliminated player to current player's elimination cache
        if player != self.current_player:
            self.current_player.players_eliminated.add(player)

        # Update priest info for all other players
        for p in self.active_players:
            if p != player:
                # TODO: Address private attribute access
                if player in p._priest_targets:
                    p.remove_priest_target(player)

    def _reset(self) -> Observation:
        # Clear winners from last game
        self.winners = []
        self.game_over = False

        # Reset player state
        for p in self.players:
            p.reset()

        if self.randomize_player_count:
            active_players = self.np_random.integers(2, self.num_players, endpoint=True)

            # Skip the active players from the front of the list.
            # We're assuming the main player is always in position 0.
            for p in self.players[active_players:]:
                p.active = False

        self.current_player = self.np_random.choice(self.active_players)
        self.starting_player = self.current_player

        # Clear action history & discard pile
        self.action_history = []
        self.discard_pile.reset()

        # Shuffle and deal
        self.deck.shuffle()
        for player in self.players:
            player.draw(self.deck)
            if player == self.current_player:
                player.draw(self.deck)

        return self.observe()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)  # Farama requires this to initialize np_random
        deck_seed = int(self.np_random.integers(2 ** 63))
        self.deck.seed(deck_seed)
        return self._reset().vector, {}

    def observe(self) -> Observation:
        return Observation(
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

    def play(self, card: Card) -> None:
        # Player discards the card they play
        self.current_player.play(card)
        self.discard_pile.append(card)

        for p in self.active_players:
            if p != self.current_player:
                if p.priest_info().get(self.current_player, None) == card:
                    p.remove_priest_target(self.current_player)

    def _check_game_over(self) -> None:
        # If no cards remain, compare hands
        if self.deck.remaining() == 0:
            max_card = max([p.card for p in self.active_players if p.card is not None])
            for player in self.active_players:
                # Only player(s) with the max value card remain
                if player.card != max_card:
                    self.eliminate(player)

        if len(self.active_players) <= 0:
            raise RuntimeError("No players remaining")

        # If the game has ended, determine winners
        if len(self.active_players) == 1 or self.deck.remaining() == 0:
            self.game_over = True
            self.winners = self.active_players

    def _next_player(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        # NOTE: The current player may not actually be active, but we still need
        # to provide that player done + reward.
        self.current_player = self.players[
            (self.current_player.position + 1) % self.num_players
        ]

        # Unmark new current player as safe
        self.current_player.safe = False

        # Determine the reward of the current agent
        reward = self.reward(self)

        # Reset new current player's elimination cache after reward has been determined
        self.current_player.players_eliminated = set()

        # Determine whether the new current player's game has ended
        done = not self.current_player.active or self.game_over
        if not done:
            self.current_player.draw(self.deck)

        obs = self.observe()
        return obs.vector, reward, done, False, {"observation": obs}

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Validates and reindexes action
        action = self.decode_action(action_id)
        card = action.card
        discarding_player: Player | None = None
        discard: Card | None = None

        self.play(card)

        # Define target when appropriate
        if card.takes_target:
            if action.target is not None:
                target = self.players[action.target]

                # We expect the target to have a card at this point
                target_card = target.card
                if target_card is None:
                    # Saw this happen on 11/17
                    import ipdb; ipdb.set_trace()
                    raise RuntimeError(f"Target player {target.position} has no card")

                if card == Card.GUARD:
                    # If card guessed correctly, the player is out!
                    if target_card == action.guess:
                        discarding_player = target
                        discard = target_card
                        self.eliminate(target)

                elif card == Card.PRIEST:
                    self.current_player.add_priest_target(target)

                elif card == Card.BARON:
                    current_player_card = self.current_player.card
                    if current_player_card is None:
                        breakpoint()
                        raise RuntimeError(f"Player {self.current_player.position} has no card")

                    # The player with the lower card value is out. If tie, nothing happens.
                    if current_player_card > target_card:
                        discarding_player = target
                        discard = target_card
                        self.eliminate(target)
                    elif current_player_card < target_card:
                        discarding_player = self.current_player
                        discard = current_player_card
                        self.eliminate(self.current_player)

                elif card == Card.PRINCE:
                    self.discard(target, target_card)
                    discarding_player = target
                    discard = target_card

                    # If the player discards the princess, they lose
                    if target_card == Card.PRINCESS:
                        self.eliminate(target)
                    else:
                        try:
                            target.draw(self.deck)
                        except IndexError:
                            self.eliminate(target)

                elif card == Card.KING:
                    # Swap card
                    current_player_hand = self.current_player.hand
                    self.current_player.hand = target.hand
                    target.hand = current_player_hand

                    self.current_player.add_priest_target(target)
                    target.add_priest_target(self.current_player)

                    for p in self.active_players:
                        if p != self.current_player and p != target:
                            p.swap_priest_knowledge(self.current_player, target)

        elif card == Card.HANDMAID:
            self.current_player.safe = True

        elif card == Card.COUNTESS:
            # Nothing special to do in this case
            pass

        else:
            # Should never get here
            # TODO remove
            breakpoint()
            raise InvalidPlayError(f"Invalid action {action} played")

        self.action_history.append(
            ActionWrapper(action, self.current_player, discarding_player, discard)
        )

        self._check_game_over()

        # IMPORTANT: After a card has been played, we change the current player
        # and compute the observation, reward, etc, from that new perspective.
        return self._next_player()

    @classmethod
    def load(vector: np.array) -> LoveLetterBaseEnv:
        pass


class LoveLetterMultiAgentEnv(LoveLetterBaseEnv):
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        options = options or {}
        training = options.get("training", True)

        obs, info = super().reset(seed=seed)

        if training:
            # The setup is only valid for training if the training agent hasn't
            # already been eliminated before its first move!
            valid = not self.game_over and self.current_player.active

            while not valid:
                obs, info = super().reset()

                # Assumes that the training agent is in the 0th position
                # TODO: Assure that we don't loop forever
                terminated = False
                while self.current_player.position != 0:
                    if not terminated:
                        player_agent = self.current_player.agent
                        mask = self.valid_action_mask()
                        action_id, _ = player_agent.predict(obs, action_masks=mask)
                        obs, _, terminated, _, _ = super().step(action_id)
                    else:
                        obs, _, terminated, _, _ = super()._next_player()

                valid = not self.game_over and self.current_player.active

        return obs, info

    def step(
        self, action_id: int, full_cycle: bool = True
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # sanity check
        if self.game_over:
            raise Exception("game over already?")

        # Make the current player's move
        try:
            obs, reward, terminated, truncated, info = super().step(action_id)
        except InvalidPlayError:
            # import ipdb; ipdb.set_trace()
            obs = self.observe()
            print(f"Obs: {obs}")
            print(f"Action: {self.actions[action_id]}")
            print(f"Valid Actions: {self.valid_actions}")
            # TODO: Deal with this magic number
            return obs.vector, -10, True, False, {"observation": obs}

        if full_cycle:
            # Make a move for every other agent in the game to come back around to the current player
            for i in range(self.num_players - 1):
                if not terminated:
                    player_agent = self.current_player.agent
                    mask = self.valid_action_mask()
                    action_id, _ = player_agent.predict(obs, action_masks=mask)
                    obs, reward, terminated, truncated, info = super().step(action_id)
                else:
                    obs, reward, terminated, truncated, info = super()._next_player()

        return obs, reward, terminated, truncated, info

    def protected_step(
        self, action_id: int, *args, **kwargs
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        This wrapper around step that is allowed to raise an exception if a
        move is invalid. Exceptions cannot be handled during training, so
        step() can't raise any. Here, we can.
        """

        if not self.valid_action_mask()[action_id]:
            raise InvalidPlayError("Invalid action played")
        return self.step(action_id, *args, **kwargs)
