Gym Love Letter
---

An OpenAI gym environment for Love Letter, a competitive multiplayer strategy game.

You're probably looking for `LoveLetterMultiAgentEnv` from `gym_love_letter.envs`.

## Installation

TODO

## Basic Usage

```python
from gym_love_letter.agents import HumanAgent, RandomAgent
from gym_love_letter.envs import LoveLetterBaseEnv

# Initialize the environment.
# By default the environment assigns RandomAgents to each player.
env = LoveLetterBaseEnv(num_players=2)

# Set the agents as desired. Agents are assigned to players by corresponding position.
agents = [HumanAgent(env), RandomAgent(env)]
env.set_agents(agents)

# Now you can use the environment like you normally would
obs = env.reset()

# Let's check who the current player is
player = env.current_player
position = player.position

# Get the set of valid actions
actions = env.valid_actions()
# TODO no private attribute access
action_id = actions[0]._id  # pick one arbitrarily

obs, reward, done, info = env.step(action_id)

# You could also pass the observation to the player's agent
# and let it choose an action.
env.current_player.agent.predict(obs)

# TODO Explain LoveLetterMultiAgentEnv as well
```

## Architecture

### The Environment

A Love Letter gym environment is slightly different from other common gym environments
because it is "multi-agent." The observation the environment provides to each agent is
a function of the actions taken by the other agents in the game. Additionally, each
agent has access to different private information, and thus must receive a different
observation. When one agent takes an action, the environment isn't ready to provide that
agent with its next observation, but it _is_ able to give an observation to the next
agent in the turn order. Therefore, the environment keeps track of a mapping of agents
to "players" in the env, where each player has a position in the game's turn order. This
allows the environment to specify which agent should be the recipient of the observation
returned by `step()` or `reset()`.

In an offline setting, e.g. training or testing, the environment can automatically step
back around to the target agent by calling `predict()` for all non-target agents. In an
online setting where agents are engaged in interactive play, the environment can provide
appropriate player-specific observations between agent turns, allowing for game UIs to
update state continually for all players.

### Players and Agents

A "player" is effectively just a seat at the game table. An "agent" is any class that
subclasses the `Agent` ABC in `gym_love_letter.agents.abstract` and implements the
[Stable Baselines 3 algorithm API][sb3], particularly `predict()`. For interactive play,
the `HumanAgent` subclass may be used to indicate that the environment must wait for
input, since `predict()` is unavailable.

In order to support [invalid action masking][iam], the environment provides an extra
kwarg called `action_masks` to agent `predict()` calls in offline mode. This gives the
target agent's opponents an opportunity to take advantage of action masking, but it also
means that the agent must be prepared to receive the kwarg regardless. The expected
pattern to support algorithms that don't expect `action_masks` is to wrap them in
`Agent` subclasses:

```python
from stable_baselines3 import PPO

from gym_love_letter.agents import Agent


class MyStableBaselinesAgent(PPO, Agent):
    def predict(self, observation, action_masks=None, **kwargs)
      # Ignore action_masks and pass the remaining kwargs to the SB3 algorithm
      super().predict(observation, **kwargs)
```

In online mode, all agent steps are handled externally from the env, so it is up to each
agent to decide if it wants to request the invalid action mask for its state.

[sb3]: https://stable-baselines3.readthedocs.io/en/master/modules/base.html
[iam]: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
