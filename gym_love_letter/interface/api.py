from flask import Blueprint, jsonify
from marshmallow import Schema, fields

from gym_love_letter.engine import Card
from gym_love_letter.envs.base import InvalidPlayError, LoveLetterMultiAgentEnv


class CardSchema(Schema):
    name = fields.Str()
    value = fields.Int()


class PlayerStatusSchema(Schema):
    position = fields.Int()
    name = fields.Str()
    active = fields.Boolean()
    safe = fields.Boolean()


class PlayerFullSchema(PlayerStatusSchema):
    hand = fields.Method("get_player_hand")

    def get_player_hand(self, obj):
        return [card.value for card in obj.hand if card != Card.EMPTY]


class ActionSchema(Schema):
    card = fields.Pluck(CardSchema, "value")
    target = fields.Int()
    guess = fields.Pluck(CardSchema, "value")
    _id = fields.Int(data_key="id")


class PlaySchema(Schema):
    action = fields.Nested(ActionSchema)
    player = fields.Pluck(PlayerStatusSchema, "position")
    discarding_player = fields.Pluck(
        PlayerStatusSchema, "position", data_key="discardingPlayer"
    )
    discard = fields.Pluck(CardSchema, "value")


class PriestInfoSchema(Schema):
    _priest_targets = fields.Dict(
        keys=fields.Pluck(PlayerStatusSchema, "position"),
        values=fields.Pluck(CardSchema, "value"),
    )


class ObservationSchema(Schema):
    players = fields.List(fields.Nested(PlayerStatusSchema))
    curr_player = fields.Pluck(PlayerStatusSchema, "position", data_key="currentPlayer")
    hand = fields.Method("get_human_player_hand")
    priestInfo = fields.Method("get_priest_info")
    discard = fields.List(fields.Pluck(CardSchema, "value"))
    cardsRemaining = fields.Method("get_cards_remaining")
    plays = fields.List(fields.Nested(PlaySchema))
    validActions = fields.Method("get_valid_actions")
    game_over = fields.Boolean(data_key="gameOver")
    winners = fields.List(fields.Pluck(PlayerStatusSchema, "position"))

    def get_human_player_hand(self, obj):
        return [card.value for card in obj.players[0].hand if card != Card.EMPTY]

    def get_cards_remaining(self, obj):
        return obj.deck.remaining()

    def get_priest_info(self, obj):
        return PriestInfoSchema().dump(obj.players[0])["_priest_targets"]

    def get_valid_actions(self, obj):
        if obj.curr_player.position != 0:
            return []
        return ActionSchema().dump(obj.valid_actions, many=True)


class GameOverSchema(ObservationSchema):
    players = fields.List(fields.Nested(PlayerFullSchema))


def make_api(env: LoveLetterMultiAgentEnv):
    api = Blueprint("api", __name__)

    @api.route("/reset")
    def reset():
        env.reset()
        return ObservationSchema().dump(env.observe())

    @api.route("/current_state")
    def current_state():
        schema = ObservationSchema() if env.players[0].active else GameOverSchema()
        return schema.dump(env.observe())

    @api.route("/valid_actions")
    def valid_actions():
        return jsonify(ActionSchema(many=True).dump(env.valid_actions))

    @api.route("/step")
    def step():
        # Find the next active player to take a step for.
        # Break if no players are active.
        starting_player = env.current_player
        while not env.current_player.active:
            env._next_player()
            if env.current_player == starting_player:
                break

        # Raise exception if no players are active (should never happen)
        if not env.current_player.active:
            raise RuntimeError("No active players")

        # Human player just returns current observation, no action taken.
        # Note that the player must be active if we've gotten here.
        if env.current_player.position == 0:
            return ObservationSchema().dump(env.observe())

        count = 0
        LIMIT = 100

        while count < LIMIT:
            try:
                agent = env.agents[env.current_player.position]
                action_id, _state = agent.predict(
                    env.observe().vector, action_masks=env.valid_action_mask()
                )
                obs, reward, done, info = env.protected_step(
                    action_id, full_cycle=False
                )
                break
            except InvalidPlayError:
                count += 1

        if count == LIMIT:
            import ipdb; ipdb.set_trace()
            raise RuntimeError("No valid play found")

        # Step forward so that the next API call will be for an active player
        while not env.current_player.active:
            env._next_player()

        schema = ObservationSchema() if env.players[0].active else GameOverSchema()

        return schema.dump(env.observe())

    @api.route("/step/<int:action_id>")
    def step_action(action_id):
        try:
            env.protected_step(action_id, full_cycle=False)
        except InvalidPlayError as e:
            return {"error": e}, 400

        # Step forward so that the next API call will be for an active player
        while not env.current_player.active:
            env._next_player()

        schema = ObservationSchema() if env.players[0].active else GameOverSchema()

        return schema.dump(env.observe())

    return api
