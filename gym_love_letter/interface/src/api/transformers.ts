import {
  Card,
  GameAction,
  GameState,
  Play,
  Player,
  PriestInfo,
  RawApiGameAction,
  RawApiGameState,
  RawApiPlay,
  RawApiPlayer,
  RawApiPriestInfo
} from '../types'

export const buildGameState = (data: RawApiGameState): GameState => {
  return {
    cardsRemaining: data.cardsRemaining,
    currentPlayer: data.currentPlayer,
    discard: data.discard.map(cardId => new Card(cardId)),
    gameOver: data.gameOver,
    hand: data.hand.map(cardId => new Card(cardId)),
    players: data.players.map(player => buildPlayer(player)),
    plays: data.plays.map(play => buildPlay(play)),
    priestInfo: buildPriestInfo(data.priestInfo),
    validActions: data.validActions.map(action => buildGameAction(action)),
    winners: data.winners
  }
}

const buildGameAction = (data: RawApiGameAction): GameAction => {
  return {
    card: new Card(data.card),
    guess: data.guess ? new Card(data.guess) : null,
    id: data.id,
    target: data.target
  }
}

const buildPlay = (data: RawApiPlay): Play => {
  return {
    action: buildGameAction(data.action),
    player: data.player,
    discard: data.discard ? new Card(data.discard) : null,
    discardingPlayer: data.discardingPlayer
  }
}

const buildPlayer = (data: RawApiPlayer): Player => {
  return {
    active: data.active,
    hand: data.hand?.map(cardId => new Card(cardId)),
    name: data.name,
    position: data.position,
    safe: data.safe,
    wins: 0
  }
}

const buildPriestInfo = (data: RawApiPriestInfo): PriestInfo => {
  const newData: {[key: number]: Card} = {}
  for (const [key, value] of Object.entries(data)) {
    newData[Number.parseInt(key)] = new Card(value)
  }
  return newData
}
