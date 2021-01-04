enum CardValue {
  GUARD = 1,
  PRIEST,
  BARON,
  HANDMAID,
  PRINCE,
  KING,
  COUNTESS,
  PRINCESS
}

export class Card {
  name: string
  value: CardId

  static needTargets = ["GUARD", "PRIEST", "BARON", "PRINCE", "KING"]
  static CardValue = CardValue

  constructor(value: CardId) {
    this.name = CardValue[value]
    this.value = value
  }

  hasTarget(): boolean {
    return Card.needTargets.includes(this.name)
  }
}

type CardId = number
type PlayerPosition = number

export interface GameAction {
  card: Card
  guess: Card | null
  id: number
  target: PlayerPosition | null
}

export interface Play {
  action: GameAction
  player: PlayerPosition
  discard: Card | null
  discardingPlayer: PlayerPosition | null
}

export interface Player {
  active: boolean
  hand?: Card[]
  name: string
  position: PlayerPosition
  safe: boolean
  wins: number
}

export interface PriestInfo {
  [target: number]: Card
}

export interface GameState {
  cardsRemaining: number
  currentPlayer: PlayerPosition | null
  discard: Card[]
  gameOver: boolean
  hand: Card[]
  plays: Play[]
  players: Player[]
  priestInfo: PriestInfo
  validActions: GameAction[]
  winners: PlayerPosition[]
}

/*---------------------------------------------------------------------------*/
// Interfaces for raw API json responses, must be converted to classes
/*---------------------------------------------------------------------------*/

export interface RawApiGameAction {
  card: CardId
  guess: CardId | null
  id: number
  target: PlayerPosition | null
}

export interface RawApiPlay {
  action: RawApiGameAction
  player: PlayerPosition
  discard: CardId | null
  discardingPlayer: PlayerPosition | null
}

export interface RawApiPlayer {
  active: boolean
  hand?: CardId[]
  name: string
  position: PlayerPosition
  safe: boolean
}
export interface RawApiPriestInfo {
  [target: number]: CardId
}

export interface RawApiGameState {
  cardsRemaining: number
  currentPlayer: PlayerPosition | null
  discard: CardId[]
  gameOver: boolean
  hand: CardId[]
  players: RawApiPlayer[]
  plays: RawApiPlay[]
  priestInfo: RawApiPriestInfo
  validActions: RawApiGameAction[]
  winners: PlayerPosition[]
}
