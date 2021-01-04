import { produce } from 'immer'
import { current } from 'immer'
import { applyMiddleware, createStore, Reducer } from 'redux'
import thunk from 'redux-thunk'

import { Card, GameState, Player } from '../types'
import { ActionTypes, AppAction } from './actions'


export interface State extends GameState {
  chosenCard: Card | null
  guess: Card | null
  players: Player[]
  running: boolean
  target: Player | null
  timeouts: ReturnType<typeof setTimeout>[]
  watching: boolean
}

const initialState: State = {
  chosenCard: null,
  guess: null,
  running: false,
  target: null,
  timeouts: [],

  cardsRemaining: 0,
  currentPlayer: null,
  discard: [],
  gameOver: false,
  hand: [],
  players: [],
  plays: [],
  priestInfo: [],
  validActions: [],
  watching: false,
  winners: [],
}

const reducer: Reducer<State, AppAction> = produce((state: State, action: AppAction) => {
  switch (action.type) {
    case ActionTypes.CHOOSE_CARD:
      state.chosenCard = action.card
      break

    case ActionTypes.CHOOSE_GUESS:
      state.guess = action.guess
      break

    case ActionTypes.CHOOSE_TARGET:
      state.target = action.target
      break

    case ActionTypes.REGISTER_TIMEOUT:
      state.timeouts.push(action.timeout)
      break

    case ActionTypes.RESET:
      state = {
        ...initialState,
        ...action.data,
        players: action.data.players.map(p => {
          const { wins, ...rest } = p
          return {...rest, wins: state.players[p.position]?.wins ?? wins} as Player
        }),
        running: true
      }
      return state

    case ActionTypes.UPDATE:
      state = {
        ...state,
        ...action.data,
        players: action.data.players.map(p => {
          const { wins, ...rest } = p
          return {...rest, wins: state.players[p.position].wins} as Player
        })
      }
      for (const winner of state.winners) {
        state.players[winner].wins += 1
      }
      return state

    case ActionTypes.WATCH:
      state.watching = true
      break
  }
}, initialState)

export default createStore(reducer, applyMiddleware(thunk))
