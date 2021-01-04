import { Action } from 'redux'
import { ThunkAction } from 'redux-thunk'

import { buildGameState, Endpoint } from '../api'
import { Card, GameAction, GameState, Player } from '../types'
import { State } from './reducer'
import {getHumanPlayer} from './selectors'

type Timeout = ReturnType<typeof setTimeout>

const MOVE_DELAY = 2500

export enum ActionTypes {
  CHOOSE_CARD = 'CHOOSE_CARD',
  CHOOSE_GUESS = 'CHOOSE_GUESS',
  CHOOSE_TARGET = 'CHOOSE_TARGET',
  REGISTER_TIMEOUT = 'REGISTER_TIMEOUT',
  RESET = 'RESET',
  UPDATE = 'UPDATE',
  WATCH = 'WATCH'
}

export interface ChooseCardAction extends Action {
  type: typeof ActionTypes.CHOOSE_CARD
  card: Card | null
}

export interface ChooseGuessAction extends Action {
  type: typeof ActionTypes.CHOOSE_GUESS
  guess: Card | null
}

export interface ChooseTargetAction extends Action {
  type: typeof ActionTypes.CHOOSE_TARGET
  target: Player | null
}

interface RegisterTimeoutAction extends Action {
  type: typeof ActionTypes.REGISTER_TIMEOUT
  timeout: Timeout
}

interface ResetAction extends Action {
  type: typeof ActionTypes.RESET
  data: GameState
}

export interface UpdateAction extends Action {
  type: typeof ActionTypes.UPDATE
  data: GameState
}
 
export interface WatchAction extends Action {
  type: typeof ActionTypes.WATCH
}

export const chooseCard = (card: Card | null): ChooseCardAction => {
  return {
    type: ActionTypes.CHOOSE_CARD,
    card 
  }
}

export const chooseGuess = (guessId: number | null): ChooseGuessAction => {
  const guess = guessId ? new Card(guessId) : null
  return {
    type: ActionTypes.CHOOSE_GUESS,
    guess
  }
}

export const chooseTarget = (target: Player | null): ChooseTargetAction => {
  return {
    type: ActionTypes.CHOOSE_TARGET,
    target
  }
}

export const play = (action: GameAction): ThunkAction<
  void,
  State,
  unknown,
  (
      ChooseCardAction
    | ChooseGuessAction
    | ChooseTargetAction
    | RegisterTimeoutAction
    | UpdateAction
  )
> => {
  return (dispatch, getState) => {
    dispatch(chooseCard(null))
    dispatch(chooseGuess(null))
    dispatch(chooseTarget(null))
    fetch(`${Endpoint.STEP}/${action.id}`)
      .then(response => response.json())
      .then(data => {
        dispatch(update(buildGameState(data)))
        const state = getState()
        const human = getHumanPlayer(state)
        if (!state.gameOver && ((human.active && state.currentPlayer !== human.position) || state.watching)) {
          const tid = setTimeout(() => dispatch(step()), MOVE_DELAY)
          dispatch(registerTimeout(tid))
        }
      })
  }
}

const registerTimeout = (timeout: Timeout): RegisterTimeoutAction => {
  return {
    type: ActionTypes.REGISTER_TIMEOUT,
    timeout
  }
}

const _reset = (data: GameState): ResetAction => {
  return {
    type: ActionTypes.RESET,
    data
  }
}

export const reset = (): ThunkAction<void, State, unknown, RegisterTimeoutAction | ResetAction> => {
  return (dispatch, getState) => {
    // Clear any queued actions
    const { timeouts } = getState()
    timeouts.map(id => clearTimeout(id))

    fetch(Endpoint.RESET)
      .then(response => response.json())
      .then(data => {
        dispatch(_reset(buildGameState(data)))

        // If the first action isn't being taken by the human player, start stepping forward
        const state = getState()
        if (state.currentPlayer !== getHumanPlayer(state).position) {
          const tid = setTimeout(() => dispatch(step()), MOVE_DELAY)
          dispatch(registerTimeout(tid))
        }
      })
  }
}

const step = (): ThunkAction<void, State, unknown, RegisterTimeoutAction | UpdateAction> => {
  return (dispatch, getState) => {
    fetch(Endpoint.STEP)
      .then(response => response.json())
      .then(data => {
        dispatch(update(buildGameState(data)))
        const state = getState()
        const human = getHumanPlayer(state)
        if (!state.gameOver && ((human.active && state.currentPlayer !== human.position) || state.watching)) {
          const tid = setTimeout(() => dispatch(step()), MOVE_DELAY)
          dispatch(registerTimeout(tid))
        }
      })
  }
}

export const update = (data: GameState): UpdateAction => {
  return {
    type: ActionTypes.UPDATE,
    data
  }
}

const _watch = (): WatchAction => {
  return {
    type: ActionTypes.WATCH
  }
}

export const watch = (): ThunkAction<
  void,
  State,
  unknown,
  WatchAction | RegisterTimeoutAction
> => {
  return (dispatch, getState) => {
    dispatch(_watch())
    const { gameOver } = getState()
    if (!gameOver) {
      const tid = setTimeout(() => dispatch(step()), MOVE_DELAY)
      dispatch(registerTimeout(tid))
    }
  }
}

export type AppAction = (
    ChooseCardAction
  | ChooseGuessAction
  | ChooseTargetAction
  | RegisterTimeoutAction
  | ResetAction
  | UpdateAction
  | WatchAction
)
