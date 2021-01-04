import { createSelector } from 'reselect'

import { Card } from '../types'
import { State } from './reducer'

const actions = (state: State) => state.validActions
const cardId = (state: State) => state.chosenCard ? state.chosenCard.value : null
const chosenCard = (state: State) => state.chosenCard
const guessId = (state: State) => state.guess ? state.guess.value : null
const players = (state: State) => state.players
const targetPosition = (state: State) => state.target ? state.target.position : null

export const getChosenGameAction = createSelector(
  [ actions, cardId, guessId, targetPosition ],
  (actions, cardId, guessId, targetPos) => {
    return actions.find(action => action.card.value === cardId && (action.guess ? action.guess.value : action.guess) === guessId && action.target === targetPos)
  }
)

export const getHumanPlayer = createSelector(
  [ players ],
  (players) => {
    return players[0]
  }
)

export const validTargetsExist = createSelector(
  [ chosenCard, players ],
  (chosenCard, players) => {
    if (!chosenCard?.hasTarget()) {
      return false
    }

    let validPlayers = players
    if (chosenCard.value !== Card.CardValue.PRINCE) {
      validPlayers = players.filter(p => p.position !== 0)
    }

    return validPlayers.some(p => p.active && !p.safe)
  }
)

