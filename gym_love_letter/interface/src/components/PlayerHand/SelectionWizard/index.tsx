import React from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { chooseCard, chooseGuess, chooseTarget, play } from '../../../redux/actions'
import { State } from '../../../redux/reducer'
import { getChosenGameAction, validTargetsExist } from '../../../redux/selectors'
import { Card } from '../../../types'

import styles from './index.module.scss'

const mapProps = (state: State) => ({
  action: getChosenGameAction(state),
  chosenCard: state.chosenCard,
  guess: state.guess,
  target: state.target,
  validTargetsExist: validTargetsExist(state)
})

const mapDispatch = {
  chooseCard,
  chooseGuess,
  chooseTarget,
  play
}

const connector = connect(mapProps, mapDispatch)

type Props = ConnectedProps<typeof connector>

const SelectionWizard: React.FC<Props> = (props) => {
  if (props.chosenCard === null) return null
    const clickCancel = () => {
      props.chooseCard(null)
      props.chooseGuess(null)
      props.chooseTarget(null)
    }

  let playButtonProps = {}
  if (props.action !== undefined) {
    let action = props.action
    playButtonProps = { onClick: () => props.play(action) }
  } else {
    playButtonProps = { disabled: true }
  }

  const handleGuess = (iStr: string) => {
    const i = Number.parseInt(iStr)
    props.chooseGuess(i)
  }

  const guessDropdown = (
    <div className={styles.guessDropdown}>
      <select onChange={(e) => handleGuess(e.target.value)}>
        <option value={0}>Choose guess</option>
        <option value={Card.CardValue.PRIEST}>{Card.CardValue[Card.CardValue.PRIEST]}</option>
        <option value={Card.CardValue.BARON}>{Card.CardValue[Card.CardValue.BARON]}</option>
        <option value={Card.CardValue.HANDMAID}>{Card.CardValue[Card.CardValue.HANDMAID]}</option>
        <option value={Card.CardValue.PRINCE}>{Card.CardValue[Card.CardValue.PRINCE]}</option>
        <option value={Card.CardValue.KING}>{Card.CardValue[Card.CardValue.KING]}</option>
        <option value={Card.CardValue.COUNTESS}>{Card.CardValue[Card.CardValue.COUNTESS]}</option>
        <option value={Card.CardValue.PRINCESS}>{Card.CardValue[Card.CardValue.PRINCESS]}</option>
      </select>
    </div>
  )

  const playButtons = (
    <div className={styles.buttons}>
      <button {...playButtonProps}>Play</button>
      <button onClick={clickCancel}>Cancel</button>
    </div>
  )

  return (
    <div className={styles.SelectionWizard}>
      {props.chosenCard?.hasTarget()
        ? props.target !== null
          ? props.chosenCard.value === Card.CardValue.GUARD
            ? (
              <>
                {guessDropdown}
                {props.guess !== null && playButtons}
              </>
            )
            : playButtons
          : props.validTargetsExist
            ? (
              <div className={styles.message}>
                Select a target
              </div>
            )
            : (
              <>
                <div className={styles.message}>
                  Discard this card with no effect?
                </div>
                {playButtons}
              </>
            )
        : playButtons
      }
    </div>
  )
}

export default connector(SelectionWizard);
