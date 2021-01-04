import React from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { reset, watch } from '../../redux/actions'
import { State } from '../../redux/reducer'
import { getHumanPlayer } from '../../redux/selectors'

import styles from './index.module.scss'

const mapState = (state: State) => ({
  gameOver: state.gameOver,
  playerOut: !getHumanPlayer(state).active,
  watching: state.watching
})

const mapDispatch = {
  reset,
  watch
}

const connector = connect(mapState, mapDispatch)

type Props = ConnectedProps<typeof connector>

const GameStatus: React.FC<Props> = (props) => {
  const display = (props.gameOver || props.playerOut) && !props.watching

  if (display) {
    return (
      <>
        <div className={styles.Overlay}/>
        <div className={styles.GameStatus} onClick={props.watch}>
          <div className={styles.gameOverMenu} onClick={(e) => {
            e.stopPropagation()
          }}>
            <h1 className={styles.announcement}>{props.playerOut ? 'DEFEAT' : 'VICTORY'}</h1>
            <div>
              <button onClick={props.watch}>{props.gameOver ? 'Review' : 'Continue watching'} Game</button>
              <button onClick={props.reset}>New game</button>
            </div>
          </div>
        </div>
      </>
    )
  }
  else {
    return null
  }
}

export default connector(GameStatus)
