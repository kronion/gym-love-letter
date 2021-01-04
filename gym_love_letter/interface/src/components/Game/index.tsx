import React from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { reset } from '../../redux/actions'
import Board from '../Board'
import GameStatus from '../GameStatus'
import Opponents from '../Opponents'
import PlayerHand from '../PlayerHand'

import styles from './index.module.scss'

const mapDispatch = {
  reset
}

const connector = connect(null, mapDispatch)

type Props = ConnectedProps<typeof connector>

const Game: React.FC<Props> = (props) => {
  return (
    <div className={styles.Game}>
      <button onClick={props.reset}>Reset</button>
      <Opponents/>
      <Board/>
      <PlayerHand/>
      <GameStatus/>
    </div>
  )
}

export default connector(Game);
