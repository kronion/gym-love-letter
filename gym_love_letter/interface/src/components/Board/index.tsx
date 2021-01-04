import React from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { State } from '../../redux/reducer'
import Deck from '../Deck'
import Play from '../Play'

import styles from './index.module.scss'

const mapProps = (state: State) => ({
  plays: state.plays,
  discard: state.discard,
})

const connector = connect(mapProps)

type Props = ConnectedProps<typeof connector>

const Board: React.FC<Props> = (props) => {
  return (
    <div className={styles.Board}>
      <Deck/>
      <div className={styles.plays}>
      {props.plays.map((play, i) => ({play, i})).reverse().map(o => {
          return <Play key={o.i} play={o.play}/>
        })}
      </div>
    </div>
  )
}

export default connector(Board);
