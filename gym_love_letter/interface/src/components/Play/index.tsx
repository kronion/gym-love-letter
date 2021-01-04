import React from 'react'

import { Play as PlayType } from '../../types'
import Card from '../Card'

import styles from './index.module.scss'

type Props = {
  play: PlayType
}

const Play: React.FC<Props> = (props) => {
  return (
    <div className={styles.Play}>
      <div>
        <p>Player {props.play.player} plays:</p>
        {props.play.action.target !== null &&
          <p>Target: Player {props.play.action.target}</p>
        }
        {props.play.action.guess !== null &&
          <p>Guess: {props.play.action.guess.name}</p>
        }
        <Card card={props.play.action.card}/>
      </div>
      {props.play.discard !== null &&
        <div className={styles.discard}>
          <div className={styles.arrows}>
            <div className={styles.arrow}></div>
            <div className={styles.arrow}></div>
            <div className={styles.arrow}></div>
          </div>
          <div>
            <p>Player {props.play.discardingPlayer} discards:</p>
            <Card card={props.play.discard}/>
          </div>
        </div>
      }
    </div>
  )
}

export default Play;
