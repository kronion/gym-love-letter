import classnames from 'classnames'
import React from 'react'

import { Card as CardType } from '../../types'

import styles from './index.module.scss'


type Props = {
  card?: CardType | null
  disabled?: boolean
  empty?: boolean
  faceUp?: boolean
  onClick?: () => void
  selectable?: boolean
  selected?: boolean
}

const Card: React.FC<Props> = (props) => {
  // Basic validation of some mutually exclusive states
  const faceUp = props.faceUp && !props.empty
  const disabled = props.disabled && faceUp
  const selectable = props.selectable && faceUp
  const selected = props.selected && faceUp

  const classes = classnames([
    `${styles.Card}`,
    {[`${styles.disabled}`]: disabled},
    {[`${styles.empty}`]: props.empty},
    {[`${styles.faceUp}`]: faceUp},
    {[`${styles.selectable}`]: selectable},
    {[`${styles.selected}`]: selected},
  ])

  return (
    <div className={classes} onClick={props.onClick}>
      {!props.empty && (
        faceUp && props.card
          ? (
            <div className={styles.properties}>
              <span>{props.card.name}</span>
              <span>{props.card.value}</span>
            </div>
          )
          : <div className={styles.backDesign}>?</div>
      )}
    </div>
  )
}

Card.defaultProps = {
  card: null,
  disabled: false,
  empty: false,
  faceUp: true,
  selectable: false,
  selected: false
};

export default Card;
