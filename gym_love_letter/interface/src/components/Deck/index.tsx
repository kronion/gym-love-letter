import React from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { State } from '../../redux/reducer'
import Card from '../Card'

import styles from './index.module.scss'

const mapProps = (state: State) => ({
  cardsRemaining: state.cardsRemaining
})

const connector = connect(mapProps)

type Props = ConnectedProps<typeof connector>

const Deck: React.FC<Props> = (props) => {
  return (
    <div className={styles.Deck}>
      <p>{props.cardsRemaining} cards left</p>
      <Card empty={props.cardsRemaining <= 0} faceUp={false}/>
    </div>
  )
}

export default connector(Deck)
