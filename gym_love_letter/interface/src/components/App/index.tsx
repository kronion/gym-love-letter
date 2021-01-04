import React from 'react'
import { connect } from 'react-redux'

import { State } from '../../redux/reducer'
import Game from '../Game'
import Menu from '../Menu'

import styles from './index.module.scss'

type Props = {
  running: boolean
}

const App: React.FC<Props> = props => {
  return (
    <div className={styles.App}>
      {props.running
        ? <Game />
        : <Menu />
      }
    </div>
  )
}

const mapStateToProps = (state: State) => {
  return {
    running: state.running
  }
}

export default connect(mapStateToProps)(App);
