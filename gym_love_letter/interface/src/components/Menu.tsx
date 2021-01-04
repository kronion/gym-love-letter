import React from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { reset } from '../redux/actions'

const connector = connect(null, { reset })

type Props = ConnectedProps<typeof connector>

class Menu extends React.Component<Props> {
  render() {
    return (
      <div>
        <button onClick={this.props.reset}>Click me</button>
      </div>
    )
  }
}

export default connector(Menu);
