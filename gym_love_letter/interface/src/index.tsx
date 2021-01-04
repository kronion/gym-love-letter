import React from 'react'
import ReactDOM from 'react-dom'
import { Provider } from 'react-redux'

import App from './components/App'
import store from './redux/reducer'

const mountNode = document.getElementById("app")
ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  mountNode
)
