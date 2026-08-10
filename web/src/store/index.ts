import { createStore } from 'vuex'
import auth from './auth'
import ui from './ui'

export default createStore({ modules: { auth, ui } })
