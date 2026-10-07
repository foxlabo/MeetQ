import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App from './App'
import SessionList from './pages/SessionList'
import NewSession from './pages/NewSession'
import SessionDetail from './pages/SessionDetail'
import SettingsPage from './pages/SettingsPage'
import './styles.css'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <SessionList /> },
      { path: 'new', element: <NewSession /> },
      { path: 'sessions/:id', element: <SessionDetail /> },
      { path: 'settings', element: <SettingsPage /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
