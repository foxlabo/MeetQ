import { NavLink, Outlet } from 'react-router-dom'

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">
          MeetQ
        </NavLink>
        <nav>
          <NavLink to="/" end data-testid="nav-sessions">
            セッション
          </NavLink>
          <NavLink to="/new" data-testid="nav-new">
            新規
          </NavLink>
          <NavLink to="/settings" data-testid="nav-settings">
            設定
          </NavLink>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  )
}
