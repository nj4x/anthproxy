import { BrowserRouter, Routes, Route, NavLink, Outlet } from 'react-router-dom';
import useSWR from 'swr';
import { api } from './api/client';
import type { Config } from './api/types';
import Dashboard from './pages/Dashboard';
import Sessions from './pages/Sessions';
import SessionDetail from './pages/SessionDetail';
import Analytics from './pages/Analytics';
import Stats from './pages/Stats';
import Settings from './pages/Settings';
import AuditLog from './pages/AuditLog';

function Layout() {
  const { data: config } = useSWR<Config>('layout-config', () => api.getConfig(), {
    refreshInterval: 60000,
  });

  const navItems = [
    { to: '/', label: 'Dashboard', end: true },
    { to: '/sessions', label: 'Sessions' },
    { to: '/analytics', label: 'Analytics' },
    { to: '/stats', label: 'Stats' },
    { to: '/settings', label: 'Settings' },
    { to: '/audit', label: 'Audit Log' },
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 flex flex-col flex-shrink-0">
        <div className="px-4 py-5 border-b border-gray-700">
          <span className="text-white font-semibold text-sm">anthproxy admin</span>
        </div>
        <nav className="flex-1 px-2 py-4 space-y-1">
          {navItems.map(({ to, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `block px-3 py-2 rounded text-sm transition-colors ${
                  isActive
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-300 hover:text-white hover:bg-gray-800'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between flex-shrink-0">
          <h1 className="text-gray-800 font-semibold text-base">anthproxy admin</h1>
          <div>
            {config === undefined ? (
              <span className="text-gray-400 text-sm">Loading...</span>
            ) : config.routing_enabled ? (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                routing ON
              </span>
            ) : (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                routing OFF
              </span>
            )}
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter basename="/ui">
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="sessions" element={<Sessions />} />
          <Route path="sessions/:sessionId" element={<SessionDetail />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="stats" element={<Stats />} />
          <Route path="settings" element={<Settings />} />
          <Route path="audit" element={<AuditLog />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
