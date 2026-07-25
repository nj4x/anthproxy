import useSWR from 'swr';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { SessionsResponse, CostResponse, Config, CostRow } from '../api/types';
import { StatusPanel } from '../components/StatusPanel';
import { parseSessionId } from '../utils';

function formatCost(usd: number): string {
  return '$' + usd.toFixed(2);
}

function SummaryCard({ title, value, sub }: { title: string; value: string; sub?: string }) {
  return (
    <div className="bg-white shadow rounded-lg p-5">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{title}</div>
      <div className="mt-1 text-2xl font-semibold text-gray-800">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
    </div>
  );
}

export default function Dashboard() {
  const { data: sessionsData, error: sessionsError } = useSWR<SessionsResponse>(
    'dashboard-sessions',
    () => api.getSessions(10, 0),
    { refreshInterval: 60000 }
  );

  const { data: config, error: configError } = useSWR<Config>(
    'dashboard-config',
    () => api.getConfig(),
    { refreshInterval: 60000 }
  );

  const { data: costData, error: costError } = useSWR<CostResponse>(
    'dashboard-cost',
    () => api.getCost('model', '7d'),
    { refreshInterval: 60000 }
  );

  const totalCost = costData?.items.reduce((sum: number, row: CostRow) => sum + row.cost_usd, 0) ?? null;

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Dashboard</h2>

      <StatusPanel />

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <SummaryCard
          title="Active Backend"
          value={config?.active_backend ?? (configError ? 'Error' : '—')}
          sub={config ? `mode: ${config.auto_backend_mode}` : undefined}
        />
        <SummaryCard
          title="Routing"
          value={config === undefined ? '—' : config.routing_enabled ? 'ON' : 'OFF'}
        />
        <SummaryCard
          title="Total Sessions"
          value={sessionsData ? String(sessionsData.total) : (sessionsError ? 'Error' : '—')}
        />
        <SummaryCard
          title="Cost This Week"
          value={totalCost !== null ? formatCost(totalCost) : (costError ? 'Error' : '—')}
          sub="7-day total"
        />
      </div>

      {/* Recent sessions table */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-5 py-4 border-b border-gray-200">
          <h3 className="text-sm font-medium text-gray-700">Recent Sessions</h3>
        </div>
        {sessionsError ? (
          <div className="bg-red-50 text-red-700 p-4 rounded">{sessionsError.message}</div>
        ) : sessionsData === undefined ? (
          <div className="p-4 text-gray-500">Loading...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Session</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Last Seen</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wide">Requests</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wide">Cost</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {sessionsData.items.map((s) => (
                  <tr key={s.session_id} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      {(() => {
                        const sid = parseSessionId(s.session_id);
                        return (
                          <div>
                            <Link
                              to={`/sessions/${encodeURIComponent(s.session_id)}`}
                              className="text-blue-600 hover:underline font-mono text-xs"
                              title={sid.tooltip}
                            >
                              {sid.label}
                            </Link>
                            {s.summary && (
                              <p className="mt-0.5 max-w-xs text-xs text-gray-400 truncate" title={s.summary}>
                                {s.summary}
                              </p>
                            )}
                          </div>
                        );
                      })()}
                    </td>
                    <td className="px-4 py-3 text-gray-600 text-xs">
                      {new Date(s.last_seen_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-right text-gray-600">{s.request_count}</td>
                    <td className="px-4 py-3 text-right text-gray-600">{formatCost(s.estimated_cost_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
