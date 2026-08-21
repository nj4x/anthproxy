import { useState } from 'react';
import useSWR from 'swr';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../api/client';
import type { CostResponse, RoutingResponse, CostRow } from '../api/types';
import { backendLabel } from '../utils';
import { CostScopeNote } from '../components/CostScopeNote';

const TIME_RANGES = ['1d', '7d', '30d'] as const;
type TimeRange = typeof TIME_RANGES[number];

const GROUP_BY_OPTIONS = ['model', 'tier', 'backend'] as const;
type GroupBy = typeof GROUP_BY_OPTIONS[number];

const BAR_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#14b8a6', '#f97316',
];

export default function Analytics() {
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');
  const [groupBy, setGroupBy] = useState<GroupBy>('model');

  const { data: costData, error: costError } = useSWR<CostResponse>(
    ['cost', groupBy, timeRange],
    () => api.getCost(groupBy, timeRange)
  );

  const { data: routingData, error: routingError } = useSWR<RoutingResponse>(
    ['routing', timeRange],
    () => api.getRouting(timeRange)
  );

  const displayName = (name: string) =>
    groupBy === 'backend' ? backendLabel(name) : name;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4 flex-wrap">
        <h2 className="text-lg font-semibold text-gray-800">Analytics</h2>

        {/* Time range tabs */}
        <div className="flex rounded overflow-hidden border border-gray-300">
          {TIME_RANGES.map((tr) => (
            <button
              key={tr}
              onClick={() => setTimeRange(tr)}
              className={`px-3 py-1 text-sm ${
                timeRange === tr
                  ? 'bg-gray-800 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              {tr}
            </button>
          ))}
        </div>

        {/* Group by */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Group by:</span>
          <select
            value={groupBy}
            onChange={(e) => setGroupBy(e.target.value as GroupBy)}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            {GROUP_BY_OPTIONS.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Cost section */}
      {costError ? (
        <div className="bg-red-50 text-red-700 p-4 rounded">{costError.message}</div>
      ) : !costData ? (
        <div className="text-gray-500">Loading...</div>
      ) : (
        <div className="space-y-4">
          {/* Chart */}
          {costData.items.length > 0 && (
            <div className="bg-white shadow rounded-lg p-5">
              <h3 className="text-sm font-medium text-gray-700 mb-4">Cost by {groupBy}</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={costData.items.map((row) => ({ ...row, name: displayName(row.name) }))} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => `$${v.toFixed(3)}`} />
                  <Tooltip formatter={(v: any) => typeof v === 'number' ? `$${v.toFixed(5)}` : String(v)} />
                  <Bar dataKey="cost_usd" radius={[2, 2, 0, 0]}>
                    {costData.items.map((_: CostRow, index: number) => (
                      <Cell key={index} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Cost table */}
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="px-5 py-3 border-b border-gray-200 text-sm font-medium text-gray-700">
              Cost Table
              <CostScopeNote className="mt-1 font-normal" />
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Requests</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Input Tokens</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Output Tokens</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Cost</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-100">
                  {costData.items.map((row: CostRow) => (
                    <tr key={row.name} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-gray-800 font-medium">{displayName(row.name)}</td>
                      <td className="px-4 py-3 text-right text-gray-600">{row.requests.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right text-gray-600">{row.input_tokens.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right text-gray-600">{row.output_tokens.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right text-gray-800 font-medium">${row.cost_usd.toFixed(5)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Routing Economics */}
          {(() => {
            const routedCost = costData.items.reduce((a, r) => a + r.cost_usd, 0);
            const netSavings = costData.items.reduce((a, r) => a + (r.net_savings_usd ?? 0), 0);
            const clfOverhead = costData.items.reduce((a, r) => a + (r.classifier_overhead_usd ?? 0), 0);
            const opusBaseline = routedCost + netSavings + clfOverhead;
            const hasEconomics = costData.items.some(
              (r) => r.net_savings_usd != null || r.classifier_overhead_usd != null
            );
            if (!hasEconomics) return null;
            const cards = [
              { label: 'Opus Baseline', value: opusBaseline, tone: 'text-gray-800' },
              { label: 'Routed Cost', value: routedCost, tone: 'text-gray-800' },
              { label: 'Classifier Overhead', value: clfOverhead, tone: 'text-amber-600' },
              {
                label: 'Net Savings',
                value: netSavings,
                tone: netSavings >= 0 ? 'text-green-600' : 'text-red-600',
              },
            ];
            return (
              <div className="bg-white shadow rounded-lg p-5">
                <h3 className="text-sm font-medium text-gray-700 mb-4">Routing Economics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {cards.map(({ label, value, tone }) => (
                    <div key={label} className="rounded-lg border border-gray-100 p-4">
                      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
                      <div className={`mt-1 text-xl font-semibold ${tone}`}>${value.toFixed(5)}</div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Routing KPIs */}
      {routingData && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {[
            { label: 'Upgrades', value: routingData.upgrade_count },
            { label: 'Downgrades', value: routingData.downgrade_count },
            { label: 'Unchanged', value: routingData.unchanged_count },
            { label: 'Size Forced', value: routingData.size_forced_count },
            { label: 'Affirmation', value: routingData.affirmation_count },
            { label: 'Cached Tier', value: routingData.cached_tier_count },
          ].map(({ label, value }) => (
            <div key={label} className="bg-white shadow rounded-lg p-4 text-center">
              <div className="text-2xl font-semibold text-gray-800">{value.toLocaleString()}</div>
              <div className="text-xs text-gray-500 mt-1">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Routing section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Reason codes */}
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-200 text-sm font-medium text-gray-700">
            Routing Reason Codes
          </div>
          {routingError ? (
            <div className="bg-red-50 text-red-700 p-4 rounded">{routingError.message}</div>
          ) : !routingData ? (
            <div className="p-4 text-gray-500">Loading...</div>
          ) : (
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Reason Code</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Count</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {routingData.reason_codes.map(({ reason_code, count }) => (
                  <tr key={reason_code} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-mono text-gray-800 text-xs">{reason_code}</td>
                    <td className="px-4 py-3 text-right text-gray-600">{count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Tier transitions */}
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-200 text-sm font-medium text-gray-700">
            Tier Transitions
          </div>
          {routingError ? (
            <div className="bg-red-50 text-red-700 p-4 rounded">{routingError.message}</div>
          ) : !routingData ? (
            <div className="p-4 text-gray-500">Loading...</div>
          ) : (
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Requested</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Routed</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Count</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {routingData.tier_transitions.map(({ requested_tier, routed_tier, count }, i) => (
                  <tr key={i} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-gray-800 text-xs">{requested_tier}</td>
                    <td className="px-4 py-3 text-gray-800 text-xs">{routed_tier}</td>
                    <td className="px-4 py-3 text-right text-gray-600">{count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Model distributions */}
      {routingData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="px-5 py-3 border-b border-gray-200 text-sm font-medium text-gray-700">
              Original Model Distribution
            </div>
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Count</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {routingData.original_model_distribution.map(({ model, count }) => (
                  <tr key={model} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-mono text-gray-800 text-xs">{model}</td>
                    <td className="px-4 py-3 text-right text-gray-600">{count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="px-5 py-3 border-b border-gray-200 text-sm font-medium text-gray-700">
              Routed Model Distribution
            </div>
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Count</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {routingData.routed_model_distribution.map(({ model, count }) => (
                  <tr key={model} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-mono text-gray-800 text-xs">{model}</td>
                    <td className="px-4 py-3 text-right text-gray-600">{count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
