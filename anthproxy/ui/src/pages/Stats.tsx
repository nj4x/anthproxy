import { Fragment, useRef, useState } from 'react';
import useSWR from 'swr';
import { api } from '../api/client';
import type { StatsResponse, StatsRow, BackendsResponse } from '../api/types';
import { backendLabel } from '../utils';
import { CostScopeNote } from '../components/CostScopeNote';

const PERIODS = [
  { label: 'Day', value: 'day' },
  { label: 'Week', value: 'week' },
  { label: 'Month', value: 'month' },
  { label: 'Quarter', value: 'quarter' },
];

function formatDuration(secs: number): string {
  const d = Math.floor(secs / 86400);
  const h = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const parts: string[] = [];
  if (d > 0) parts.push(`${d}d`);
  if (h > 0) parts.push(`${h}h`);
  if (parts.length === 0 || m > 0) parts.push(`${m}m`);
  return parts.join(' ') || '0m';
}

function fmtUsd(v: number | null): string {
  if (v == null) return '—';
  return `$${v.toFixed(2)}`;
}

function cacheHitRatio(row: StatsRow): string {
  const denom = row.input_tokens + row.cache_read_tokens;
  if (denom === 0) return '—';
  return (row.cache_read_tokens / denom * 100).toFixed(1) + '%';
}

function DataRow({ row, bold }: { row: StatsRow; bold?: boolean }) {
  const cls = bold
    ? 'font-semibold bg-gray-50 border-t border-gray-200'
    : 'hover:bg-gray-50';
  return (
    <tr className={cls}>
      <td className="px-3 py-2 text-gray-700">{row.backend ? backendLabel(row.backend) : '—'}</td>
      <td className="px-3 py-2 text-gray-700">{row.model_tier || '—'}</td>
      <td className="px-3 py-2 text-right text-gray-700">{row.requests.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-gray-700">{row.input_tokens.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-gray-700">{row.output_tokens.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-gray-700">{row.cache_read_tokens.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-gray-700">{row.cache_creation_tokens.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-gray-700">${row.cost_usd.toFixed(2)}</td>
      <td className="px-3 py-2 text-right text-gray-700">${row.cache_savings_usd.toFixed(2)}</td>
      <td className="px-3 py-2 text-right text-gray-700">{fmtUsd(row.net_savings_usd)}</td>
      <td className="px-3 py-2 text-right text-gray-700">{fmtUsd(row.classifier_overhead_usd)}</td>
    </tr>
  );
}

export default function Stats() {
  const [period, setPeriod] = useState('week');
  const [backend, setBackend] = useState('');
  const [openBuckets, setOpenBuckets] = useState<Set<number>>(new Set());

  const initializedKeyRef = useRef<string | null>(null);
  const currentKey = `${period}:${backend}`;

  const { data: backendsData } = useSWR<BackendsResponse>('backends', () => api.getBackends());

  const { data, error } = useSWR<StatsResponse>(
    ['stats', period, backend],
    () => api.getStats(period, backend || undefined),
    {
      revalidateOnFocus: true,
      onSuccess(d) {
        if (initializedKeyRef.current === currentKey) return;
        initializedKeyRef.current = currentKey;
        const count = d.buckets.length;
        if (count > 7) {
          setOpenBuckets(new Set([0]));
        } else {
          setOpenBuckets(new Set(Array.from({ length: count }, (_, i) => i)));
        }
      },
    }
  );

  const toggleBucket = (i: number) => {
    setOpenBuckets((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  const handlePeriodChange = (p: string) => {
    setPeriod(p);
    initializedKeyRef.current = null;
  };

  const handleBackendChange = (b: string) => {
    setBackend(b);
    initializedKeyRef.current = null;
  };

  const backendNames = backendsData?.backends.map((b) => b.name) ?? [];
  const total = data?.total;
  const isEmpty = data != null && (data.buckets.length === 0 || data.total.requests === 0);

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Stats</h2>

      {/* Control bar */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="flex rounded-md shadow-sm border border-gray-300 overflow-hidden">
          {PERIODS.map(({ label, value }, i) => (
            <button
              key={value}
              onClick={() => handlePeriodChange(value)}
              className={`px-4 py-2 text-sm ${
                period === value
                  ? 'bg-gray-800 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }${i > 0 ? ' border-l border-gray-300' : ''}`}
            >
              {label}
            </button>
          ))}
        </div>

        <select
          value={backend}
          onChange={(e) => handleBackendChange(e.target.value)}
          className="border border-gray-300 rounded-md px-3 py-2 text-sm bg-white text-gray-700"
        >
          <option value="">All Backends</option>
          <option value="subscription">Subscription</option>
          {backendNames.map((name) => (
            <option key={name} value={name}>{backendLabel(name)}</option>
          ))}
        </select>
      </div>

      {error && (
        <div className="bg-red-50 text-red-700 p-4 rounded mb-4 text-sm">{error.message}</div>
      )}

      {!data && !error && (
        <div className="text-gray-500 text-sm">Loading...</div>
      )}

      {data && (
        <>
          {/* Summary metrics */}
          {total && !isEmpty && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
              {([
                { label: 'Requests', value: total.requests.toLocaleString() },
                { label: 'Total Cost', value: `$${total.cost_usd.toFixed(2)}` },
                { label: 'Cache Savings', value: `$${total.cache_savings_usd.toFixed(2)}` },
                { label: 'Cache Hit Ratio', value: cacheHitRatio(total) },
                { label: 'Active Time', value: formatDuration(total.active_time_secs) },
              ] as { label: string; value: string }[]).map(({ label, value }) => (
                <div key={label} className="bg-white shadow rounded-lg p-4">
                  <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
                  <div className="mt-1 text-xl font-semibold text-gray-800">{value}</div>
                </div>
              ))}
              <CostScopeNote className="col-span-2 md:col-span-5" />
            </div>
          )}

          {/* Empty state */}
          {isEmpty && (
            <div className="bg-white shadow rounded-lg p-10 text-center text-gray-500 text-sm">
              No requests recorded for this period and backend filter.
            </div>
          )}

          {/* Bucketed table */}
          {!isEmpty && data.buckets.length > 0 && total && (
            <div className="bg-white shadow rounded-lg overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200 text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    {['Backend', 'Model', 'Requests', 'Input', 'Output', 'Cache Read', 'Cache Created', 'Cost', 'Cache Savings', 'Net Savings', 'Clf Overhead'].map(
                      (col, i) => (
                        <th
                          key={col}
                          className={`px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide ${i >= 2 ? 'text-right' : 'text-left'}`}
                        >
                          {col}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {data.buckets.map((bucket, bi) => (
                    <Fragment key={`bucket-${bi}`}>
                      <tr
                        className="bg-gray-100 cursor-pointer hover:bg-gray-200"
                        onClick={() => toggleBucket(bi)}
                      >
                        <td colSpan={11} className="px-3 py-2 text-xs font-semibold text-gray-600">
                          <span className="mr-2 text-gray-400">{openBuckets.has(bi) ? '▼' : '▶'}</span>
                          {bucket.label}
                        </td>
                      </tr>

                      {openBuckets.has(bi) && bucket.rows.map((row, ri) => (
                        <DataRow key={`${bi}-${ri}`} row={row} />
                      ))}

                      {openBuckets.has(bi) && bucket.rows.length > 1 && (
                        <>
                          <DataRow row={bucket.subtotal} bold />
                          <tr className="bg-gray-50">
                            <td colSpan={11} className="px-3 py-1 text-xs text-gray-400">
                              Active: {formatDuration(bucket.subtotal.active_time_secs)}
                            </td>
                          </tr>
                        </>
                      )}
                    </Fragment>
                  ))}

                  {/* Grand Total */}
                  <tr className="bg-gray-100 border-t-2 border-gray-300">
                    <td colSpan={2} className="px-3 py-2 text-xs font-bold text-gray-700">Grand Total</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">{total.requests.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">{total.input_tokens.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">{total.output_tokens.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">{total.cache_read_tokens.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">{total.cache_creation_tokens.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">${total.cost_usd.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">${total.cache_savings_usd.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">{fmtUsd(total.net_savings_usd)}</td>
                    <td className="px-3 py-2 text-right text-xs font-bold text-gray-700">{fmtUsd(total.classifier_overhead_usd)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
