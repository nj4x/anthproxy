import useSWR from 'swr';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { StatusResponse, UsageWindow, CreditsWindow, EnterpriseToken } from '../api/types';
import { backendLabel } from '../utils';

function formatAge(ageSecs: number | null | undefined): string | null {
  if (ageSecs == null) return null;
  if (ageSecs < 60) return 'just now';
  const m = Math.floor(ageSecs / 60);
  return `${m}m ago`;
}

function formatActiveDuration(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h >= 24) return `${Math.floor(h / 24)}d ${h % 24}h`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function resetSecs(w: UsageWindow): number | null {
  let secs = w.reset_in_secs;
  if (secs == null && w.reset_at != null) {
    secs = Math.floor((new Date(w.reset_at).getTime() - Date.now()) / 1000);
  }
  return secs == null || secs < 0 ? null : secs;
}

function formatResetDateTime(resetAt: string, includeMinutes: boolean): string {
  const date = new Date(resetAt);
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();

  const hour = date.getHours();
  const minute = date.getMinutes();
  const isPm = hour >= 12;
  const displayHour = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour;
  const meridiem = isPm ? 'p.m.' : 'a.m.';

  let timeStr: string;
  if (includeMinutes) {
    const paddedMinute = String(minute).padStart(2, '0');
    timeStr = `${displayHour}:${paddedMinute} ${meridiem}`;
  } else {
    timeStr = `${displayHour} ${meridiem}`;
  }

  if (sameDay) return timeStr;
  const weekday = date.toLocaleDateString('en-US', { weekday: 'short' });
  return `${weekday}, ${timeStr}`;
}

function formatCountdown(w: UsageWindow, format: 'hm' | 'dhm'): string {
  const secs = resetSecs(w);
  if (secs == null) return 'soon';

  let countdownStr = '';
  if (format === 'dhm') {
    const d = Math.floor(secs / 86400);
    const h = Math.floor((secs % 86400) / 3600);
    if (d > 0) countdownStr = `reset in ${d}d ${h}h`;
    else {
      const m = Math.floor((secs % 3600) / 60);
      if (h > 0) countdownStr = `reset in ${h}h ${m}m`;
      else countdownStr = `reset in ${m}m`;
    }
  } else {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    if (h > 0) countdownStr = `reset in ${h}h ${m}m`;
    else countdownStr = `reset in ${m}m`;
  }

  if (w.reset_at != null) {
    const exactTime = formatResetDateTime(w.reset_at, format === 'hm');
    return `${countdownStr} (${exactTime})`;
  }
  return countdownStr;
}

function UsageSection({ title, window: w, format, windowHours }: {
  title: string;
  window: UsageWindow;
  format: 'hm' | 'dhm';
  windowHours?: number;
}) {
  const pct = Math.min(w.pct ?? 0, 100);
  const barColor = 'bg-blue-500';

  // Red "head": portion of the bar that's ahead of the linear burn pace.
  // redStart = 100 - Y*(100/windowHours), where Y = hours until reset.
  // The red segment covers [redStart, pct] when pct > redStart.
  let redStart: number | null = null;
  if (windowHours != null) {
    const secs = resetSecs(w);
    if (secs != null) {
      redStart = Math.max(0, 100 - (secs / 3600) * (100 / windowHours));
    }
  }
  const hasRedHead = redStart != null && pct > redStart;
  // 0.5% floor avoids a sub-pixel green sliver at the exact-pace boundary
  const hasGreenHead = redStart != null && redStart - pct > 0.5;

  return (
    <div>
      <div className="text-xs text-gray-500 mb-1">{title}</div>
      <div className="text-sm text-gray-700">
        {(() => {
          const active = w.active_secs != null && w.active_secs > 0 ? ` over ${formatActiveDuration(w.active_secs)} of LLM usage` : '';
          if (w.used_tokens != null)
            return `${w.used_tokens.toLocaleString()} / ${w.limit_tokens?.toLocaleString() ?? '—'}${active}`;
          if (w.pct != null)
            return `${w.pct.toFixed(0)}% used${active}`;
          return '—';
        })()}
      </div>
      {(w.limit_tokens != null || w.pct != null) && (
        <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden relative">
          <div className={`absolute inset-y-0 left-0 ${barColor} transition-all`} style={{ width: `${pct}%` }} />
          {hasRedHead && (
            <div
              className="absolute inset-y-0 bg-red-500"
              style={{ left: `${redStart}%`, width: `${pct - redStart!}%` }}
            />
          )}
          {hasGreenHead && (
            <div
              className="absolute inset-y-0 bg-green-500 opacity-40"
              style={{ left: `${pct}%`, width: `${redStart! - pct}%` }}
            />
          )}
        </div>
      )}
      <div className="text-xs text-gray-400 mt-1">
        {w.reset_at != null || w.reset_in_secs != null ? formatCountdown(w, format) : '—'}
      </div>
    </div>
  );
}

function CreditsSection({ credits: c }: { credits: CreditsWindow }) {
  const pct = c.pct ?? 0;
  const barColor =
    c.pct == null ? 'bg-blue-500' :
    c.pct >= 100 ? 'bg-red-500' :
    c.pct >= 80 ? 'bg-amber-500' :
    'bg-blue-500';

  return (
    <div>
      <div className="text-xs text-gray-500 mb-1">Credits</div>
      <div className="text-sm text-gray-700">
        {c.pct != null ? `${c.pct.toFixed(0)}% used` : '—'}
      </div>
      {c.used_usd != null && c.total_usd != null && (
        <div className="text-xs text-gray-500 mt-0.5">
          ${c.used_usd.toFixed(2)} of ${c.total_usd.toFixed(2)}
        </div>
      )}
      {c.total_usd != null && (
        <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full ${barColor} rounded-full transition-all`}
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

function EnterpriseTokenCard({ token }: { token: EnterpriseToken }) {
  const ageLabel = formatAge(token.usage_age_seconds);
  const statusLabel =
    token.monthly_blocked ? 'Spend cap reached' :
    token.cooldown_remaining_seconds > 0 ? `Cooling down ${Math.ceil(token.cooldown_remaining_seconds)}s` :
    token.eligible ? 'Eligible' :
    'Not eligible';
  const statusColor =
    token.monthly_blocked ? 'bg-red-100 text-red-800' :
    token.cooldown_remaining_seconds > 0 ? 'bg-amber-100 text-amber-800' :
    token.eligible ? 'bg-green-100 text-green-800' :
    'bg-gray-100 text-gray-600';
  const burn = token.burn_pct;

  return (
    <div className="bg-gray-50 rounded-lg p-4 min-w-48 space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-700">Anthropic-OAuth token</span>
        {ageLabel != null && (
          <span className="text-xs text-gray-400">updated {ageLabel}{token.usage_stale ? ' (stale)' : ''}</span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${statusColor}`}>
          {statusLabel}
        </span>
      </div>
      <div>
        <div className="text-xs text-gray-500 mb-1">Monthly quota</div>
        <div className="text-sm text-gray-700">
          {burn != null ? `${burn.toFixed(0)}% used` : '—'}
        </div>
        {token.used_usd != null && token.total_usd != null && (
          <div className="text-xs text-gray-500 mt-0.5">
            ${token.used_usd.toFixed(2)} of ${token.total_usd.toFixed(2)}
          </div>
        )}
        {burn != null && (() => {
          const pct = Math.min(burn, 100);
          const barColor = burn >= 100 ? 'bg-red-500' : burn >= 80 ? 'bg-amber-500' : 'bg-blue-500';
          // Pace reference for the calendar-month quota: fraction of the UTC
          // month already elapsed, computed server-side (matches the self-pace
          // gate the server actually acts on, ADR-0016).
          const monthElapsedPct = token.month_elapsed_pct;
          const hasRedHead = pct > monthElapsedPct;
          const hasGreenHead = monthElapsedPct - pct > 0.5;
          return (
            <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden relative">
              <div className={`absolute inset-y-0 left-0 ${barColor} transition-all`} style={{ width: `${pct}%` }} />
              {hasRedHead && (
                <div
                  className="absolute inset-y-0 bg-red-500"
                  style={{ left: `${monthElapsedPct}%`, width: `${pct - monthElapsedPct}%` }}
                />
              )}
              {hasGreenHead && (
                <div
                  className="absolute inset-y-0 bg-green-500 opacity-40"
                  style={{ left: `${pct}%`, width: `${monthElapsedPct - pct}%` }}
                />
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}

export function StatusPanel() {
  const { data, error } = useSWR<StatusResponse>(
    'status',
    () => api.getStatus(),
    { refreshInterval: 60000 }
  );

  if (error) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded mb-6 text-sm">{error.message}</div>
    );
  }
  if (!data) {
    return <div className="p-4 text-gray-500 text-sm mb-6">Loading status...</div>;
  }

  return (
    <div className="bg-white shadow rounded-lg mb-6 divide-y divide-gray-100">
      {/* Active backend */}
      <div className="px-5 py-3 flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Active Backend</span>
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-sm font-medium bg-indigo-100 text-indigo-800">
          {backendLabel(data.active_backend)}
        </span>
        <div className="flex flex-wrap gap-2">
          {data.backends.filter(b => b.name !== data.active_backend).map((b) => {
            const cls =
              b.available === true ? 'bg-green-100 text-green-800' :
              b.available === false ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-600';
            return (
              <span key={b.name} className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
                {b.available === null ? `${backendLabel(b.name)} (Unknown)` : backendLabel(b.name)}
              </span>
            );
          })}
        </div>
        {data.auto_selection != null && (
          <span className="text-xs text-gray-400 ml-1">auto-selection: {data.auto_selection}</span>
        )}
      </div>

      {/* Model routing */}
      <div className="px-5 py-3 flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Model Routing</span>
        {data.routing_enabled ? (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            Routing ON
          </span>
        ) : (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
            Routing OFF
          </span>
        )}
        {data.routing_enabled && (
          <span className="text-xs text-gray-500">{data.routing_mode}</span>
        )}
        {data.routing_enabled && data.routing_mode === 'classifier' && (
          <span className="text-xs text-gray-400 font-mono">{data.classifier_model}</span>
        )}
      </div>

      {/* Session overrides */}
      {data.session_overrides.length > 0 && (
        <div className="px-5 py-3">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Session Overrides</div>
          <table className="text-xs w-full">
            <thead>
              <tr className="text-gray-400">
                <th className="text-left pb-1 font-medium">Session</th>
                <th className="text-left pb-1 font-medium">Pinned Backend</th>
                <th className="text-left pb-1 font-medium">Pinned Tier</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {data.session_overrides.map((o) => {
                const label = o.display_name ?? (
                  o.session_id.length > 20
                    ? o.session_id.slice(0, 20) + '…'
                    : o.session_id
                );
                return (
                  <tr key={o.session_id}>
                    <td className="py-1">
                      <Link
                        to={`/sessions/${encodeURIComponent(o.session_id)}`}
                        className="text-blue-600 hover:underline font-mono"
                        title={o.session_id}
                      >
                        {label}
                      </Link>
                    </td>
                    <td className="py-1 text-gray-600">{o.pinned_backend ? backendLabel(o.pinned_backend) : '—'}</td>
                    <td className="py-1 text-gray-600">{o.pinned_tier ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Subscription usage */}
      {(Object.keys(data.subscription_usage).length > 0 || data.enterprise_token?.present) && (
        <div className="px-5 py-4">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">Subscription Usage</div>
          <div className="flex flex-wrap gap-4">
            {data.enterprise_token?.present && (
              <EnterpriseTokenCard token={data.enterprise_token} />
            )}
            {Object.entries(data.subscription_usage).map(([name, usage]) => {
              const ageLabel = formatAge(usage.age_secs);
              return (
                <div key={name} className="bg-gray-50 rounded-lg p-4 min-w-48 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-700 capitalize">{name}</span>
                    {ageLabel != null && (
                      <span className="text-xs text-gray-400">updated {ageLabel}</span>
                    )}
                  </div>
                  {usage.five_hour && <UsageSection title={`${usage.five_hour.window_hours ?? 5}-hour window`} window={usage.five_hour} format="hm" windowHours={usage.five_hour.window_hours ?? 5} />}
                  {usage.weekly && <UsageSection title={name === 'codex' ? `${usage.weekly.window_hours ?? 168}-hour window` : 'Weekly'} window={usage.weekly} format="dhm" windowHours={usage.weekly.window_hours ?? 168} />}
                  {usage.credits && <CreditsSection credits={usage.credits} />}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
