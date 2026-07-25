import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import useSWR from 'swr';
import { api } from '../api/client';
import type { SessionDetail as SessionDetailType, TraceResponse, RequestRecord, BackendsResponse } from '../api/types';
import { RequestDetailDrawer } from '../components/RequestDetailDrawer';
import { ExcerptHighlight } from '../components/ExcerptHighlight';
import { parseSessionId, parseConversationAnchor } from '../utils';

const TIERS = ['haiku', 'sonnet', 'opus', 'fable'];
const TRACE_PAGE_SIZE = 100;

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === 'success'
      ? 'bg-green-100 text-green-800'
      : status === 'rate_limited'
      ? 'bg-yellow-100 text-yellow-800'
      : 'bg-red-100 text-red-800';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {status}
    </span>
  );
}

function ClassBadge({ cls }: { cls: string | null }) {
  if (!cls) return <span className="text-gray-400">—</span>;
  const colorCls =
    cls === 'trivial'
      ? 'bg-blue-100 text-blue-800'
      : cls === 'standard'
      ? 'bg-green-100 text-green-800'
      : cls === 'deep'
      ? 'bg-purple-100 text-purple-800'
      : 'bg-gray-100 text-gray-800';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${colorCls}`}>
      {cls}
    </span>
  );
}

export default function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const id = sessionId ?? '';

  const [searchParams, setSearchParams] = useSearchParams();
  const urlQ = searchParams.get('q') ?? '';
  const [filterInput, setFilterInput] = useState(urlQ);
  const filterDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { data: session, error: sessionError, mutate: mutateSession } = useSWR<SessionDetailType>(
    ['session', id],
    () => api.getSession(id),
    { refreshInterval: 60000 }
  );

  const { data: backendsData } = useSWR<BackendsResponse>('backends', () => api.getBackends());

  const [selectedAnchor, setSelectedAnchor] = useState<string | null>(null);
  const [selectedRequestId, setSelectedRequestId] = useState<number | null>(null);
  const [pendingBackend, setPendingBackend] = useState<string>('');
  const [pendingTier, setPendingTier] = useState<string>('');
  const [saving, setSaving] = useState(false);
  const [tracePage, setTracePage] = useState(0);

  const traceQ = urlQ.trim() || undefined;

  const { data: traceData, error: traceError } = useSWR<TraceResponse>(
    ['trace', id, selectedAnchor, tracePage, traceQ],
    () => api.getTrace(id, selectedAnchor ?? undefined, TRACE_PAGE_SIZE, tracePage * TRACE_PAGE_SIZE, traceQ),
    { refreshInterval: 60000 }
  );

  useEffect(() => {
    setTracePage(0);
  }, [selectedAnchor]);

  useEffect(() => {
    setTracePage(0);
  }, [urlQ]);

  const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setFilterInput(val);
    if (filterDebounceRef.current) clearTimeout(filterDebounceRef.current);
    filterDebounceRef.current = setTimeout(() => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (val.trim()) next.set('q', val.trim());
          else next.delete('q');
          return next;
        },
        { replace: true }
      );
      setTracePage(0);
    }, 300);
  };

  const handleFilterKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      if (filterDebounceRef.current) clearTimeout(filterDebounceRef.current);
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (filterInput.trim()) next.set('q', filterInput.trim());
          else next.delete('q');
          return next;
        },
        { replace: true }
      );
      setTracePage(0);
    }
  };

  const anchors = session?.conversations
    ? session.conversations.map((c) => c.conversation_anchor).filter((a): a is string => a !== null)
    : [];

  const filteredTrace = traceData?.items ?? [];

  const backendNames = backendsData?.backends.map((b) => b.name) ?? [];

  // Map conversation anchors to their generated summaries for readable filters.
  const summaryByAnchor = new Map<string, string>();
  const subAgentAnchors = new Set<string>();
  session?.conversations?.forEach((c) => {
    if (c.conversation_anchor && c.summary) {
      summaryByAnchor.set(c.conversation_anchor, c.summary);
    }
    if (c.conversation_anchor && c.parent_conversation_anchor !== null) {
      subAgentAnchors.add(c.conversation_anchor);
    }
  });

  const handleSaveBackend = async () => {
    setSaving(true);
    try {
      await api.setSessionBackend(id, pendingBackend === '' || pendingBackend === 'auto' ? null : pendingBackend);
      await mutateSession();
    } finally {
      setSaving(false);
    }
  };

  const handleSaveTier = async () => {
    setSaving(true);
    try {
      await api.setSessionTier(id, pendingTier === '' || pendingTier === 'auto' ? null : pendingTier);
      await mutateSession();
    } finally {
      setSaving(false);
    }
  };

  if (sessionError) {
    return <div className="bg-red-50 text-red-700 p-4 rounded">{sessionError.message}</div>;
  }
  if (!session) {
    return <div className="text-gray-500">Loading...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Session header */}
      <div className="bg-white shadow rounded-lg p-5">
        {(() => {
          const sid = parseSessionId(id);
          return (
            <h2 className="text-sm font-semibold text-gray-800 font-mono mb-2" title={sid.tooltip}>
              {sid.label}
            </h2>
          );
        })()}
        {session.summary && (
          <p className="text-sm text-gray-600 mb-3">{session.summary}</p>
        )}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-gray-600">
          <div><span className="text-gray-400">Created:</span> {new Date(session.created_at).toLocaleString()}</div>
          <div><span className="text-gray-400">Last seen:</span> {new Date(session.last_seen_at).toLocaleString()}</div>
          <div><span className="text-gray-400">Requests:</span> {session.request_count}</div>
          <div><span className="text-gray-400">Cost:</span> ${session.estimated_cost_usd.toFixed(4)}</div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white shadow rounded-lg p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">Pinned Backend</label>
          <div className="flex gap-2">
            <select
              className="flex-1 border border-gray-300 rounded px-2 py-1.5 text-sm"
              defaultValue={session.pinned_backend ?? ''}
              onChange={(e) => setPendingBackend(e.target.value)}
            >
              <option value="">Auto</option>
              {backendNames.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <button
              onClick={handleSaveBackend}
              disabled={saving}
              className="px-3 py-1.5 bg-gray-800 text-white rounded text-sm hover:bg-gray-700 disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">Pinned Tier</label>
          <div className="flex gap-2">
            <select
              className="flex-1 border border-gray-300 rounded px-2 py-1.5 text-sm"
              defaultValue={session.pinned_tier ?? ''}
              onChange={(e) => setPendingTier(e.target.value)}
            >
              <option value="">Auto</option>
              {TIERS.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <button
              onClick={handleSaveTier}
              disabled={saving}
              className="px-3 py-1.5 bg-gray-800 text-white rounded text-sm hover:bg-gray-700 disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </div>
      </div>

      {/* Anchor filter */}
      {anchors.length > 1 && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setSelectedAnchor(null)}
            className={`px-3 py-1 rounded text-xs border ${
              selectedAnchor === null
                ? 'bg-gray-800 text-white border-gray-800'
                : 'border-gray-300 text-gray-700 hover:bg-gray-100'
            }`}
          >
            All
          </button>
          {anchors.map((anchor) => {
            const anchorParsed = parseConversationAnchor(anchor);
            const summary = summaryByAnchor.get(anchor);
            const label = summary
              ? summary.length > 48 ? summary.slice(0, 48) + '…' : summary
              : anchorParsed.label;
            const tooltip = summary
              ? `${summary}\n\n${anchorParsed.tooltip}`
              : anchorParsed.tooltip;
            return (
              <button
                key={anchor}
                onClick={() => setSelectedAnchor(anchor)}
                className={`px-3 py-1 rounded text-xs border max-w-md ${
                  summary ? '' : 'font-mono'
                } ${
                  selectedAnchor === anchor
                    ? 'bg-gray-800 text-white border-gray-800'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
                title={tooltip}
              >
                <span className="flex items-center gap-1 min-w-0">
                  <span className="truncate">{label}</span>
                  {subAgentAnchors.has(anchor) && (
                    <span className="shrink-0 inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-700">
                      sub-agent
                    </span>
                  )}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Text filter */}
      <div>
        <input
          type="text"
          aria-label="Filter requests"
          placeholder="Filter requests by prompt or response…"
          value={filterInput}
          onChange={handleFilterChange}
          onKeyDown={handleFilterKeyDown}
          className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {/* Trace table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-200 text-sm font-medium text-gray-700">
          Request Trace {traceData ? `(${traceData.total})` : ''}
        </div>
        {traceError ? (
          <div className="bg-red-50 text-red-700 p-4 rounded">{traceError.message}</div>
        ) : !traceData ? (
          <div className="p-4 text-gray-500">Loading...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Requested Model</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Routed</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Class</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Reason</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">In</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Out</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Cost</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">ms</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Backend</th>
                  <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">Applied</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {filteredTrace.map((r: RequestRecord) => (
                  <tr
                    key={r.id}
                    className="hover:bg-gray-50 cursor-pointer"
                    onClick={() => setSelectedRequestId(r.id)}
                  >
                    <td className="px-3 py-2 whitespace-nowrap text-gray-600">
                      {new Date(r.request_ts).toLocaleTimeString()}
                    </td>
                    <td className="px-3 py-2 font-mono text-gray-800 max-w-xs" title={r.requested_model}>
                      <div className="truncate">{r.requested_model}</div>
                      {r.excerpt && <ExcerptHighlight excerpt={r.excerpt} />}
                    </td>
                    <td className="px-3 py-2 font-mono text-gray-600 max-w-xs truncate" title={r.routed_model ?? undefined}>
                      {r.routed_model ?? '—'}
                    </td>
                    <td className="px-3 py-2"><ClassBadge cls={r.classification} /></td>
                    <td className="px-3 py-2 text-gray-600 max-w-xs truncate" title={r.reason_code ?? undefined}>
                      {r.reason_code ?? '—'}
                    </td>
                    <td className="px-3 py-2"><StatusBadge status={r.status} /></td>
                    <td className="px-3 py-2 text-right text-gray-600">
                      {r.input_tokens != null ? r.input_tokens.toLocaleString() : '—'}
                    </td>
                    <td className="px-3 py-2 text-right text-gray-600">
                      {r.output_tokens != null ? r.output_tokens.toLocaleString() : '—'}
                    </td>
                    <td className="px-3 py-2 text-right text-gray-600">
                      {r.cost_estimate != null ? `$${r.cost_estimate.toFixed(4)}` : '—'}
                    </td>
                    <td className="px-3 py-2 text-right text-gray-600">
                      {r.duration_ms != null ? r.duration_ms.toLocaleString() : '—'}
                    </td>
                    <td className="px-3 py-2 text-gray-600">{r.backend}</td>
                    <td className="px-3 py-2 text-center text-gray-600">
                      {r.applied === 1 ? '✓' : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {traceData && traceData.total > TRACE_PAGE_SIZE && (
          <div className="flex items-center justify-between px-5 py-3 border-t border-gray-200">
            <span className="text-xs text-gray-500">
              {tracePage * TRACE_PAGE_SIZE + 1}–{Math.min((tracePage + 1) * TRACE_PAGE_SIZE, traceData.total)} of {traceData.total}
            </span>
            <div className="flex gap-2">
              <button
                disabled={tracePage === 0}
                onClick={() => setTracePage((p) => p - 1)}
                className="px-3 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-40"
              >Prev</button>
              <button
                disabled={(tracePage + 1) * TRACE_PAGE_SIZE >= traceData.total}
                onClick={() => setTracePage((p) => p + 1)}
                className="px-3 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-40"
              >Next</button>
            </div>
          </div>
        )}
      </div>

      <RequestDetailDrawer
        requestId={selectedRequestId}
        onClose={() => setSelectedRequestId(null)}
      />
    </div>
  );
}
