import { useState, useEffect, useRef } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import useSWR from 'swr';
import { api } from '../api/client';
import type { SessionsResponse } from '../api/types';
import { ExcerptHighlight } from '../components/ExcerptHighlight';
import { parseSessionId } from '../utils';

const PAGE_SIZE = 20;

export default function Sessions() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlQ = searchParams.get('q') ?? '';
  const [inputValue, setInputValue] = useState(urlQ);
  const [offset, setOffset] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync inputValue with URL q on back/forward navigation
  useEffect(() => {
    setInputValue(urlQ);
  }, [urlQ]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInputValue(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setSearchParams(val.trim() ? { q: val.trim() } : {}, { replace: true });
      setOffset(0);
    }, 300);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      const val = inputValue.trim();
      setSearchParams(val ? { q: val } : {}, { replace: true });
      setOffset(0);
    }
  };

  // Reset offset when URL q changes
  useEffect(() => {
    setOffset(0);
  }, [urlQ]);

  const q = urlQ.trim() || undefined;

  const { data, error } = useSWR<SessionsResponse>(
    ['sessions', offset, q],
    () => api.getSessions(PAGE_SIZE, offset, q),
    { refreshInterval: 60000 }
  );

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Sessions</h2>

      {/* Filter input */}
      <div className="mb-4">
        <input
          type="text"
          aria-label="Filter sessions"
          placeholder="Filter sessions by prompt or response…"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {error ? (
        <div className="bg-red-50 text-red-700 p-4 rounded">{error.message}</div>
      ) : data === undefined ? (
        <div className="text-gray-500">Loading...</div>
      ) : (
        <>
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Session ID</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Created</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Last Seen</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wide">Requests</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wide">Cost</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Backend</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Tier</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-100">
                  {data.items.map((s) => {
                    const sid = parseSessionId(s.session_id);
                    const dest = q
                      ? `/sessions/${encodeURIComponent(s.session_id)}?q=${encodeURIComponent(q)}`
                      : `/sessions/${encodeURIComponent(s.session_id)}`;
                    return (
                      <tr key={s.session_id} className="hover:bg-gray-50">
                        <td className="px-4 py-3">
                          <div>
                            <Link
                              to={dest}
                              className="text-blue-600 hover:underline font-mono text-xs"
                              title={sid.tooltip}
                            >
                              {sid.label}
                            </Link>
                            {s.excerpt ? (
                              <ExcerptHighlight excerpt={s.excerpt} />
                            ) : s.summary ? (
                              <p className="mt-0.5 max-w-xs text-xs text-gray-400 truncate" title={s.summary}>
                                {s.summary}
                              </p>
                            ) : null}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-gray-600 text-xs whitespace-nowrap">
                          {new Date(s.created_at).toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-gray-600 text-xs whitespace-nowrap">
                          {new Date(s.last_seen_at).toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-600">{s.request_count}</td>
                        <td className="px-4 py-3 text-right text-gray-600">
                          ${s.estimated_cost_usd.toFixed(2)}
                        </td>
                        <td className="px-4 py-3 text-gray-600 text-xs">
                          {s.pinned_backend ?? <span className="text-gray-400">auto</span>}
                        </td>
                        <td className="px-4 py-3 text-gray-600 text-xs">
                          {s.pinned_tier ?? <span className="text-gray-400">auto</span>}
                        </td>
                      </tr>
                    );
                  })}
                  {data.items.length === 0 && (
                    <tr>
                      <td colSpan={7} className="px-4 py-6 text-center text-gray-400 text-sm">
                        {q ? `No sessions match "${q}"` : 'No sessions yet'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-4">
            <span className="text-sm text-gray-600">
              Page {currentPage} of {totalPages} ({data.total} total)
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0}
                className="px-3 py-1 text-sm rounded border border-gray-300 disabled:opacity-40 hover:bg-gray-100"
              >
                Prev
              </button>
              <button
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= data.total}
                className="px-3 py-1 text-sm rounded border border-gray-300 disabled:opacity-40 hover:bg-gray-100"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
