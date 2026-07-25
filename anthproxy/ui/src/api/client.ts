const BASE = '';

async function apiFetch(path: string, options?: RequestInit) {
  const res = await fetch(BASE + path, options);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export const api = {
  getSessions: (limit = 20, offset = 0, q?: string) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (q) params.set('q', q);
    return apiFetch(`/admin/sessions?${params}`);
  },
  getSession: (id: string) =>
    apiFetch(`/admin/sessions/${encodeURIComponent(id)}`),
  getTrace: (id: string, anchor?: string, limit = 100, offset = 0, q?: string) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (anchor) params.set('anchor', anchor);
    if (q) params.set('q', q);
    return apiFetch(`/admin/sessions/${encodeURIComponent(id)}/trace?${params}`);
  },
  getCost: (groupBy = 'model', timeRange = '7d', sessionId?: string) =>
    apiFetch(`/admin/cost?group_by=${groupBy}&time_range=${timeRange}${sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ''}`),
  getRouting: (timeRange = '7d') =>
    apiFetch(`/admin/routing?time_range=${timeRange}`),
  getBackends: () => apiFetch('/admin/backends'),
  getConfig: () => apiFetch('/admin/config'),
  getConfigChanges: (limit = 100) =>
    apiFetch(`/admin/config-changes?limit=${limit}`),
  setSessionBackend: (id: string, backend: string | null) =>
    apiFetch(`/admin/sessions/${encodeURIComponent(id)}/set-backend`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend }),
    }),
  setSessionTier: (id: string, tier: string | null) =>
    apiFetch(`/admin/sessions/${encodeURIComponent(id)}/set-global-tier`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tier }),
    }),
  setRouting: (enabled: boolean) =>
    apiFetch('/admin/global/routing', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    }),
  setBackendPreference: (prefer: string) =>
    apiFetch('/admin/global/backend', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prefer }),
    }),
  getStatus: () => apiFetch('/admin/status'),
  getStats: (period: string, backend?: string) => {
    const params = new URLSearchParams({ period });
    if (backend) params.set('backend', backend);
    return apiFetch(`/admin/stats?${params}`);
  },
  getRequest: (id: number) => apiFetch(`/admin/requests/${id}`),
  getSessionSummary: (id: string) => apiFetch(`/admin/sessions/${encodeURIComponent(id)}/summary`),
};
