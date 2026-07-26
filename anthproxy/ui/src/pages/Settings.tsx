import { useState, useEffect, useRef } from 'react';
import useSWR from 'swr';
import { api } from '../api/client';
import type { Config, BackendsResponse, Backend, SetBackendResponse } from '../api/types';

export default function Settings() {
  const { data: config, error: configError, mutate: mutateConfig } = useSWR<Config>(
    'config',
    () => api.getConfig()
  );

  const { data: backendsData, error: backendsError } = useSWR<BackendsResponse>(
    'backends',
    () => api.getBackends()
  );

  const [routingLoading, setRoutingLoading] = useState(false);
  const [backendPref, setBackendPref] = useState('');
  const [backendSaving, setBackendSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [backendNote, setBackendNote] = useState<string | null>(null);
  const initializedRef = useRef(false);

  // Sync dropdown to current active backend on first config load.
  useEffect(() => {
    if (config && !initializedRef.current) {
      initializedRef.current = true;
      setBackendPref(config.active_backend);
    }
  }, [config]);

  const handleToggleRouting = async () => {
    if (!config) return;
    setRoutingLoading(true);
    setMessage(null);
    setBackendNote(null);
    try {
      await api.setRouting(!config.routing_enabled);
      await mutateConfig();
      setMessage(`Routing ${!config.routing_enabled ? 'enabled' : 'disabled'} successfully`);
    } catch (e) {
      setMessage(`Error: ${e instanceof Error ? e.message : 'unknown error'}`);
    } finally {
      setRoutingLoading(false);
    }
  };

  const handleSaveBackend = async () => {
    setBackendSaving(true);
    setMessage(null);
    setBackendNote(null);
    try {
      const result = await api.setBackendPreference(backendPref) as SetBackendResponse;
      await mutateConfig();
      setMessage(`Backend preference set to ${backendPref}`);
      if (result.auto_selection != null) {
        const hint = backendPref !== 'auto' ? " — select 'auto' to resume" : '';
        setBackendNote(`auto-selection: ${result.auto_selection}${hint}`);
      }
    } catch (e) {
      setMessage(`Error: ${e instanceof Error ? e.message : 'unknown error'}`);
    } finally {
      setBackendSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-gray-800">Settings</h2>

      {message && (
        <div className="bg-blue-50 text-blue-700 px-4 py-2 rounded text-sm">
          {message}
          {backendNote && <div className="mt-0.5 text-xs opacity-75">{backendNote}</div>}
        </div>
      )}

      {/* Routing toggle */}
      <div className="bg-white shadow rounded-lg p-5">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Model Routing</h3>
        {configError ? (
          <div className="bg-red-50 text-red-700 p-4 rounded">{configError.message}</div>
        ) : !config ? (
          <div className="text-gray-500">Loading...</div>
        ) : (
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">
              Status:{' '}
              <span className={config.routing_enabled ? 'text-green-700 font-medium' : 'text-red-700 font-medium'}>
                {config.routing_enabled ? 'ON' : 'OFF'}
              </span>
            </span>
            <button
              onClick={handleToggleRouting}
              disabled={routingLoading}
              className={`px-4 py-2 rounded text-sm font-medium text-white disabled:opacity-50 ${
                config.routing_enabled ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {routingLoading ? 'Saving...' : config.routing_enabled ? 'Disable Routing' : 'Enable Routing'}
            </button>
          </div>
        )}

        {config && (
          <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-gray-600">
            <div><span className="text-gray-400">Classifier model:</span> {config.auto_model_routing_classifier_model}</div>
            <div><span className="text-gray-400">Mode:</span> {config.auto_model_routing_mode}</div>
            <div><span className="text-gray-400">Long context threshold:</span> {config.auto_model_routing_long_context_threshold}</div>
            <div><span className="text-gray-400">Affirmation inherit:</span> {config.auto_model_routing_affirmation_inherit ? 'on' : 'off'}</div>
          </div>
        )}
      </div>

      {/* Backend preference */}
      <div className="bg-white shadow rounded-lg p-5">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Backend Preference</h3>
        <div className="flex items-center gap-3">
          <select
            value={backendPref}
            onChange={(e) => setBackendPref(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1.5 text-sm"
          >
            {[...(backendsData?.modes ?? []), ...(backendsData?.known ?? [])].map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
          <button
            onClick={handleSaveBackend}
            disabled={backendSaving}
            className="px-4 py-2 bg-gray-800 text-white rounded text-sm hover:bg-gray-700 disabled:opacity-50"
          >
            {backendSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
        {config && (
          <p className="mt-2 text-xs text-gray-500">
            Current: <span className="font-medium">{config.active_backend}</span> (mode: {config.auto_backend_mode})
          </p>
        )}
      </div>

      {/* Backend status table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-200 text-sm font-medium text-gray-700">Backend Status</div>
        {backendsError ? (
          <div className="bg-red-50 text-red-700 p-4 rounded">{backendsError.message}</div>
        ) : !backendsData ? (
          <div className="p-4 text-gray-500">Loading...</div>
        ) : (
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Backend</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Active</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Available</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-100">
              {backendsData.backends.map((b: Backend) => (
                <tr key={b.name} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-gray-800">{b.name}</td>
                  <td className="px-4 py-3 text-center">
                    <span className={`inline-block w-2.5 h-2.5 rounded-full ${b.active ? 'bg-green-500' : 'bg-gray-300'}`} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    {b.available === null || b.available === undefined
                      ? <span className="text-xs text-gray-400">—</span>
                      : <span className={`inline-block w-2.5 h-2.5 rounded-full ${b.available ? 'bg-green-500' : 'bg-red-400'}`} />
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
