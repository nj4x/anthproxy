import useSWR from 'swr';
import { api } from '../api/client';
import type { ConfigChangesResponse, ConfigChange } from '../api/types';

export default function AuditLog() {
  const { data, error } = useSWR<ConfigChangesResponse>(
    'config-changes',
    () => api.getConfigChanges(100)
  );

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Audit Log</h2>

      {error ? (
        <div className="bg-red-50 text-red-700 p-4 rounded">{error.message}</div>
      ) : !data ? (
        <div className="text-gray-500">Loading...</div>
      ) : (
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Event</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actor</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actor ID</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Change</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {data.items.map((item: ConfigChange) => (
                  <tr key={item.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap text-gray-600 text-xs">
                      {new Date(item.ts).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 font-mono text-gray-800 text-xs">{item.event_type}</td>
                    <td className="px-4 py-3 text-gray-600 text-xs">{item.actor}</td>
                    <td className="px-4 py-3 font-mono text-gray-500 text-xs truncate max-w-xs" title={item.actor_id}>
                      {item.actor_id}
                    </td>
                    <td className="px-4 py-3 text-xs text-gray-700">
                      {item.prev_value != null && (
                        <span className="text-red-600 line-through mr-1">{item.prev_value}</span>
                      )}
                      {item.new_value != null && (
                        <span className="text-green-700">{item.new_value}</span>
                      )}
                    </td>
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
