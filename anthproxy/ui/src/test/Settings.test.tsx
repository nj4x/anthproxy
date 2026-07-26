import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SWRConfig } from 'swr';

vi.mock('../api/client', () => ({
  api: {
    getConfig: vi.fn(),
    getBackends: vi.fn(),
    setRouting: vi.fn(),
    setBackendPreference: vi.fn(),
  },
}));

import { api } from '../api/client';
import Settings from '../pages/Settings';
import type { BackendsResponse } from '../api/types';

function renderSettings(backendsData: BackendsResponse | undefined) {
  (api.getConfig as ReturnType<typeof vi.fn>).mockResolvedValue({
    active_backend: 'anthropic',
    routing_enabled: true,
    auto_backend_mode: 'auto',
    auto_model_routing_classifier_model: 'haiku',
    auto_model_routing_mode: 'auto',
    auto_model_routing_long_context_threshold: 50000,
    auto_model_routing_affirmation_inherit: true,
  });
  (api.getBackends as ReturnType<typeof vi.fn>).mockResolvedValue(backendsData);
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <Settings />
    </SWRConfig>,
  );
}

describe('Settings backend dropdown', () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('renders modes followed by known backends including plugin-only names', async () => {
    const backends: BackendsResponse = {
      backends: [{ name: 'anthropic', active: true, available: true }],
      active: 'anthropic',
      modes: ['auto', 'subscription'],
      known: ['anthropic', 'bedrock', 'codex', 'local', 'openrouter', 'myplugin'],
    };
    renderSettings(backends);

    // Wait for the select to appear
    const select = await screen.findByRole('combobox');
    const options = Array.from(select.querySelectorAll('option')).map((o) => o.textContent);

    // modes must come first
    expect(options.indexOf('auto')).toBeLessThan(options.indexOf('anthropic'));
    expect(options.indexOf('subscription')).toBeLessThan(options.indexOf('anthropic'));

    // plugin-only name (not in constructed backends list) must appear
    expect(options).toContain('myplugin');
  });

  it('renders without throwing when backendsData is undefined during initial fetch', () => {
    // If SWR has not resolved yet, backendsData is undefined
    // The component must not throw
    expect(() => renderSettings(undefined)).not.toThrow();
  });
});
